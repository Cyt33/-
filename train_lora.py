"""
OpenVLA LoRA 微调训练脚本

功能：
- 基于 PEFT 库实现低秩适配微调
- 支持断点续传和模型保存
- 记录训练日志和损失曲线
- 支持 TensorBoard 可视化
- 支持分布式训练

运行命令：
    # 单 GPU 训练
    python train_lora.py --config configs/train_config.yaml
    
    # 多 GPU 训练
    torchrun --standalone --nnodes 1 --nproc-per-node 2 train_lora.py --config configs/train_config.yaml
    
    # 断点续传
    python train_lora.py --config configs/train_config.yaml --resume runs/openvla_lora_xxx/checkpoint_epoch_5.pt
"""

import os
import json
import yaml
import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import (
    AutoModel,
    AutoProcessor,
    BitsAndBytesConfig,
    get_scheduler,
    set_seed
)
from torch.optim import AdamW

from data_loader import LIBEROCleanDataset, find_data_root, download_libero_dataset
from utils import setup_logging, save_checkpoint, load_checkpoint

# 设置日志
setup_logging()
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="OpenVLA LoRA Fine-tuning")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def load_config(config_path):
    """加载训练配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def collate_fn(batch):
    """自定义 collate 函数处理变长序列"""
    images = [item['images'] for item in batch]
    
    # 处理变长动作序列 - 取最后一个动作
    actions = []
    for item in batch:
        action_array = item['actions']
        if len(action_array) > 0:
            last_action = torch.from_numpy(action_array[-1:])  # (1, 7)
        else:
            last_action = torch.zeros(1, 7)
        actions.append(last_action)
    
    actions = torch.cat(actions, dim=0)  # (batch_size, 7)
    
    instructions = [item['instruction'] for item in batch]
    task_names = [item['task_name'] for item in batch]
    demo_names = [item['demo_name'] for item in batch]
    
    return {
        'images': images,
        'actions': actions,
        'instructions': instructions,
        'task_names': task_names,
        'demo_names': demo_names
    }


def build_model(config):
    """构建模型并配置 LoRA"""
    logger.info(f"Loading model: {config['model_name']}")
    
    # 配置量化
    if config.get('use_quantization', False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=config.get('load_in_4bit', False),
            load_in_8bit=config.get('load_in_8bit', True),
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    else:
        quantization_config = None
    
    # 加载模型 - 使用 AutoModel 兼容 OpenVLA
    model = AutoModel.from_pretrained(
        config['model_name'],
        quantization_config=quantization_config,
        device_map='auto',
        torch_dtype=torch.float16,
        trust_remote_code=True,
        cache_dir=config.get('cache_dir', None)
    )
    
    # 配置 LoRA
    if config['use_lora']:
        logger.info("Configuring LoRA...")
        lora_config = LoraConfig(
            r=config['lora_rank'],
            lora_alpha=config['lora_alpha'],
            target_modules=config['target_modules'],
            lora_dropout=config['lora_dropout'],
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        if config.get('use_quantization', False):
            model = prepare_model_for_kbit_training(model)
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model


def build_processor(config):
    """构建数据处理器"""
    processor = AutoProcessor.from_pretrained(
        config['model_name'],
        trust_remote_code=True,
        cache_dir=config.get('cache_dir', None)
    )
    return processor


def build_optimizer(model, config):
    """构建优化器"""
    optimizer = AdamW(
        model.parameters(),
        lr=float(config['learning_rate']),
        weight_decay=float(config['weight_decay'])
    )
    return optimizer


def build_scheduler(optimizer, config, num_training_steps):
    """构建学习率调度器"""
    scheduler = get_scheduler(
        name=config['scheduler_type'],
        optimizer=optimizer,
        num_warmup_steps=int(config['warmup_steps']),
        num_training_steps=num_training_steps
    )
    return scheduler


def train_epoch(model, dataloader, processor, optimizer, scheduler, epoch, config, writer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # 数据准备
        instructions = batch['instructions']
        actions = batch['actions'].to(device, dtype=torch.float16)
        
        # 处理图像
        images = batch['images']
        if isinstance(images, list):
            pixel_values = []
            for img in images:
                if isinstance(img, torch.Tensor):
                    pixel_values.append(img)
                else:
                    # 假设是 numpy array
                    pixel_values.append(torch.from_numpy(img))
            pixel_values = torch.stack(pixel_values).to(device, dtype=torch.float16)
        else:
            pixel_values = images.to(device, dtype=torch.float16)
        
        # 前向传播
        try:
            outputs = model(
                pixel_values=pixel_values,
                text=instructions,
                labels=actions
            )
        except Exception as e:
            logger.error(f"Forward pass error: {e}")
            continue
        
        loss = outputs.loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        if config.get('gradient_clipping', None):
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config['gradient_clipping']))
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # 日志记录
        global_step = epoch * len(dataloader) + batch_idx
        if global_step % config['log_interval'] == 0:
            avg_loss = total_loss / num_batches
            logger.info(f"Epoch [{epoch}/{config['num_epochs']}] "
                        f"Step [{batch_idx}/{len(dataloader)}] "
                        f"Loss: {avg_loss:.4f} "
                        f"LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # TensorBoard 记录
            writer.add_scalar('Train/Loss', avg_loss, global_step)
            writer.add_scalar('Train/LearningRate', scheduler.get_last_lr()[0], global_step)
    
    return total_loss / num_batches


def validate(model, dataloader, processor, epoch, config, writer, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            instructions = batch['instructions']
            actions = batch['actions'].to(device, dtype=torch.float16)
            
            # 处理图像
            images = batch['images']
            if isinstance(images, list):
                pixel_values = []
                for img in images:
                    if isinstance(img, torch.Tensor):
                        pixel_values.append(img)
                    else:
                        pixel_values.append(torch.from_numpy(img))
                pixel_values = torch.stack(pixel_values).to(device, dtype=torch.float16)
            else:
                pixel_values = images.to(device, dtype=torch.float16)
            
            try:
                outputs = model(
                    pixel_values=pixel_values,
                    text=instructions,
                    labels=actions
                )
            except Exception as e:
                logger.error(f"Validation error: {e}")
                continue
            
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    logger.info(f"Validation Epoch [{epoch}/{config['num_epochs']}] Loss: {avg_loss:.4f}")
    
    # TensorBoard 记录
    writer.add_scalar('Val/Loss', avg_loss, epoch)
    
    return avg_loss


def test(model, dataloader, processor, config, device):
    """测试模型"""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            instructions = batch['instructions']
            actions = batch['actions'].to(device, dtype=torch.float16)
            
            # 处理图像
            images = batch['images']
            if isinstance(images, list):
                pixel_values = []
                for img in images:
                    if isinstance(img, torch.Tensor):
                        pixel_values.append(img)
                    else:
                        pixel_values.append(torch.from_numpy(img))
                pixel_values = torch.stack(pixel_values).to(device, dtype=torch.float16)
            else:
                pixel_values = images.to(device, dtype=torch.float16)
            
            try:
                outputs = model(
                    pixel_values=pixel_values,
                    text=instructions,
                    labels=actions
                )
                
                # 计算 MSE
                pred_actions = outputs.logits if hasattr(outputs, 'logits') else outputs.prediction
                mse = nn.MSELoss()(pred_actions, actions)
                total_mse += mse.item()
                total_loss += outputs.loss.item()
                num_batches += 1
            except Exception as e:
                logger.error(f"Test error: {e}")
                continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    avg_mse = total_mse / num_batches if num_batches > 0 else float('inf')
    
    logger.info(f"Test Results - Loss: {avg_loss:.4f}, MSE: {avg_mse:.6f}")
    
    return avg_loss, avg_mse


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if device.type == 'cpu':
        logger.warning("WARNING: Running on CPU! This will be very slow. Consider using GPU.")
    
    # 创建输出目录
    run_name = f"openvla_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(config['output_dir']) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置文件
    with open(output_dir / 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # TensorBoard Writer
    writer = SummaryWriter(log_dir=str(output_dir / 'tensorboard'))
    
    # 检查数据集
    logger.info("Checking dataset availability...")
    data_root = find_data_root()
    
    if data_root is None:
        logger.warning("Dataset not found! Attempting to download...")
        download_success = download_libero_dataset()
        if not download_success:
            logger.error("Failed to download dataset! Please manually download and place in data_cleaned/")
            raise RuntimeError("Dataset not found and download failed")
        data_root = "./data_cleaned"
    
    logger.info(f"Using dataset from: {data_root}")
    
    # 构建数据集和数据加载器
    logger.info("Loading datasets...")
    train_dataset = LIBEROCleanDataset(
        subset='train', 
        data_root=data_root
    )
    val_dataset = LIBEROCleanDataset(
        subset='val', 
        data_root=data_root
    )
    test_dataset = LIBEROCleanDataset(
        subset='test', 
        data_root=data_root
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # 构建处理器
    processor = build_processor(config)
    
    # 构建模型
    model = build_model(config)
    model.to(device)
    
    # 构建优化器和调度器
    optimizer = build_optimizer(model, config)
    num_training_steps = config['num_epochs'] * len(train_loader)
    scheduler = build_scheduler(optimizer, config, num_training_steps)
    
    # 断点续传
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        model, optimizer, scheduler, start_epoch, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, scheduler
        )
        logger.info(f"Resumed from checkpoint: {args.resume} (epoch {start_epoch})")
    
    # 训练循环
    logger.info(f"Starting training from epoch {start_epoch}")
    training_metrics = {
        'train_losses': [],
        'val_losses': [],
        'best_val_loss': float('inf'),
        'best_epoch': 0
    }
    
    for epoch in range(start_epoch, config['num_epochs']):
        # 训练
        train_loss = train_epoch(model, train_loader, processor, optimizer, scheduler, epoch, config, writer, device)
        training_metrics['train_losses'].append(train_loss)
        
        # 验证
        val_loss = validate(model, val_loader, processor, epoch, config, writer, device)
        training_metrics['val_losses'].append(val_loss)
        
        # 保存检查点
        if epoch % config['save_interval'] == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, epoch, val_loss)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            training_metrics['best_val_loss'] = best_val_loss
            training_metrics['best_epoch'] = epoch
            best_model_path = output_dir / "best_model.pt"
            model.save_pretrained(str(best_model_path))
            logger.info(f"Best model saved to {best_model_path}")
    
    # 测试最终模型
    logger.info("Running final evaluation on test set...")
    test_loss, test_mse = test(model, test_loader, processor, config, device)
    
    # 保存最终模型
    final_model_path = output_dir / "final_model.pt"
    model.save_pretrained(str(final_model_path))
    logger.info(f"Training completed! Final model saved to {final_model_path}")
    
    # 保存训练指标
    training_metrics['test_loss'] = test_loss
    training_metrics['test_mse'] = test_mse
    with open(output_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(training_metrics, f, indent=4)
    
    writer.close()
    
    # 输出总结
    logger.info("="*50)
    logger.info("Training Summary")
    logger.info("="*50)
    logger.info(f"Best Validation Loss: {best_val_loss:.4f} (Epoch {training_metrics['best_epoch']})")
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test MSE: {test_mse:.6f}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info("="*50)


if __name__ == "__main__":
    main()
