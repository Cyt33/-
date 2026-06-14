"""
OpenVLA 简化训练脚本 - 用于验证训练流程

功能：
- 验证数据加载是否正常
- 验证训练循环是否正常
- 使用简化模型进行测试训练
"""

import os
import yaml
import numpy as np
import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

from data_loader import LIBEROCleanDataset

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="OpenVLA Simple Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    return parser.parse_args()


def load_config(config_path):
    """加载训练配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


class SimpleVLA(nn.Module):
    """简化的 VLA 模型用于测试"""
    
    def __init__(self, image_size=224, num_actions=7):
        super().__init__()
        
        # 简化的视觉编码器
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 简化的指令编码器（使用嵌入层模拟）
        self.text_embedding = nn.Embedding(1000, 256)
        self.text_proj = nn.Linear(256, 256)
        
        # 融合层
        self.fusion = nn.Linear(256 + 256, 512)
        
        # 动作预测头
        self.action_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
    
    def forward(self, pixel_values, text, labels=None):
        """
        Args:
            pixel_values: (batch, seq_len, 3, H, W) - 图像序列
            text: 指令文本（这里简化处理）
            labels: 动作标签 (batch, 7)
        """
        batch_size = pixel_values.shape[0]
        seq_len = pixel_values.shape[1]
        
        # 处理图像序列
        visual_features = []
        for i in range(seq_len):
            img = pixel_values[:, i]  # (batch, 3, H, W)
            feat = self.visual_encoder(img)  # (batch, 256, 1, 1)
            feat = feat.view(batch_size, -1)  # (batch, 256)
            visual_features.append(feat)
        
        # 取最后一帧作为视觉特征
        visual_feat = visual_features[-1]  # (batch, 256)
        
        # 文本编码（简化处理）
        text_feat = torch.randn(batch_size, 256).to(visual_feat.device)
        text_feat = self.text_proj(text_feat)
        
        # 融合
        fused = torch.cat([visual_feat, text_feat], dim=1)
        fused = self.fusion(fused)
        
        # 预测动作
        action_pred = self.action_head(fused)
        
        # 损失计算
        if labels is not None:
            # labels: (batch, 7)
            loss = nn.MSELoss()(action_pred, labels)
            return SimpleOutput(loss=loss, logits=action_pred)
        
        return SimpleOutput(logits=action_pred)


class SimpleOutput:
    """简化的输出类"""
    def __init__(self, loss=None, logits=None):
        self.loss = loss
        self.logits = logits


def train_epoch(model, dataloader, optimizer, scheduler, epoch, config, writer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # 数据准备
        images = batch['images']  # list of lists of PIL Images
        actions = batch['actions']  # (batch, 7)
        
        # 将图像转换为 tensor
        if isinstance(images, list):
            # 取最后一张图像并转换为 tensor
            last_images = []
            for img_list in images:
                if len(img_list) > 0:
                    img = img_list[-1]  # PIL Image
                    # 转换为 tensor
                    img_tensor = torch.tensor(np.array(img)).float() / 255.0
                    last_images.append(img_tensor)
                else:
                    # 空序列，创建零张量
                    last_images.append(torch.zeros(224, 224, 3))
            
            images_tensor = torch.stack(last_images).to(device)
            images_tensor = images_tensor.permute(0, 3, 1, 2)  # (batch, 3, H, W)
        else:
            images_tensor = images.to(device)
        
        actions = actions.to(device)
        instructions = batch['instructions']
        
        # 构建序列输入
        # 这里简化处理，只取单帧
        seq_images = images_tensor.unsqueeze(1)  # (batch, 1, 3, H, W)
        
        # 前向传播
        outputs = model(
            pixel_values=seq_images,
            text=instructions,
            labels=actions
        )
        
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
            lr = scheduler.get_last_lr()[0]
            logger.info(f"Epoch [{epoch}/{config['num_epochs']}] "
                        f"Step [{batch_idx}/{len(dataloader)}] "
                        f"Loss: {avg_loss:.6f} "
                        f"LR: {lr:.6f}")
            
            # TensorBoard 记录
            writer.add_scalar('Train/Loss', avg_loss, global_step)
            writer.add_scalar('Train/LearningRate', lr, global_step)
    
    return total_loss / num_batches


def validate(model, dataloader, epoch, config, writer, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['images']
            actions = batch['actions'].to(device)  # (batch, 7)
            
            if isinstance(images, list):
                # 取最后一张图像并转换为 tensor
                last_images = []
                for img_list in images:
                    if len(img_list) > 0:
                        img = img_list[-1]  # PIL Image
                        img_tensor = torch.tensor(np.array(img)).float() / 255.0
                        last_images.append(img_tensor)
                    else:
                        last_images.append(torch.zeros(224, 224, 3))
                
                images_tensor = torch.stack(last_images).to(device)
                images_tensor = images_tensor.permute(0, 3, 1, 2)  # (batch, 3, H, W)
            else:
                images_tensor = images.to(device)
            
            seq_images = images_tensor.unsqueeze(1)
            instructions = batch['instructions']
            
            outputs = model(
                pixel_values=seq_images,
                text=instructions,
                labels=actions
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    logger.info(f"Validation Epoch [{epoch}/{config['num_epochs']}] Loss: {avg_loss:.6f}")
    
    # TensorBoard 记录
    writer.add_scalar('Val/Loss', avg_loss, epoch)
    
    return avg_loss


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # 创建输出目录
    run_name = f"openvla_simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(config['output_dir']) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")
    
    # TensorBoard Writer
    writer = SummaryWriter(log_dir=str(output_dir / 'tensorboard'))
    
    # 构建数据集和数据加载器
    logger.info("Loading datasets...")
    train_dataset = LIBEROCleanDataset(
        subset='train', 
        data_root=config['train_data_dir']
    )
    val_dataset = LIBEROCleanDataset(
        subset='val', 
        data_root=config['val_data_dir']
    )
    
    logger.info(f"训练集样本数: {len(train_dataset)}")
    logger.info(f"验证集样本数: {len(val_dataset)}")
    
    def collate_fn(batch):
        """自定义 collate 函数处理变长序列"""
        images = [item['images'] for item in batch]
        
        # 处理变长动作序列 - 取最后一个动作
        actions = []
        for item in batch:
            action_array = item['actions']
            if len(action_array) > 0:
                # 取最后一个动作
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
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 构建模型
    logger.info("Building model...")
    model = SimpleVLA(
        image_size=config['image_size'],
        num_actions=7
    ).to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数总数: {total_params:,}")
    logger.info(f"可训练参数: {trainable_params:,}")
    
    # 构建优化器和调度器
    optimizer = AdamW(
        model.parameters(),
        lr=float(config['learning_rate']),
        weight_decay=float(config['weight_decay'])
    )
    
    num_training_steps = config['num_epochs'] * len(train_loader)
    scheduler = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.1,
        total_iters=num_training_steps
    )
    
    # 训练循环
    logger.info(f"Starting training for {config['num_epochs']} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, epoch, config, writer, device)
        
        # 验证
        val_loss = validate(model, val_loader, epoch, config, writer, device)
        
        # 保存检查点
        if epoch % config['save_interval'] == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = output_dir / "best_model.pt"
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Best model saved to {best_model_path}")
    
    # 保存最终模型
    final_model_path = output_dir / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Training completed! Final model saved to {final_model_path}")
    
    writer.close()
    
    # 保存训练配置
    config_path = output_dir / "config.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)
    logger.info(f"配置文件保存到 {config_path}")


if __name__ == "__main__":
    main()
