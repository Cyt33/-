
"""
OpenVLA 训练工具函数
"""

import os
import json
import logging
import torch


def setup_logging(log_level="INFO", log_file="training.log"):
    """设置日志配置"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def save_checkpoint(path, model, optimizer, scheduler, epoch, val_loss):
    """保存训练检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer, scheduler):
    """加载训练检查点"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return model, optimizer, scheduler, checkpoint['epoch'], checkpoint['val_loss']


def save_training_config(config, output_dir):
    """保存训练配置"""
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)


def save_training_metrics(metrics, output_dir):
    """保存训练指标"""
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)


def count_trainable_parameters(model):
    """统计可训练参数数量"""
    trainable_params = 0
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return trainable_params, total_params


def format_params(num_params):
    """格式化参数数量显示"""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    return str(num_params)


def log_model_info(model):
    """打印模型信息"""
    trainable_params, total_params = count_trainable_parameters(model)
    logger = logging.getLogger(__name__)
    logger.info(f"Total parameters: {format_params(total_params)}")
    logger.info(f"Trainable parameters: {format_params(trainable_params)}")
    logger.info(f"Trainable ratio: {(trainable_params / total_params * 100):.2f}%")
