"""
OpenVLA 简化模型推理测试脚本
测试训练好的模型是否能正确进行推理
"""

import os
import yaml
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from data_loader import LIBEROCleanDataset

class SimpleVLA(nn.Module):
    """简化的 VLA 模型（与训练脚本匹配）"""
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
        
        # 简化的指令编码器
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
        batch_size = pixel_values.shape[0]
        seq_len = pixel_values.shape[1]
        
        visual_features = []
        for i in range(seq_len):
            img = pixel_values[:, i]
            feat = self.visual_encoder(img)
            feat = feat.view(batch_size, -1)
            visual_features.append(feat)
        
        visual_feat = visual_features[-1]
        text_feat = torch.randn(batch_size, 256).to(visual_feat.device)
        text_feat = self.text_proj(text_feat)
        
        fused = torch.cat([visual_feat, text_feat], dim=1)
        fused = self.fusion(fused)
        action_pred = self.action_head(fused)
        
        if labels is not None:
            loss = nn.MSELoss()(action_pred, labels)
            return SimpleOutput(loss=loss, logits=action_pred)
        
        return SimpleOutput(logits=action_pred)

class SimpleOutput:
    def __init__(self, loss=None, logits=None):
        self.loss = loss
        self.logits = logits

# 加载配置
config_path = 'configs/simple_config.yaml'
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 加载测试数据集
test_dataset = LIBEROCleanDataset(subset="test", data_root=config['test_data_dir'])
print(f"测试集样本数: {len(test_dataset)}")

# 加载模型
model_path = 'runs/openvla_simple_20260612_201725/final_model.pt'
if os.path.exists(model_path):
    model = SimpleVLA(num_actions=7).to(device)
    model_weights = torch.load(model_path, map_location=device)
    model.load_state_dict(model_weights)
    model.eval()
    print("模型加载成功！")
else:
    print(f"错误：模型文件不存在: {model_path}")
    exit(1)

# 测试推理
num_test_samples = 5
print(f"\n--- 测试 {num_test_samples} 个样本 ---")

for i in range(min(num_test_samples, len(test_dataset))):
    sample = test_dataset[i]
    
    # 准备输入
    images = sample['images']
    instruction = sample['instruction']
    task_name = sample['task_name']
    demo_name = sample['demo_name']
    
    # 取最后一张图像
    if len(images) > 0:
        img = images[-1]
        img_tensor = torch.tensor(np.array(img)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).permute(0, 3, 1, 2).to(device)  # (1, 3, H, W)
        seq_images = img_tensor.unsqueeze(1)  # (1, 1, 3, H, W)
    else:
        seq_images = torch.zeros(1, 1, 3, 224, 224).to(device)
    
    # 推理
    with torch.no_grad():
        output = model(pixel_values=seq_images, text=[instruction])
        action_pred = output.logits.cpu().numpy()[0]
    
    # 真实动作（取最后一个）
    actions = sample['actions']
    if len(actions) > 0:
        action_true = actions[-1]
    else:
        action_true = np.zeros(7)
    
    print(f"\n样本 {i+1}:")
    print(f"  任务: {task_name}")
    print(f"  Demo: {demo_name}")
    print(f"  指令: {instruction}")
    print(f"  预测动作: {action_pred}")
    print(f"  真实动作: {action_true}")
    print(f"  MSE误差: {np.mean((action_pred - action_true)**2):.6f}")

print("\n--- 推理测试完成 ---")

# 计算整体测试集指标
print("\n--- 计算测试集指标 ---")
total_mse = 0.0
count = 0

for i in range(len(test_dataset)):
    sample = test_dataset[i]
    
    images = sample['images']
    instruction = sample['instruction']
    
    if len(images) > 0:
        img = images[-1]
        img_tensor = torch.tensor(np.array(img)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).permute(0, 3, 1, 2).to(device)
        seq_images = img_tensor.unsqueeze(1)
    else:
        seq_images = torch.zeros(1, 1, 3, 224, 224).to(device)
    
    with torch.no_grad():
        output = model(pixel_values=seq_images, text=[instruction])
        action_pred = output.logits.cpu().numpy()[0]
    
    actions = sample['actions']
    if len(actions) > 0:
        action_true = actions[-1]
        total_mse += np.mean((action_pred - action_true)**2)
        count += 1

if count > 0:
    avg_mse = total_mse / count
    print(f"测试集平均 MSE: {avg_mse:.6f}")
else:
    print("没有有效的测试样本")

print("\n测试完成！")
