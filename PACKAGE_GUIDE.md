# OpenVLA 项目打包指南

## 📦 项目结构

```
openvla_project/
├── 📁 configs/                    # 配置文件
│   ├── train_config.yaml          # LoRA 训练配置
│   └── simple_config.yaml         # 简化模型配置
├── 📁 data_cleaned/               # 清洗后的 LIBERO 数据集
├── 📁 runs/                       # 训练结果（可选）
├── 📄 train_lora.py               # 完整 OpenVLA + LoRA 训练脚本
├── 📄 train_simple.py             # 简化模型训练脚本
├── 📄 data_loader.py              # 数据加载器
├── 📄 utils.py                    # 工具函数
├── 📄 evaluate.py                 # 测试评估脚本
├── 📄 requirements.txt            # 依赖列表
├── 📄 environment.yml             # Conda 环境配置
└── 📄 PACKAGE_GUIDE.md            # 本指南
```

---

## 🚀 部署到 GPU 环境

### 1. 环境要求

| 项目 | 要求 |
|------|------|
| GPU | NVIDIA GPU（至少 8GB 显存） |
| CUDA | CUDA 11.8+ |
| Python | 3.10+ |
| 网络 | 可访问 HuggingFace |

### 2. 安装步骤

```bash
# 1. 克隆或解压项目
unzip openvla_project.zip
cd openvla_project

# 2. 创建 Conda 环境
conda env create -f environment.yml
conda activate trae-openvla

# 或使用 pip
pip install -r requirements.txt

# 3. 安装额外依赖（如需）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate peft datasets bitsandbytes timm
```

### 3. 运行训练

```bash
# 完整 OpenVLA + LoRA 训练
python train_lora.py --config configs/train_config.yaml

# 简化模型训练（用于测试）
python train_simple.py --config configs/simple_config.yaml

# 断点续传
python train_lora.py --config configs/train_config.yaml --resume runs/openvla_lora_xxx/checkpoint_epoch_5.pt
```

### 4. 运行测试

```bash
# 评估模型
python evaluate.py --model runs/openvla_lora_xxx/best_model.pt

# 加载测试
python load_and_test.py
```

---

## ⚙️ 配置说明

### train_config.yaml 关键参数

```yaml
# 模型配置
model_name: "openvla/openvla-7b"    # HuggingFace 模型名称
use_quantization: true              # 使用 8bit/4bit 量化
load_in_8bit: true                  # 8bit 量化（显存占用约 8GB）

# LoRA 配置
use_lora: true
lora_rank: 32                       # LoRA 秩（越小显存占用越少）
lora_alpha: 64
target_modules:                     # 目标模块
  - "q_proj"
  - "v_proj"
  - "k_proj"
  - "o_proj"

# 训练配置
batch_size: 8                       # 批次大小（根据显存调整）
num_epochs: 10                      # 训练轮数
learning_rate: 5e-4                 # 学习率
```

---

## 📊 预期输出

训练完成后，输出目录结构：

```
runs/openvla_lora_YYYYMMDD_HHMMSS/
├── tensorboard/                    # TensorBoard 日志
├── checkpoint_epoch_0.pt           # 检查点
├── checkpoint_epoch_1.pt
├── best_model.pt                   # 最佳模型
├── final_model.pt                  # 最终模型
├── config.yaml                     # 配置备份
└── metrics.json                    # 训练指标
```

---

## 🔧 常见问题

### Q: 显存不足
```bash
# 解决方案：减小 batch_size 或使用 4bit 量化
# 修改 configs/train_config.yaml
batch_size: 4
load_in_8bit: false
load_in_4bit: true
```

### Q: 模型下载失败
```bash
# 确保网络能访问 HuggingFace
ping huggingface.co

# 如果不行，手动下载模型后修改配置
model_name: "./local_openvla_model"
```

### Q: 训练速度慢
```bash
# 检查是否使用了 GPU
nvidia-smi

# 确保使用 CUDA 版本的 PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 📝 项目文件清单

| 文件 | 大小 | 说明 |
|------|------|------|
| train_lora.py | ~50KB | 完整训练脚本 |
| train_simple.py | ~30KB | 简化训练脚本 |
| data_loader.py | ~20KB | 数据加载器 |
| utils.py | ~5KB | 工具函数 |
| configs/ | ~1KB | 配置文件 |
| data_cleaned/ | ~XX GB | 数据集（按需包含） |
| requirements.txt | ~1KB | 依赖列表 |

---

## 📧 技术支持

如有问题，请检查：
1. CUDA 版本是否正确
2. 依赖是否完整安装
3. 网络是否能访问 HuggingFace
4. 显存是否足够

---

*Last updated: 2024*
