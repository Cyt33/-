
# OpenVLA 项目复现

OpenVLA（Open Vision-Language-Action）是一个开源的视觉-语言-动作模型，用于机器人操作任务的端到端学习。本项目实现了 OpenVLA 的完整复现，包括环境搭建、数据预处理、模型训练、推理测试和结果分析。

---

## 📁 项目结构

```
OpenVLA/
├── data/                    # 数据集目录
│   ├── modified_libero_rlds/   # LIBERO 原始数据集
│   └── processed_libero/       # 预处理后数据集
├── models/                  # 模型目录
│   └── openvla-7b/             # OpenVLA 预训练模型
├── scripts/                 # 自定义脚本
├── vla-scripts/             # 官方脚本
├── configs/                 # 配置文件
│   └── train_config.yaml       # 训练配置
├── visualizations/          # 可视化结果
├── results/                 # 评估结果
├── runs/                    # 训练日志和模型
├── README.md                # 项目说明
├── environment.yml          # Conda 环境配置
├── requirements.txt         # 依赖列表
├── demo.py                  # 环境验证脚本
└── .env                     # 环境变量配置
```

---

## 🚀 快速开始

### 1. 环境搭建

```bash
# 创建并激活虚拟环境
conda env create -f environment.yml
conda activate openvla

# 或手动安装依赖
pip install -r requirements.txt
```

### 2. 下载数据集

```bash
python download_libero.py
```

### 3. 数据预处理

```bash
python preprocess_libero.py
```

### 4. 模型训练

```bash
# 单 GPU 训练
python train_lora.py --config configs/train_config.yaml

# 多 GPU 训练
torchrun --standalone --nnodes 1 --nproc-per-node 2 train_lora.py \
    --config configs/train_config.yaml

# 断点续传
python train_lora.py --config configs/train_config.yaml \
    --resume runs/openvla_lora_xxx/checkpoint_epoch_5.pt
```

### 5. 模型评估

```bash
# 基础评估
python evaluate.py --model runs/best_model.pt --data ./data/processed_libero/test

# 对比评估
python evaluate.py --model runs/best_model.pt --baseline openvla/openvla-7b \
    --data ./data/processed_libero/test
```

### 6. 结果可视化

```bash
# 可视化训练曲线
python visualize_results.py --log_dir runs/openvla_lora_xxx/tensorboard

# 可视化评估结果
python visualize_results.py --metrics results/eval_xxx/metrics.json
```

---

## 📊 团队分工

| 组员 | 负责模块 | 文件 |
|------|----------|------|
| **组员一** | 环境搭建 + 数据集处理 | `download_libero.py`, `preprocess_libero.py`, `data_loader.py`, `环境配置说明.md`, `数据集说明.md` |
| **组员二** | 模型架构 + 核心算法复现 | `openvla_inference.py`, `openvla_quick_demo.py`, `OpenVLA_模型架构详细分析报告.md`, `OpenVLA_架构图.md` |
| **组员三** | 模型训练 + 微调调参 | `train_lora.py`, `configs/train_config.yaml`, `utils.py`, `组员三_模型训练与微调调参.md` |
| **组员四** | 实验测试 + 结果可视化 + 报告统筹 | `evaluate.py`, `visualize_results.py`, `组员四_实验测试与结果可视化.md`, `项目报告模板.md` |

---

## 📝 核心文件说明

### 训练相关

| 文件 | 说明 |
|------|------|
| `train_lora.py` | LoRA 微调训练脚本 |
| `configs/train_config.yaml` | 训练配置文件 |
| `utils.py` | 训练工具函数 |
| `vla-scripts/finetune.py` | 官方训练脚本 |

### 数据处理

| 文件 | 说明 |
|------|------|
| `download_libero.py` | 数据集下载脚本 |
| `preprocess_libero.py` | 数据预处理脚本 |
| `data_loader.py` | 数据加载器 |

### 推理与评估

| 文件 | 说明 |
|------|------|
| `openvla_inference.py` | 推理模块 |
| `openvla_quick_demo.py` | 快速演示 |
| `evaluate.py` | 评估脚本 |
| `visualize_results.py` | 结果可视化 |

### 文档

| 文件 | 说明 |
|------|------|
| `环境配置说明.md` | 环境搭建指南 |
| `数据集说明.md` | 数据集使用说明 |
| `OpenVLA_模型架构详细分析报告.md` | 模型架构分析 |
| `OpenVLA_架构图.md` | 模型架构图 |
| `组员三_模型训练与微调调参.md` | 训练模块文档 |
| `组员四_实验测试与结果可视化.md` | 测试模块文档 |
| `项目报告模板.md` | 项目报告模板 |

---

## ⚙️ 配置说明

### 训练配置 (`configs/train_config.yaml`)

```yaml
# 模型配置
model_name: "openvla/openvla-7b"
use_quantization: true
load_in_8bit: true

# LoRA 配置
use_lora: true
lora_rank: 32
lora_alpha: 64
target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

# 训练配置
batch_size: 8
learning_rate: 5e-4
num_epochs: 10
save_interval: 1
```

### 环境变量 (`.env`)

```bash
HF_ENDPOINT=https://hf-mirror.com
PYTHONPATH=/path/to/project
CUDA_VISIBLE_DEVICES=0
```

---

## 📈 预期结果

### 训练指标

| 指标 | 预期值 |
|------|--------|
| 训练损失 | < 1.0 |
| 验证损失 | < 1.0 |
| 任务成功率 | > 70% |

### 推理性能

| 指标 | 预期值 |
|------|--------|
| 推理耗时 | < 200ms/样本 |
| 显存占用 | < 16GB |

---

## 🛠️ 常见问题

### Q1: CUDA out of memory

**解决方案**：
- 减小 `batch_size`（建议 4-8）
- 使用 8-bit/4-bit 量化
- 启用梯度累积

### Q2: 模型下载慢

**解决方案**：
- 设置环境变量 `HF_ENDPOINT=https://hf-mirror.com`
- 手动下载模型并放置到 `models/openvla-7b/`

### Q3: 依赖版本冲突

**解决方案**：
- 使用 `pip install --force-reinstall` 重新安装
- 参考 `environment.yml` 中的版本号

---

## 📚 参考文献

- OpenVLA: An Open-Source Vision-Language-Action Model for Robotics
- LoRA: Low-Rank Adaptation of Large Language Models
- LIBERO: A Benchmark for Long-Horizon Robot Manipulation

---

## 📧 联系方式

如有问题，请联系项目团队成员。
