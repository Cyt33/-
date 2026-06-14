# data_cleaned 数据集使用说明

## 概述

`data_cleaned` 是经过清洗和预处理后的 LIBERO 机器人操作数据集，专门为 OpenVLA 模型训练准备的高质量数据。

---

## 📁 数据结构

```
data_cleaned/
├── libero_goal/                    # 目标导向任务子集
│   ├── put_the_bowl_on_the_plate_demo/
│   │   ├── demo_0/
│   │   │   ├── images/
│   │   │   │   ├── 0000.jpg
│   │   │   │   ├── 0001.jpg
│   │   │   │   └── ...
│   │   │   └── actions.npy       # 动作序列
│   │   ├── demo_1/
│   │   │   └── ...
│   │   └── ...
├── libero_object/                  # 物体操作任务子集
├── libero_spatial/                 # 空间操作任务子集
├── libero_goal_splits.npy          # 训练/验证/测试划分信息
├── libero_object_splits.npy
└── libero_spatial_splits.npy
```

---

## 🎯 数据清洗内容

清洗过程删除了以下无效数据：

| 清洗类型 | 说明 |
|----------|------|
| 失败轨迹过滤 | 移除未成功完成任务的轨迹 |
| 黑帧过滤 | 移除过暗或损坏的图像（平均像素值 < 10） |
| 重复帧过滤 | 移除连续重复的图像（差异 < 3） |
| 无动作帧过滤 | 移除没有动作的帧（动作值和 < 0.005） |
| 短轨迹过滤 | 移除少于 5 帧的轨迹 |

---

## 📊 数据集划分

每个子集按 **8:1:1** 比例划分为训练集、验证集和测试集：

```python
# 划分比例
train: 80%  # 训练集
val: 10%    # 验证集
test: 10%   # 测试集
```

划分信息存储在 `*_splits.npy` 文件中。

---

## 🚀 使用方法

### 方法一：使用数据加载器（推荐）

```python
from data_loader import LIBEROCleanDataset, get_data_loaders

# 方式1：直接加载单个数据集
train_dataset = LIBEROCleanDataset(
    subset='train',           # 'train', 'val', 或 'test'
    data_root='./data_cleaned'
)

# 获取一个样本
sample = train_dataset[0]
images = sample['images']     # 图像列表
actions = sample['actions']  # 动作数组 (T, 7)
instruction = sample['instruction']  # 指令文本

# 方式2：批量加载所有数据集
train_loader, val_loader, test_loader = get_data_loaders(
    batch_size=8,
    num_workers=4
)

# 使用 DataLoader
for batch in train_loader:
    images = batch['images']      # 列表的列表
    actions = batch['actions']    # (batch_size, T, 7)
    instructions = batch['instruction']  # 指令列表
```

### 方法二：加载单个轨迹

```python
from data_loader import load_trajectory

# 加载单个轨迹
images, actions, instruction = load_trajectory(
    subset='libero_goal',
    task_name='put_the_bowl_on_the_plate_demo',
    demo_name='demo_0'
)

print(f"图像数量: {len(images)}")
print(f"动作形状: {actions.shape}")
print(f"指令: {instruction}")
```

### 方法三：查看数据集统计信息

```python
from data_loader import print_dataset_info

# 打印数据集统计
print_dataset_info()
```

---

## 📖 数据格式说明

### 图像格式

| 属性 | 值 |
|------|-----|
| 格式 | JPEG |
| 尺寸 | 224 × 224 |
| 通道 | RGB (3 channels) |
| 命名 | 4位数字，如 `0000.jpg`, `0001.jpg` |

### 动作格式

| 属性 | 值 |
|------|-----|
| 形状 | (T, 7)，T 为轨迹长度 |
| 维度 | 7（6D位姿 + 抓手状态） |
| 范围 | [-1, 1] 归一化 |
| 格式 | NumPy array (float32) |

### 指令文本

预定义的指令映射（从任务名称推断）：

```python
TASK_INSTRUCTIONS = {
    "put_the_bowl_on_the_plate_demo": "Put the bowl on the plate",
    "pick_up_the_bottle_demo": "Pick up the bottle",
    # ... 更多任务指令
}
```

---

## 🔍 数据验证

运行验证脚本检查数据完整性：

```bash
python verify_data.py
```

验证内容包括：
- ✅ 目录结构检查
- ✅ 数据划分文件检查
- ✅ 样本数据完整性检查
- ✅ 数据加载器功能测试

---

## 🏋️ 训练配置

训练配置文件 `configs/train_config.yaml` 已更新为使用清洗后的数据：

```yaml
# 数据配置
train_data_dir: "./data_cleaned"
val_data_dir: "./data_cleaned"
test_data_dir: "./data_cleaned"
batch_size: 8
image_size: 224
```

---

## 📈 数据统计

运行数据验证后可以看到：

| 子集 | 任务数 | 训练demo | 验证demo | 测试demo |
|------|--------|----------|----------|----------|
| libero_goal | ~10 | ~200 | ~25 | ~25 |
| libero_object | ~15 | ~300 | ~37 | ~37 |
| libero_spatial | ~10 | ~200 | ~25 | ~25 |
| **总计** | **~35** | **~700** | **~87** | **~87** |

---

## ⚠️ 注意事项

1. **数据完整性**：确保所有子集和划分文件都存在
2. **路径配置**：数据加载器默认使用 `./data_cleaned` 作为根目录
3. **内存占用**：加载大量图像时会占用较多内存，建议使用 DataLoader 的 `num_workers > 0`
4. **图像预处理**：数据加载器支持自定义图像变换（`transform` 参数）

---

## 🛠️ 故障排除

### 问题1：ImportError: cannot import name 'LIBEROCleanDataset'

**解决方案**：确保 `data_loader.py` 在 Python 路径中，或在项目根目录运行脚本。

### 问题2：FileNotFoundError: *_splits.npy not found

**解决方案**：检查 `data_cleaned` 目录结构是否完整，确保划分文件存在。

### 问题3：OutOfMemoryError

**解决方案**：
- 减小 `batch_size`
- 增加 DataLoader 的 `num_workers`
- 使用图像下采样

---

## 📚 相关文件

| 文件 | 说明 |
|------|------|
| `data_loader.py` | 数据加载器实现 |
| `verify_data.py` | 数据验证脚本 |
| `clean_libero.py` | 数据清洗脚本（原始） |
| `configs/train_config.yaml` | 训练配置 |

---

## ✅ 快速开始

```bash
# 1. 验证数据完整性
python verify_data.py

# 2. 开始训练
python train_lora.py --config configs/train_config.yaml

# 3. 评估模型
python evaluate.py --model runs/best_model.pt --data ./data_cleaned
```
