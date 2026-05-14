import os
import json
import numpy as np
import tensorflow as tf
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ==============================================
# 【配置】直接用你项目的路径
# ==============================================
DATA_ROOT = "./data/modified_libero_rlds"
OUTPUT_DIR = "./data/processed_libero"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# 图像预处理（适配 OpenVLA）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==============================================
# 步骤1：创建输出文件夹
# ==============================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, "actions"), exist_ok=True)

# ==============================================
# 步骤2：读取 RLDS 格式的 TFRecord 文件
# ==============================================
all_samples = []
print("正在扫描 RLDS 数据集...")

# 遍历所有 .tfrecord 文件
for root, dirs, files in os.walk(DATA_ROOT):
    for file in files:
        if file.endswith(".tfrecord"):
            tfrecord_path = os.path.join(root, file)
            print(f"正在读取：{tfrecord_path}")

            # 解析 TFRecord
            raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
            for raw_record in raw_dataset:
                example = tf.train.Example.FromString(raw_record.numpy())
                feature = example.features.feature

                # 读取图片和动作（字段名根据 LIBERO RLDS 格式调整）
                if "observation/image" in feature and "action" in feature:
                    img_bytes = feature["observation/image"].bytes_list.value[0]
                    action = np.array(feature["action"].float_list.value)

                    # 解码图片
                    img = tf.io.decode_jpeg(img_bytes).numpy()

                    all_samples.append({
                        "img": img,
                        "act": action
                    })

print(f"✅ 扫描完成，总样本数：{len(all_samples)}")

if len(all_samples) == 0:
    print("❌ 没有读取到任何数据，请检查数据集路径和格式！")
    exit()

# ==============================================
# 步骤3：划分训练集 / 验证集 / 测试集
# ==============================================
train_samples, temp_samples = train_test_split(all_samples, test_size=1 - TRAIN_RATIO, random_state=42)
val_samples, test_samples = train_test_split(temp_samples, test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO), random_state=42)

split_map = {
    "train": train_samples,
    "val": val_samples,
    "test": test_samples
}

# ==============================================
# 步骤4：预处理并保存
# ==============================================
print("开始预处理并保存数据...")
idx = 0
for split_name, samples in split_map.items():
    print(f"处理 {split_name} 集...")
    for sample in tqdm(samples):
        # 图像预处理
        img = transforms.ToPILImage()(sample["img"])
        img_tensor = transform(img)

        # 动作标准化
        action = torch.tensor(sample["act"], dtype=torch.float32)

        # 保存
        torch.save(img_tensor, os.path.join(OUTPUT_DIR, split_name, "images", f"{idx:06d}.pt"))
        torch.save(action, os.path.join(OUTPUT_DIR, split_name, "actions", f"{idx:06d}.pt"))
        idx += 1

# ==============================================
# 步骤5：生成数据集说明文件
# ==============================================
stats = {
    "total_samples": len(all_samples),
    "train": len(train_samples),
    "val": len(val_samples),
    "test": len(test_samples),
    "image_size": "224x224",
    "action_dim": len(all_samples[0]["act"]) if all_samples else 0
}

with open(os.path.join(OUTPUT_DIR, "dataset_info.json"), "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=4)

print("✅ 数据预处理全部完成！")
print(f"📂 输出路径：{OUTPUT_DIR}")
print(f"📊 数据集统计：{stats}")