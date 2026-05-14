import os
import h5py
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ===================== 路径配置 =====================
DATA_ROOT = r"D:\OpenVLA\openvla-main\data"
SAVE_ROOT = r"D:\OpenVLA\openvla-main\data_cleaned"
SUBSETS = ["libero_goal", "libero_object", "libero_spatial"]
IMAGE_SIZE = (224, 224)
# ====================================================

# 创建输出目录
for subset in SUBSETS:
    os.makedirs(os.path.join(SAVE_ROOT, subset), exist_ok=True)

# --------------------- 1. 处理单个 HDF5 文件 ---------------------
def process_hdf5_file(file_path, save_dir):
    """读取并清洗单个 LIBERO HDF5 文件"""
    with h5py.File(file_path, "r") as f:
        # 读取所有 demo 组
        demo_groups = [k for k in f["data"].keys() if k.startswith("demo_")]
        if not demo_groups:
            return False

        valid_trajs = []

        for demo in demo_groups:
            demo_path = f"data/{demo}"
            actions = f[f"{demo_path}/actions"][:]
            images = f[f"{demo_path}/obs/agentview_rgb"][:]
            dones = f[f"{demo_path}/dones"][:]

            # 判断是否成功：轨迹最后一步是 done
            if not dones[-1]:
                continue

            # 清洗帧：去黑帧、重复帧、无动作帧
            clean_imgs = []
            clean_acts = []
            last_img = None

            for i in range(len(actions)):
                img = images[i]
                act = actions[i]

                # 统一尺寸（先resize，再比较）
                img_resized = cv2.resize(img, IMAGE_SIZE)

                # 黑帧过滤
                if img_resized.mean() < 10:
                    continue
                # 重复帧过滤（现在两张图都是224x224了，不会报错）
                if last_img is not None and cv2.absdiff(img_resized, last_img).mean() < 3:
                    continue
                # 无动作过滤
                if np.abs(act).sum() < 0.005:
                    continue

                clean_imgs.append(img_resized)
                clean_acts.append(act)
                last_img = img_resized

            if len(clean_imgs) < 5:
                continue

            # 保存这条清洗后的轨迹
            traj_save_dir = os.path.join(save_dir, demo)
            os.makedirs(os.path.join(traj_save_dir, "images"), exist_ok=True)
            for idx, img in enumerate(clean_imgs):
                cv2.imwrite(os.path.join(traj_save_dir, "images", f"{idx:04d}.jpg"), img)
            np.save(os.path.join(traj_save_dir, "actions.npy"), np.array(clean_acts))

            valid_trajs.append(demo)

        return len(valid_trajs) > 0

# --------------------- 2. 主流程 ---------------------
for subset in SUBSETS:
    print(f"\n==================================")
    print(f" 处理数据集：{subset}")
    print(f"==================================")

    subset_path = os.path.join(DATA_ROOT, subset)
    save_subset_path = os.path.join(SAVE_ROOT, subset)

    # 找出所有 .hdf5 文件
    hdf5_files = [f for f in os.listdir(subset_path) if f.endswith(".hdf5")]
    print(f"发现文件数量：{len(hdf5_files)}")

    valid_cleaned = []
    for file_name in tqdm(hdf5_files, desc="清洗文件"):
        file_path = os.path.join(subset_path, file_name)
        task_name = os.path.splitext(file_name)[0]
        save_task_path = os.path.join(save_subset_path, task_name)

        if process_hdf5_file(file_path, save_task_path):
            valid_cleaned.append(task_name)

    print(f"✅ 清洗完成，有效任务数：{len(valid_cleaned)}")

    if len(valid_cleaned) < 3:
        print(f"⚠️ 数据太少，不划分数据集")
        continue

    # 划分训练集/验证集/测试集 8:1:1
    train, temp = train_test_split(valid_cleaned, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    split_info = {
        "train": train,
        "val": val,
        "test": test
    }
    np.save(os.path.join(SAVE_ROOT, f"{subset}_splits.npy"), split_info)
    print(f"✅ 数据集划分完成：train={len(train)}, val={len(val)}, test={len(test)}")

print("\n🎉 全部完成！")
print("📁 清洗后数据保存在：", SAVE_ROOT)