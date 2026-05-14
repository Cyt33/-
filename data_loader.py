import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# 你的数据集路径
DATA_ROOT = "D:/OpenVLA/openvla-main/data"
SUBSETS = ["libero_goal", "libero_object", "libero_spatial"]

def load_trajectory(subset, traj_id):
    """读取单条轨迹的图像、动作、指令"""
    traj_path = os.path.join(DATA_ROOT, subset, f"traj_{traj_id}")
    # 读取图像
    images = []
    for img_file in sorted(os.listdir(os.path.join(traj_path, "images"))):
        img = Image.open(os.path.join(traj_path, "images", img_file)).convert("RGB")
        images.append(np.array(img))
    # 读取动作和指令（根据LIBERO格式调整）
    actions = np.load(os.path.join(traj_path, "actions.npy"))
    instruction = open(os.path.join(traj_path, "instruction.txt"), "r").read().strip()
    return images, actions, instruction