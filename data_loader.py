import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import subprocess

# 数据集路径 - 支持多个位置
DATA_ROOT = None
SUBSETS = ["libero_goal", "libero_object", "libero_spatial"]

def find_data_root():
    """自动查找数据集根目录"""
    possible_paths = [
        "./data_cleaned",
        "../data_cleaned",
        "/content/data_cleaned",
        "/content/drive/MyDrive/data_cleaned",
        os.path.expanduser("~/.cache/libero/data_cleaned")
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and len(os.listdir(path)) > 0:
            return path
    return None

def download_libero_dataset(target_path="./data_cleaned"):
    """下载 LIBERO 数据集（需要外网）"""
    print(f"正在下载 LIBERO 数据集到 {target_path}...")
    
    # 创建目录
    os.makedirs(target_path, exist_ok=True)
    
    # 使用 git 克隆数据集（如果可用）
    try:
        subprocess.run(
            ["git", "clone", "https://github.com/Lifelong-Robot-Learning/LIBERO.git", target_path],
            check=True,
            capture_output=True,
            text=True
        )
        print("数据集下载成功！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"下载失败: {e.stderr}")
        return False

# 初始化数据路径
DATA_ROOT = find_data_root()
if DATA_ROOT is None:
    print("警告：未找到数据集目录！")
    print("请确保 data_cleaned 目录存在，或手动设置 DATA_ROOT 环境变量")

# LIBERO 任务指令映射（基于任务名称）
TASK_INSTRUCTIONS = {
    # libero_goal
    "put_the_bowl_on_the_plate_demo": "Put the bowl on the plate",
    "put_the_bowl_on_the_stove_demo": "Put the bowl on the stove",
    "turn_on_the_stove_demo": "Turn on the stove",
    "turn_off_the_stove_demo": "Turn off the stove",
    "open_the_drawer_demo": "Open the drawer",
    "close_the_drawer_demo": "Close the drawer",
    "pick_up_the_bottle_demo": "Pick up the bottle",
    "place_the_bottle_on_the_table_demo": "Place the bottle on the table",
    # libero_object
    "pick_up_the_bowl_demo": "Pick up the bowl",
    "place_the_bowl_in_the_sink_demo": "Place the bowl in the sink",
    "move_the_bottle_demo": "Move the bottle",
    "push_the_object_demo": "Push the object",
    "grasp_the_object_demo": "Grasp the object",
    # libero_spatial
    "move_left_demo": "Move to the left",
    "move_right_demo": "Move to the right",
    "move_forward_demo": "Move forward",
    "move_backward_demo": "Move backward",
}

def load_trajectory(subset, task_name, demo_name):
    """读取单条轨迹的图像、动作、指令
    
    Args:
        subset: 数据子集名称 (libero_goal/libero_object/libero_spatial)
        task_name: 任务名称 (e.g., put_the_bowl_on_the_plate_demo)
        demo_name: Demo名称 (e.g., demo_0)
    
    Returns:
        images: 图像列表
        actions: 动作数组
        instruction: 指令文本
    """
    traj_path = os.path.join(DATA_ROOT, subset, task_name, demo_name)
    
    # 读取图像
    images = []
    img_dir = os.path.join(traj_path, "images")
    if os.path.exists(img_dir):
        for img_file in sorted(os.listdir(img_dir)):
            if img_file.endswith(('.jpg', '.png', '.jpeg')):
                img = Image.open(os.path.join(img_dir, img_file)).convert("RGB")
                images.append(np.array(img))
    
    # 读取动作
    actions_path = os.path.join(traj_path, "actions.npy")
    if os.path.exists(actions_path):
        actions = np.load(actions_path)
    else:
        actions = np.array([])
    
    # 获取指令文本
    instruction = TASK_INSTRUCTIONS.get(task_name, f"Complete the task: {task_name}")
    
    return images, actions, instruction


class LIBEROCleanDataset(Dataset):
    """LIBERO 清洗后数据集
    
    数据结构：
        data_cleaned/
        ├── libero_goal/
        │   ├── task_name/
        │   │   ├── demo_0/
        │   │   │   ├── images/
        │   │   │   └── actions.npy
        │   │   └── demo_1/
        │   │       └── ...
        ├── libero_object/
        ├── libero_spatial/
        └── *_splits.npy  # 数据划分信息
    """
    
    def __init__(self, subset="train", data_root=DATA_ROOT, transform=None):
        """
        Args:
            subset: 数据子集 ('train', 'val', 'test')
            data_root: 数据根目录
            transform: 图像变换
        """
        self.data_root = data_root
        self.transform = transform
        self.samples = []
        
        # 加载所有子集的数据
        for subset_name in SUBSETS:
            splits_file = os.path.join(data_root, f"{subset_name}_splits.npy")
            if os.path.exists(splits_file):
                splits = np.load(splits_file, allow_pickle=True).item()
                if subset in splits:
                    task_names = splits[subset]
                    for task_name in task_names:
                        task_path = os.path.join(data_root, subset_name, task_name)
                        if os.path.exists(task_path):
                            # 遍历该任务的所有demo
                            for demo_name in os.listdir(task_path):
                                demo_path = os.path.join(task_path, demo_name)
                                if os.path.isdir(demo_path):
                                    # 检查是否有有效数据
                                    if self._check_valid_demo(demo_path):
                                        self.samples.append({
                                            'subset': subset_name,
                                            'task_name': task_name,
                                            'demo_name': demo_name,
                                            'path': demo_path
                                        })
    
    def _check_valid_demo(self, demo_path):
        """检查demo是否有效（有图像和动作）"""
        has_images = os.path.exists(os.path.join(demo_path, "images"))
        has_actions = os.path.exists(os.path.join(demo_path, "actions.npy"))
        return has_images and has_actions
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图像
        img_dir = os.path.join(sample['path'], "images")
        images = []
        for img_file in sorted(os.listdir(img_dir)):
            if img_file.endswith(('.jpg', '.png', '.jpeg')):
                img = Image.open(os.path.join(img_dir, img_file)).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                images.append(img)
        
        # 加载动作
        actions = np.load(os.path.join(sample['path'], "actions.npy"))
        
        # 获取指令
        instruction = TASK_INSTRUCTIONS.get(
            sample['task_name'], 
            f"Complete the task: {sample['task_name']}"
        )
        
        return {
            'images': images,
            'actions': actions,
            'instruction': instruction,
            'task_name': sample['task_name'],
            'demo_name': sample['demo_name']
        }


def get_data_loaders(batch_size=8, num_workers=4, transform=None):
    """获取训练、验证、测试数据加载器
    
    Args:
        batch_size: 批次大小
        num_workers: 工作进程数
        transform: 图像变换
    
    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = LIBEROCleanDataset(subset='train', transform=transform)
    val_dataset = LIBEROCleanDataset(subset='val', transform=transform)
    test_dataset = LIBEROCleanDataset(subset='test', transform=transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


def print_dataset_info():
    """打印数据集统计信息"""
    print("=" * 60)
    print("LIBERO 清洗后数据集统计")
    print("=" * 60)
    
    for subset_name in SUBSETS:
        splits_file = os.path.join(DATA_ROOT, f"{subset_name}_splits.npy")
        if os.path.exists(splits_file):
            splits = np.load(splits_file, allow_pickle=True).item()
            print(f"\n{subset_name}:")
            for split_name, tasks in splits.items():
                print(f"  {split_name}: {len(tasks)} 个任务")
    
    print("\n数据集加载器：")
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=1)
    print(f"  训练集: {len(train_loader.dataset)} 个样本")
    print(f"  验证集: {len(val_loader.dataset)} 个样本")
    print(f"  测试集: {len(test_loader.dataset)} 个样本")
    print("=" * 60)


if __name__ == "__main__":
    # 测试数据加载
    print_dataset_info()
    
    # 加载一个样本测试
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=1)
    
    print("\n测试加载一个样本：")
    for batch in train_loader:
        print(f"  图像数量: {len(batch['images'])}")
        print(f"  动作形状: {batch['actions'].shape}")
        print(f"  指令: {batch['instruction'][0]}")
        break