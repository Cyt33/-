"""
数据验证脚本

检查 data_cleaned 数据集的结构和完整性
"""

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
import json

# 设置默认编码
sys.stdout.reconfigure(encoding='utf-8')

# 数据集路径
DATA_ROOT = "./data_cleaned"
SUBSETS = ["libero_goal", "libero_object", "libero_spatial"]


def check_dataset_structure():
    """检查数据集目录结构"""
    print("=" * 70)
    print("检查数据集目录结构")
    print("=" * 70)
    
    if not os.path.exists(DATA_ROOT):
        print(f"ERROR: 数据集目录不存在: {DATA_ROOT}")
        return False
    
    print(f"OK: 数据集目录存在: {DATA_ROOT}\n")
    
    all_valid = True
    
    for subset in SUBSETS:
        subset_path = os.path.join(DATA_ROOT, subset)
        
        if not os.path.exists(subset_path):
            print(f"WARN: 子集不存在: {subset}")
            all_valid = False
            continue
        
        print(f"\nDIR {subset}:")
        
        # 统计任务数量
        tasks = [d for d in os.listdir(subset_path) if os.path.isdir(os.path.join(subset_path, d))]
        print(f"  任务数量: {len(tasks)}")
        
        # 统计demo数量
        total_demos = 0
        valid_demos = 0
        
        for task in tasks[:5]:  # 只检查前5个任务
            task_path = os.path.join(subset_path, task)
            demos = [d for d in os.listdir(task_path) if os.path.isdir(os.path.join(task_path, d))]
            total_demos += len(demos)
            
            # 检查demo的有效性
            for demo in demos[:3]:  # 只检查前3个demo
                demo_path = os.path.join(task_path, demo)
                has_images = os.path.exists(os.path.join(demo_path, "images"))
                has_actions = os.path.exists(os.path.join(demo_path, "actions.npy"))
                if has_images and has_actions:
                    valid_demos += 1
        
        print(f"  示例任务: {tasks[:3] if len(tasks) >= 3 else tasks}")
        print(f"  总demo数（估算）: ~{total_demos * (len(demos) / 3 if len(demos) >= 3 else 1):.0f}")
        print(f"  有效demo数（估算）: ~{valid_demos * (len(demos) / 3 if len(demos) >= 3 else 1):.0f}")
    
    return all_valid


def check_splits():
    """检查数据划分文件"""
    print("\n" + "=" * 70)
    print("检查数据划分文件")
    print("=" * 70)
    
    for subset in SUBSETS:
        splits_file = os.path.join(DATA_ROOT, f"{subset}_splits.npy")
        
        if not os.path.exists(splits_file):
            print(f"WARN: 划分文件不存在: {splits_file}")
            continue
        
        splits = np.load(splits_file, allow_pickle=True).item()
        
        print(f"\nINFO {subset}:")
        print(f"  划分文件: {splits_file}")
        
        for split_name, items in splits.items():
            print(f"    {split_name}: {len(items)} 个任务")
            print(f"      示例: {items[:3] if len(items) >= 3 else items}")


def check_sample_data():
    """检查样本数据的完整性"""
    print("\n" + "=" * 70)
    print("检查样本数据完整性")
    print("=" * 70)
    
    # 查找第一个有效的demo
    for subset in SUBSETS:
        subset_path = os.path.join(DATA_ROOT, subset)
        if not os.path.exists(subset_path):
            continue
        
        tasks = [d for d in os.listdir(subset_path) if os.path.isdir(os.path.join(subset_path, d))]
        
        for task in tasks:
            task_path = os.path.join(subset_path, task)
            demos = [d for d in os.listdir(task_path) if os.path.isdir(os.path.join(task_path, d))]
            
            for demo in demos:
                demo_path = os.path.join(task_path, demo)
                images_path = os.path.join(demo_path, "images")
                actions_path = os.path.join(demo_path, "actions.npy")
                
                if os.path.exists(images_path) and os.path.exists(actions_path):
                    # 检查图像
                    images = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
                    
                    # 检查动作
                    actions = np.load(actions_path)
                    
                    print(f"\nSAMPLE 第一个有效样本:")
                    print(f"  路径: {demo_path}")
                    print(f"  图像数量: {len(images)}")
                    print(f"  图像示例: {images[:5] if len(images) >= 5 else images}")
                    print(f"  动作形状: {actions.shape}")
                    print(f"  动作范围: [{actions.min():.4f}, {actions.max():.4f}]")
                    
                    # 尝试读取一个图像
                    if len(images) > 0:
                        img = Image.open(os.path.join(images_path, images[0]))
                        print(f"  图像尺寸: {img.size}")
                        print(f"  图像模式: {img.mode}")
                    
                    return True
    
    print("ERROR: 未找到有效样本")
    return False


def test_data_loader():
    """测试 data_loader.py 中的 LIBEROCleanDataset"""
    print("\n" + "=" * 70)
    print("测试数据加载器")
    print("=" * 70)
    
    try:
        from data_loader import LIBEROCleanDataset, get_data_loaders
        
        print("\nOK: 成功导入 LIBEROCleanDataset")
        
        # 测试数据集加载
        print("\n加载数据集...")
        train_dataset = LIBEROCleanDataset(subset='train', data_root=DATA_ROOT)
        val_dataset = LIBEROCleanDataset(subset='val', data_root=DATA_ROOT)
        test_dataset = LIBEROCleanDataset(subset='test', data_root=DATA_ROOT)
        
        print(f"OK: 训练集样本数: {len(train_dataset)}")
        print(f"OK: 验证集样本数: {len(val_dataset)}")
        print(f"OK: 测试集样本数: {len(test_dataset)}")
        
        # 测试加载一个样本
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"\nITEM 样本内容:")
            print(f"  图像数量: {len(sample['images'])}")
            print(f"  动作形状: {sample['actions'].shape}")
            print(f"  指令: {sample['instruction']}")
            print(f"  任务名: {sample['task_name']}")
            print(f"  Demo名: {sample['demo_name']}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: 数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("OpenVLA 数据集验证")
    print("=" * 70)
    
    # 1. 检查目录结构
    check_dataset_structure()
    
    # 2. 检查数据划分
    check_splits()
    
    # 3. 检查样本数据
    check_sample_data()
    
    # 4. 测试数据加载器
    test_data_loader()
    
    print("\n" + "=" * 70)
    print("数据验证完成！")
    print("=" * 70)
    
    print("\n下一步:")
    print("1. 确保数据验证全部通过")
    print("2. 运行训练: python train_lora.py --config configs/train_config.yaml")
    print("3. 运行评估: python evaluate.py --model runs/best_model.pt --data ./data_cleaned")


if __name__ == "__main__":
    main()
