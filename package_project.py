#!/usr/bin/env python3
"""
OpenVLA 项目打包脚本

功能：
- 将项目核心文件打包成 zip 压缩包
- 可选择是否包含数据集（大文件）
- 生成打包清单和说明

运行命令：
    # 打包所有文件（不含大数据集）
    python package_project.py
    
    # 打包包含数据集
    python package_project.py --include_data
    
    # 指定输出目录
    python package_project.py --output ./output
"""

import os
import sys
import zipfile
import argparse
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="OpenVLA Project Packager")
    parser.add_argument("--include_data", action="store_true", help="Include data_cleaned directory")
    parser.add_argument("--output", type=str, default="./", help="Output directory")
    parser.add_argument("--name", type=str, default=None, help="Custom package name")
    return parser.parse_args()


def get_project_files(include_data=False):
    """获取项目核心文件列表"""
    
    # 必须包含的文件
    core_files = [
        "train_lora.py",
        "train_simple.py",
        "data_loader.py",
        "utils.py",
        "evaluate.py",
        "load_and_test.py",
        "demo.py",
        "requirements.txt",
        "environment.yml",
        "PACKAGE_GUIDE.md",
        "README.md",
        "OpenVLA_模型架构详细分析报告.md",
        "OpenVLA_架构图.md",
        "data_cleaned使用说明.md",
    ]
    
    # 配置文件
    config_files = [
        "configs/train_config.yaml",
        "configs/simple_config.yaml",
    ]
    
    # 数据集（可选）
    data_files = []
    if include_data:
        data_dir = "data_cleaned"
        if os.path.exists(data_dir):
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if file.endswith(('.npy', '.json', '.png', '.jpg', '.jpeg')):
                        data_files.append(os.path.join(root, file))
    
    # 训练结果（可选，通常不包含）
    runs_files = []
    # 如果需要包含训练结果，可以取消注释
    # if os.path.exists("runs"):
    #     for root, dirs, files in os.walk("runs"):
    #         for file in files:
    #             runs_files.append(os.path.join(root, file))
    
    return {
        "core": core_files,
        "config": config_files,
        "data": data_files,
        "runs": runs_files
    }


def create_package(file_list, output_dir, package_name=None):
    """创建压缩包"""
    
    # 生成包名
    if package_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        package_name = f"openvla_project_{timestamp}.zip"
    
    output_path = os.path.join(output_dir, package_name)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建压缩包
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        total_files = 0
        total_size = 0
        
        # 添加核心文件
        for file in file_list["core"]:
            if os.path.exists(file):
                zf.write(file)
                total_files += 1
                total_size += os.path.getsize(file)
                print(f"OK {file}")
        
        # 添加配置文件
        for file in file_list["config"]:
            if os.path.exists(file):
                zf.write(file)
                total_files += 1
                total_size += os.path.getsize(file)
                print(f"OK {file}")
        
        # 添加数据集
        for file in file_list["data"]:
            if os.path.exists(file):
                zf.write(file)
                total_files += 1
                total_size += os.path.getsize(file)
        
        # 添加训练结果
        for file in file_list["runs"]:
            if os.path.exists(file):
                zf.write(file)
                total_files += 1
                total_size += os.path.getsize(file)
    
    # 格式化大小
    if total_size >= 1e9:
        size_str = f"{total_size / 1e9:.2f} GB"
    elif total_size >= 1e6:
        size_str = f"{total_size / 1e6:.2f} MB"
    elif total_size >= 1e3:
        size_str = f"{total_size / 1e3:.2f} KB"
    else:
        size_str = f"{total_size} B"
    
    print("\n" + "="*50)
    print(f"打包完成！")
    print(f"输出文件: {output_path}")
    print(f"文件数量: {total_files}")
    print(f"总大小: {size_str}")
    print("="*50)
    
    return output_path


def generate_manifest(file_list, output_dir):
    """生成打包清单"""
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "files": {
            "core": [f for f in file_list["core"] if os.path.exists(f)],
            "config": [f for f in file_list["config"] if os.path.exists(f)],
            "data": len(file_list["data"]),
            "runs": len(file_list["runs"])
        }
    }
    
    manifest_path = os.path.join(output_dir, "package_manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        import json
        json.dump(manifest, f, indent=4, ensure_ascii=False)
    
    print(f"清单文件: {manifest_path}")


def main():
    args = parse_args()
    
    print("OpenVLA 项目打包工具")
    print("="*50)
    
    # 获取文件列表
    print("\n收集文件...")
    file_list = get_project_files(include_data=args.include_data)
    
    # 显示文件统计
    print(f"\n核心文件: {len(file_list['core'])}")
    print(f"配置文件: {len(file_list['config'])}")
    print(f"数据文件: {len(file_list['data'])}")
    print(f"训练结果: {len(file_list['runs'])}")
    
    # 创建打包
    print("\n开始打包...")
    output_path = create_package(file_list, args.output, args.name)
    
    # 生成清单
    generate_manifest(file_list, args.output)
    
    print(f"\n提示: 将 {output_path} 复制到有 GPU/外网的环境后解压即可使用")


if __name__ == "__main__":
    main()
