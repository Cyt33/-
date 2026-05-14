import os
from huggingface_hub import snapshot_download

# 强制使用国内镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("🚀 开始从 hf-mirror.com 下载数据集...")
print("📂 目标路径：./data/modified_libero_rlds")

try:
    dataset_path = snapshot_download(
        repo_id="openvla/modified_libero_rlds",
        repo_type="dataset",
        local_dir="./data/modified_libero_rlds",
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print(f"✅ 下载成功！文件保存在：{dataset_path}")

except Exception as e:
    print(f"❌ 下载失败，错误信息：{e}")
    print("\n💡 手动下载方案：")
    print("1. 浏览器访问：https://hf-mirror.com/datasets/openvla/modified_libero_rlds")
    print("2. 点击「Files and versions」→ 下载 zip 包")
    print("3. 解压后复制到 ./data/modified_libero_rlds 文件夹即可")