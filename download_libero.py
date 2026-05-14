import os
from huggingface_hub import snapshot_download

# 配置国内镜像，解决下载慢/域名解析问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 下载 OpenVLA 适配的 LIBERO 数据集，直接存到 data 目录
dataset_path = snapshot_download(
    repo_id="openvla/modified_libero_rlds",
    repo_type="dataset",
    local_dir="./data/modified_libero_rlds",
    local_dir_use_symlinks=False,
    resume_download=True  # 支持断点续传，中断后重新运行可继续下载
)

print(f"✅ 数据集已成功下载到：{dataset_path}")