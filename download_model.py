from huggingface_hub import snapshot_download
import os

# 启用国内镜像加速（解决下载慢的问题）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 模型 ID（选主版本 openvla-7b）
model_id = "openvla/openvla-7b"

# 下载到本地的文件夹（可以自己改路径）
local_dir = "E:/jushen/models/openvla-7b"

# 开始下载
print("开始下载模型...")
snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    revision="main"
)

print(f"✅ 模型已下载到：{local_dir}")