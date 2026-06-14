from transformers import AutoModel, AutoProcessor

# 模型路径（你已经下载好了）
model_path = "E:/jushen/models/openvla-7b"

# 关键：加上 trust_remote_code=True，让 transformers 直接执行模型里的代码
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True  # 这行是解决所有问题的核心
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

print("✅ 模型加载成功！")