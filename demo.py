import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def main():
    print("=" * 50)
    print("  从本地加载 OpenVLA 模型...")
    print("=" * 50)

    # 从本地文件夹加载，不联网
    processor = AutoProcessor.from_pretrained(
        "./openvla-7b",
        trust_remote_code=True,
        local_files_only=True  # 关键：强制只从本地加载
    )

    model = AutoModelForVision2Seq.from_pretrained(
        "./openvla-7b",
        trust_remote_code=True,
        local_files_only=True,  # 关键：强制只从本地加载
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="cpu"  # 没有显卡也能跑
    )

    print("✅ 模型加载完成！")

    # 测试图片和指令
    image = Image.new("RGB", (224, 224), color=(255, 255, 255))
    prompt = "pick up the cup"

    inputs = processor(prompt, image, return_tensors="pt")
    action = model.predict_action(**inputs, unnorm_key="bridge_orig")

    print("\n" + "=" * 50)
    print("🎉 Demo 运行成功！环境搭建完成！")
    print("指令：", prompt)
    print("预测动作：", action)
    print("=" * 50)

if __name__ == "__main__":
    main()