"""
OpenVLA 快速使用示例
====================

组员二：模型架构 + 核心算法复现

本脚本提供OpenVLA模型的最简使用方式，适合快速测试和演示
"""

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


def register_openvla_to_hf():
    """注册OpenVLA到HuggingFace Auto类"""
    from transformers import AutoConfig, AutoImageProcessor, AutoProcessor
    
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


def load_model_demo():
    """演示：加载模型"""
    print("\n" + "=" * 60)
    print("演示1：基础模型加载")
    print("=" * 60)
    
    # 注册模型
    register_openvla_to_hf()
    
    # 从HuggingFace加载
    print("[*] 从HuggingFace加载模型...")
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to("cuda:0")
    
    print("[✓] 模型加载完成")
    return model, processor


def inference_demo(model, processor):
    """演示：执行推理"""
    print("\n" + "=" * 60)
    print("演示2：执行动作预测")
    print("=" * 60)
    
    # 准备输入
    image = Image.new("RGB", (224, 224), color=(255, 128, 64))
    prompt = "In: What action should the robot take to pick up the cup?\nOut:"
    
    print(f"[*] 输入图像: {image.size}")
    print(f"[*] 指令: pick up the cup")
    
    # 处理输入
    inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
    
    # 预测动作
    action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    
    print(f"[✓] 预测动作: {action}")
    
    return action


def quantization_demo():
    """演示：量化模型加载"""
    print("\n" + "=" * 60)
    print("演示3：量化模型加载")
    print("=" * 60)
    
    from transformers import BitsAndBytesConfig
    
    # INT8量化配置
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )
    
    register_openvla_to_hf()
    
    print("[*] 加载INT8量化模型...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    
    print("[✓] 量化模型加载完成")
    print("[*] 显存占用大幅减少，可支持更大批量推理")
    
    return model, processor


def cpu_demo():
    """演示：CPU加载（无GPU）"""
    print("\n" + "=" * 60)
    print("演示4：CPU模式加载")
    print("=" * 60)
    
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    register_openvla_to_hf()
    
    print("[*] 从本地加载模型到CPU...")
    processor = AutoProcessor.from_pretrained(
        "./openvla-7b",
        trust_remote_code=True,
        local_files_only=True
    )
    
    model = AutoModelForVision2Seq.from_pretrained(
        "./openvla-7b",
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="cpu"
    )
    
    print("[✓] CPU模型加载完成")
    
    return model, processor


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("  OpenVLA 快速使用演示")
    print("  组员二：模型架构 + 核心算法复现")
    print("=" * 70)
    
    try:
        # 演示1：基础加载
        model, processor = load_model_demo()
        
        # 演示2：推理
        inference_demo(model, processor)
        
        print("\n[✓] 基础演示完成")
        
    except Exception as e:
        print(f"[!] 演示1失败: {e}")
        print("[*] 尝试CPU模式...")
        
        try:
            model, processor = cpu_demo()
            inference_demo(model, processor)
        except Exception as e2:
            print(f"[!] CPU模式也失败: {e2}")
    
    # 演示3：量化（仅在有GPU时）
    if torch.cuda.is_available():
        try:
            model_quant, processor_quant = quantization_demo()
            
            image = Image.new("RGB", (224, 224), color=(100, 150, 200))
            prompt = "In: What action should the robot take?\nOut:"
            inputs = processor_quant(prompt, image).to("cuda:0")
            action = model_quant.predict_action(**inputs, unnorm_key="bridge_orig")
            
            print(f"[✓] 量化模型预测: {action}")
            
        except Exception as e:
            print(f"[!] 量化演示失败: {e}")
    
    print("\n" + "=" * 70)
    print("  演示结束")
    print("=" * 70)


if __name__ == "__main__":
    main()
