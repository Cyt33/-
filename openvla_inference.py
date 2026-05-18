"""
OpenVLA 模型加载与推理模块
============================

组员二：模型架构 + 核心算法复现

本模块提供OpenVLA模型的完整加载和推理功能，包括：
- 基础模型加载
- 量化模型加载（INT8/INT4）
- CPU推理支持
- 完整的推理流程封装
"""

import os
import torch
from PIL import Image
from typing import Optional, Union, Dict, Any
from transformers import (
    AutoModelForVision2Seq, 
    AutoProcessor, 
    BitsAndBytesConfig
)

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


class OpenVLALoader:
    """
    OpenVLA模型加载器
    
    支持多种加载方式：
    - 标准加载（BF16）
    - Flash Attention加速
    - INT8量化
    - INT4量化
    - CPU加载
    """
    
    # 系统提示词模板
    SYSTEM_PROMPT_V01 = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        
    def register_to_hf(self):
        """
        将OpenVLA模型注册到HuggingFace Auto类
        
        这是加载本地模型或未在HF Hub注册的模型时必需的操作
        """
        from transformers import AutoConfig, AutoImageProcessor, AutoProcessor
        
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
        
    def load_standard(
        self,
        model_path: str,
        use_flash_attention: bool = True,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda:0"
    ) -> tuple:
        """
        标准加载方式（BF16）
        
        Args:
            model_path: 模型路径或HuggingFace模型ID
            use_flash_attention: 是否使用Flash Attention加速
            torch_dtype: 模型数据类型
            device: 加载设备
        
        Returns:
            (model, processor) 元组
        """
        print(f"[*] 加载OpenVLA模型（标准模式 - BF16）")
        if use_flash_attention:
            print(f"[*] 启用 Flash Attention 2 加速")
        
        self.register_to_hf()
        
        # 加载Processor
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=False
        )
        
        # 加载模型
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        
        if use_flash_attention:
            load_kwargs["attn_implementation"] = "flash_attention_2"
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            **load_kwargs
        )
        
        # 移动到设备
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        if device != "cpu":
            self.model = self.model.to(self.device)
        
        print(f"[✓] 模型加载完成，设备: {self.device}")
        
        return self.model, self.processor
    
    def load_quantized(
        self,
        model_path: str,
        quantization_mode: str = "int8",
        torch_dtype: torch.dtype = torch.bfloat16
    ) -> tuple:
        """
        量化加载方式（INT8或INT4）
        
        Args:
            model_path: 模型路径或HuggingFace模型ID
            quantization_mode: 量化模式，"int8" 或 "int4"
            torch_dtype: 模型数据类型
        
        Returns:
            (model, processor) 元组
        """
        if quantization_mode not in ["int8", "int4"]:
            raise ValueError(f"quantization_mode必须是'int8'或'int4'，得到: {quantization_mode}")
        
        print(f"[*] 加载OpenVLA模型（量化模式 - {quantization_mode.upper()}）")
        
        self.register_to_hf()
        
        # 加载Processor
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=False
        )
        
        # 配置量化
        if quantization_mode == "int8":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            print(f"[*] 使用 INT8 量化配置")
        else:  # int4
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            print(f"[*] 使用 INT4 量化配置（NF4格式）")
        
        # 加载量化模型
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        
        self.device = torch.device("cuda:0")
        print(f"[✓] 量化模型加载完成，设备: {self.device}")
        
        return self.model, self.processor
    
    def load_cpu(
        self,
        model_path: str,
        torch_dtype: torch.dtype = torch.float32
    ) -> tuple:
        """
        CPU加载方式（无GPU环境）
        
        Args:
            model_path: 模型路径
            torch_dtype: 模型数据类型
        
        Returns:
            (model, processor) 元组
        """
        print(f"[*] 加载OpenVLA模型（CPU模式）")
        
        self.register_to_hf()
        
        # 加载Processor
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        
        # 加载模型到CPU
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map="cpu"
        )
        
        self.device = torch.device("cpu")
        print(f"[✓] CPU模型加载完成")
        
        return self.model, self.processor
    
    def load_auto(
        self,
        model_path: str,
        quantization: bool = False,
        quantization_mode: str = "int8",
        use_flash_attention: bool = True,
        force_cpu: bool = False
    ) -> tuple:
        """
        自动选择最佳加载方式
        
        Args:
            model_path: 模型路径
            quantization: 是否使用量化
            quantization_mode: 量化模式
            use_flash_attention: 是否使用Flash Attention
            force_cpu: 强制使用CPU
        
        Returns:
            (model, processor) 元组
        """
        if force_cpu:
            return self.load_cpu(model_path)
        
        if quantization:
            return self.load_quantized(model_path, quantization_mode)
        
        return self.load_standard(
            model_path,
            use_flash_attention=use_flash_attention
        )


class OpenVLAInference:
    """
    OpenVLA推理器
    
    提供完整的推理流程，包括：
    - 图像预处理
    - 提示词构建
    - 动作预测
    - 结果后处理
    """
    
    def __init__(self, model, processor, device: Optional[torch.device] = None):
        """
        初始化推理器
        
        Args:
            model: OpenVLA模型
            processor: 预处理器
            device: 运行设备
        """
        self.model = model
        self.processor = processor
        self.device = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        
    def build_prompt(
        self,
        instruction: str,
        model_version: str = "v1"
    ) -> str:
        """
        构建提示词
        
        Args:
            instruction: 任务指令（如"pick up the cup"）
            model_version: 模型版本，"v1" 或 "v01"
        
        Returns:
            格式化后的提示词
        """
        if model_version == "v01":
            prompt = (
                f"A chat between a curious user and an artificial intelligence assistant. "
                f"The assistant gives helpful, detailed, and polite answers to the user's questions. "
                f"USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
            )
        else:
            prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        
        return prompt
    
    def preprocess_image(
        self,
        image: Union[Image.Image, str],
        center_crop: bool = False,
        crop_scale: float = 0.9
    ) -> Image.Image:
        """
        图像预处理
        
        Args:
            image: PIL图像或图像路径
            center_crop: 是否进行中心裁剪
            crop_scale: 裁剪比例
        
        Returns:
            预处理后的PIL图像
        """
        import tensorflow as tf
        import numpy as np
        
        # 加载图像
        if isinstance(image, str):
            image = Image.open(image)
        
        image = image.convert("RGB")
        
        # 中心裁剪
        if center_crop:
            # 转换为TF Tensor
            image_array = np.array(image)
            image_tf = tf.convert_to_tensor(image_array)
            orig_dtype = image_tf.dtype
            
            # 转换为float32并归一化到[0,1]
            image_tf = tf.image.convert_image_dtype(image_tf, tf.float32)
            
            # 获取图像尺寸
            height, width = tf.cast(tf.shape(image_tf)[0], tf.float32), tf.cast(tf.shape(image_tf)[1], tf.float32)
            
            # 计算裁剪尺寸
            new_height = height * tf.sqrt(tf.constant(crop_scale))
            new_width = width * tf.sqrt(tf.constant(crop_scale))
            
            # 计算偏移量
            height_offset = (height - new_height) / 2
            width_offset = (width - new_width) / 2
            
            # 构建边界框
            bounding_box = tf.stack([
                height_offset, width_offset,
                height_offset + new_height, width_offset + new_width
            ], axis=0)
            
            # 裁剪并调整大小
            cropped = tf.image.crop_and_resize(
                tf.expand_dims(image_tf, 0),
                tf.expand_dims(bounding_box, 0),
                [0],
                [224, 224]
            )[0]
            
            # 转换回原始数据类型
            cropped = tf.clip_by_value(cropped, 0, 1)
            cropped = tf.image.convert_image_dtype(cropped, orig_dtype, saturate=True)
            
            image = Image.fromarray(cropped.numpy())
            image = image.convert("RGB")
        
        return image
    
    def predict(
        self,
        image: Union[Image.Image, str],
        instruction: str,
        unnorm_key: str = "bridge_orig",
        do_sample: bool = False,
        center_crop: bool = False,
        model_version: str = "v1",
        return_dict: bool = True
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """
        执行动作预测
        
        Args:
            image: 输入图像（PIL Image或路径）
            instruction: 任务指令
            unnorm_key: 数据集归一化键名
            do_sample: 是否使用采样（否则使用贪心解码）
            center_crop: 是否对图像进行中心裁剪
            model_version: 模型版本
            return_dict: 是否返回字典格式结果
        
        Returns:
            预测的动作向量，或包含详细信息的字典
        """
        import numpy as np
        
        # 预处理图像
        image = self.preprocess_image(image, center_crop=center_crop)
        
        # 构建提示词
        prompt = self.build_prompt(instruction, model_version)
        
        # 处理输入
        inputs = self.processor(prompt, image).to(self.device)
        
        # 预测动作
        action = self.model.predict_action(
            **inputs,
            unnorm_key=unnorm_key,
            do_sample=do_sample
        )
        
        if return_dict:
            return {
                "action": action,
                "instruction": instruction,
                "prompt": prompt,
                "device": str(self.device)
            }
        
        return action
    
    def predict_batch(
        self,
        images: list,
        instructions: list,
        unnorm_key: str = "bridge_orig",
        do_sample: bool = False
    ) -> list:
        """
        批量预测
        
        注意：OpenVLA目前对批量大小有限制，建议每次处理单个样本
        
        Args:
            images: 图像列表
            instructions: 指令列表
            unnorm_key: 归一化键名
            do_sample: 是否使用采样
        
        Returns:
            动作向量列表
        """
        results = []
        for image, instruction in zip(images, instructions):
            action = self.predict(
                image=image,
                instruction=instruction,
                unnorm_key=unnorm_key,
                do_sample=do_sample,
                return_dict=False
            )
            results.append(action)
        
        return results


def create_inference_pipeline(
    model_path: str = "openvla/openvla-7b",
    quantization: bool = False,
    quantization_mode: str = "int8",
    use_flash_attention: bool = True,
    device: Optional[str] = None
) -> OpenVLAInference:
    """
    创建推理流水线的便捷函数
    
    Args:
        model_path: 模型路径或HuggingFace模型ID
        quantization: 是否使用量化
        quantization_mode: 量化模式
        use_flash_attention: 是否使用Flash Attention
        device: 设备选择
    
    Returns:
        OpenVLAInference推理器实例
    """
    # 自动设备选择
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # 加载模型
    loader = OpenVLALoader()
    
    if quantization:
        model, processor = loader.load_quantized(
            model_path,
            quantization_mode=quantization_mode
        )
    elif device == "cpu":
        model, processor = loader.load_cpu(model_path)
    else:
        model, processor = loader.load_standard(
            model_path,
            use_flash_attention=use_flash_attention,
            device=device
        )
    
    # 创建推理器
    inference = OpenVLAInference(model, processor)
    
    return inference


# 使用示例
if __name__ == "__main__":
    # 示例1：标准加载（需要GPU）
    print("=" * 60)
    print("示例1：标准加载（BF16 + Flash Attention）")
    print("=" * 60)
    
    try:
        loader = OpenVLALoader()
        model, processor = loader.load_standard(
            "openvla/openvla-7b",
            use_flash_attention=True
        )
        
        inference = OpenVLAInference(model, processor)
        
        # 创建测试图像
        test_image = Image.new("RGB", (224, 224), color=(255, 255, 255))
        
        # 预测动作
        result = inference.predict(
            image=test_image,
            instruction="pick up the red block",
            unnorm_key="bridge_orig"
        )
        
        print(f"预测动作: {result['action']}")
        print(f"指令: {result['instruction']}")
        
    except Exception as e:
        print(f"标准加载失败（可能无GPU）: {e}")
        print("尝试CPU加载...")
    
    print("\n" + "=" * 60)
    print("示例2：CPU加载")
    print("=" * 60)
    
    try:
        loader = OpenVLALoader()
        model, processor = loader.load_cpu("./openvla-7b")
        
        inference = OpenVLAInference(model, processor)
        
        # 创建测试图像
        test_image = Image.new("RGB", (224, 224), color=(200, 200, 200))
        
        # 预测动作
        result = inference.predict(
            image=test_image,
            instruction="move forward",
            unnorm_key="bridge_orig"
        )
        
        print(f"预测动作: {result['action']}")
        
    except Exception as e:
        print(f"CPU加载失败: {e}")
    
    print("\n" + "=" * 60)
    print("示例3：量化加载")
    print("=" * 60)
    
    try:
        loader = OpenVLALoader()
        
        # INT8量化
        model, processor = loader.load_quantized(
            "openvla/openvla-7b",
            quantization_mode="int8"
        )
        
        inference = OpenVLAInference(model, processor)
        
        test_image = Image.new("RGB", (224, 224), color=(150, 150, 150))
        
        result = inference.predict(
            image=test_image,
            instruction="grasp the object",
            unnorm_key="bridge_orig"
        )
        
        print(f"INT8量化预测动作: {result['action']}")
        
    except Exception as e:
        print(f"量化加载需要GPU或bitsandbytes库: {e}")
    
    print("\n" + "=" * 60)
    print("所有示例执行完成！")
    print("=" * 60)
