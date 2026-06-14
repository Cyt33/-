
"""
OpenVLA 模型评估脚本

功能：
- 加载训练好的模型进行推理测试
- 计算任务成功率、推理耗时等指标
- 支持对比实验（微调前后效果对比）
- 输出评估报告

运行命令：
    # 基础评估
    python evaluate.py --model runs/best_model.pt --data ./data/processed_libero/test
    
    # 对比评估（微调前后）
    python evaluate.py --model runs/best_model.pt --baseline openvla/openvla-7b \
        --data ./data/processed_libero/test
"""

import os
import json
import time
import argparse
import logging
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from transformers import AutoModelForVision2Seq, AutoProcessor
from data_loader import LIBEROCleanDataset

# 设置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="OpenVLA Model Evaluation")
    parser.add_argument("--model", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--baseline", type=str, default=None, help="Path to baseline model")
    parser.add_argument("--data", type=str, required=True, help="Path to test dataset")
    parser.add_argument("--output", type=str, default="./results", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to evaluate")
    return parser.parse_args()


def load_model(model_path):
    """加载模型"""
    logger.info(f"Loading model: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        device_map='auto',
        torch_dtype=torch.float16
    )
    model.eval()
    return model, processor


def compute_metrics(predictions, ground_truth):
    """计算评估指标"""
    metrics = {}
    
    # 动作预测误差（MSE）
    mse = np.mean((predictions - ground_truth) ** 2)
    metrics['mse'] = float(mse)
    
    # 动作预测准确率（误差小于阈值的比例）
    threshold = 0.1
    accuracy = np.mean(np.all(np.abs(predictions - ground_truth) < threshold, axis=-1))
    metrics['accuracy'] = float(accuracy)
    
    # 任务成功率（简化版：轨迹级别的成功率）
    success_rate = np.mean(np.all(np.abs(predictions - ground_truth) < threshold, axis=(1, 2)))
    metrics['success_rate'] = float(success_rate)
    
    return metrics


def evaluate_model(model, processor, dataloader, num_samples=None):
    """评估模型"""
    all_predictions = []
    all_ground_truth = []
    inference_times = []
    
    total_samples = num_samples if num_samples else len(dataloader.dataset)
    processed_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if processed_samples >= total_samples:
                break
            
            images = batch['images'].to('cuda', dtype=torch.float16)
            instructions = batch['instructions']
            actions = batch['actions'].cpu().numpy()
            
            # 记录推理时间
            start_time = time.time()
            
            # 推理
            inputs = processor(images=images, text=instructions, return_tensors="pt").to('cuda', dtype=torch.float16)
            outputs = model.generate(**inputs)
            
            # 解码动作
            predictions = processor.decode(outputs, skip_special_tokens=True)
            
            end_time = time.time()
            inference_times.append(end_time - start_time)
            
            all_predictions.extend(predictions)
            all_ground_truth.extend(actions)
            
            processed_samples += len(images)
    
    # 转换为 numpy 数组
    all_predictions = np.array(all_predictions)
    all_ground_truth = np.array(all_ground_truth)
    
    # 计算指标
    metrics = compute_metrics(all_predictions, all_ground_truth)
    metrics['avg_inference_time'] = float(np.mean(inference_times))
    metrics['num_samples'] = processed_samples
    
    return metrics, all_predictions, all_ground_truth


def save_results(results, output_dir):
    """保存评估结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存指标
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Results saved to {output_dir}")


def main():
    args = parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output) / f"eval_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载测试数据集
    logger.info(f"Loading test dataset from {args.data}")
    test_dataset = LIBEROCleanDataset(subset='test', data_root=args.data)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # 评估微调后模型
    logger.info("Evaluating fine-tuned model...")
    model, processor = load_model(args.model)
    ft_metrics, ft_preds, ft_gts = evaluate_model(model, processor, test_loader, args.num_samples)
    
    results = {
        'fine_tuned': ft_metrics
    }
    
    # 评估基线模型（如果提供）
    if args.baseline:
        logger.info("Evaluating baseline model...")
        baseline_model, baseline_processor = load_model(args.baseline)
        baseline_metrics, baseline_preds, baseline_gts = evaluate_model(
            baseline_model, baseline_processor, test_loader, args.num_samples
        )
        results['baseline'] = baseline_metrics
        
        # 计算提升
        results['improvement'] = {
            'mse': (baseline_metrics['mse'] - ft_metrics['mse']) / baseline_metrics['mse'] * 100,
            'accuracy': (ft_metrics['accuracy'] - baseline_metrics['accuracy']) / baseline_metrics['accuracy'] * 100,
            'success_rate': (ft_metrics['success_rate'] - baseline_metrics['success_rate']) / baseline_metrics['success_rate'] * 100
        }
    
    # 保存结果
    save_results(results, output_dir)
    
    # 打印结果
    logger.info("\n=== Evaluation Results ===")
    for key, value in results.items():
        if isinstance(value, dict):
            logger.info(f"\n{key}:")
            for k, v in value.items():
                if isinstance(v, float):
                    logger.info(f"  {k}: {v:.4f}")
                else:
                    logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
