
#!/bin/bash
# OpenVLA 一键运行脚本
# 执行顺序：环境验证 → 数据下载 → 数据预处理 → 模型训练 → 模型评估 → 结果可视化

set -e  # 遇到错误立即退出

echo "=========================================="
echo "  OpenVLA 项目一键运行脚本"
echo "=========================================="

# 1. 环境验证
echo ""
echo "[1/6] 环境验证..."
python demo.py

# 2. 数据下载
echo ""
echo "[2/6] 下载数据集..."
python download_libero.py

# 3. 数据预处理
echo ""
echo "[3/6] 数据预处理..."
python preprocess_libero.py

# 4. 模型训练
echo ""
echo "[4/6] 模型训练..."
python train_lora.py --config configs/train_config.yaml

# 5. 模型评估
echo ""
echo "[5/6] 模型评估..."
MODEL_DIR=$(ls -td runs/openvla_lora_* | head -1)
BEST_MODEL="$MODEL_DIR/best_model.pt"
python evaluate.py --model "$BEST_MODEL" --data ./data/processed_libero/test

# 6. 结果可视化
echo ""
echo "[6/6] 结果可视化..."
RESULTS_DIR=$(ls -td results/eval_* | head -1)
python visualize_results.py --metrics "$RESULTS_DIR/metrics.json"
python visualize_results.py --log_dir "$MODEL_DIR/tensorboard"

echo ""
echo "=========================================="
echo "  所有任务已完成！"
echo "=========================================="
echo "训练日志: $MODEL_DIR"
echo "评估结果: $RESULTS_DIR"
echo "可视化结果: visualizations/"
