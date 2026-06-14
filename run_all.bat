
@echo off
REM OpenVLA 一键运行脚本（Windows）
REM 执行顺序：环境验证 → 数据下载 → 数据预处理 → 模型训练 → 模型评估 → 结果可视化

echo ==========================================
echo   OpenVLA 项目一键运行脚本
echo ==========================================

REM 1. 环境验证
echo.
echo [1/6] 环境验证...
python demo.py

REM 2. 数据下载
echo.
echo [2/6] 下载数据集...
python download_libero.py

REM 3. 数据预处理
echo.
echo [3/6] 数据预处理...
python preprocess_libero.py

REM 4. 模型训练
echo.
echo [4/6] 模型训练...
python train_lora.py --config configs/train_config.yaml

REM 5. 模型评估
echo.
echo [5/6] 模型评估...
python evaluate.py --model runs/best_model.pt --data ./data/processed_libero/test

REM 6. 结果可视化
echo.
echo [6/6] 结果可视化...
python visualize_results.py --metrics results/eval_*/metrics.json

echo.
echo ==========================================
echo   所有任务已完成！
echo ==========================================
echo 训练日志: runs/
echo 评估结果: results/
echo 可视化结果: visualizations/
pause
