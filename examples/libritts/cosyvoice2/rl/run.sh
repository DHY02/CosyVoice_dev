#!/bin/bash

# 基础路径
COSYVOICE_DIR="/root/autodl-tmp/CosyVoice_dev"
TRAIN_CORPUS="casia"
METHOD="emo-grpo-olnl|b0.04c0.2s3l1e-5"
MODEL_SRC_DIR="${COSYVOICE_DIR}/examples/libritts/cosyvoice2/exp/cosyvoice2/llm/torch_ddp/${TRAIN_CORPUS}_${METHOD}"
MODEL_DEST_DIR="${COSYVOICE_DIR}/pretrained_models/CosyVoice2-0.5B"
INFER_SCRIPT="${COSYVOICE_DIR}/examples/libritts/cosyvoice2/rl/test_vllm_infer.py"

start_epoch=4
stop_epoch=8
epoch_interval=1
stage=1
stop_stage=3

echo "目标方法：${METHOD}"
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1 Inference"
    for ((i=start_epoch; i<=stop_epoch; i+=epoch_interval)); do
        echo "正在推理 ${METHOD} epoch ${i}..."
        
        # 1. 复制模型文件
        cp "${MODEL_SRC_DIR}/epoch_${i}_whole.pt" "${MODEL_DEST_DIR}/llm.pt" || exit 1
        # 2. 执行Python脚本
        python "${INFER_SCRIPT}" --num_epoch "${i}" --method "${METHOD}"
        
        echo "epoch ${i} 推理完成"
        echo "----------------------------------"
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2 Evaluate"
    for ((i=start_epoch; i<=stop_epoch; i+=epoch_interval)); do
        echo "正在评估 ${METHOD} epoch ${i}..."
        
        # 2. 执行Python脚本
        python evaluate/evaluate.py --num_epoch "${i}" --method "${METHOD}"
        
        echo "epoch ${i} 评估完成"
        echo "----------------------------------"
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3 Visualize ${METHOD}"
    python evaluate/visualize.py
    echo "${METHOD} 评估结果可视化完成"
fi

echo "任务结束时间: $(date)"
echo "${METHOD} 推理、评估、可视化任务执行完毕"
