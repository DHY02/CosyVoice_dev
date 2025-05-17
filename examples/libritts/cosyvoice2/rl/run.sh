#!/bin/bash

# 基础路径
COSYVOICE_DIR="/root/autodl-tmp/CosyVoice_dev"

# 默认参数值
TRAIN_CORPUS="esd"
TEST_CORPUS="esd"
METHOD="grpo"
beta=0.04
clip=0.2

TRAINING_FRAMEWORK="deepspeed"

# 训练轮次设置
start_epoch=0
stop_epoch=4
epoch_interval=1

# 执行阶段设置
stage=1
stop_stage=3

exp_name=""
# 参数解析，支持传入部分超参数
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -c|--train_corpus)
      TRAIN_CORPUS="$2"
      shift 2
      ;;
    -t|--test_corpus)
      TEST_CORPUS="$2"
      shift 2
      ;;
    -m|--method)
      METHOD="$2"
      shift 2
      ;;
    --beta)
      beta="$2"
      shift 2
      ;;
    --clip)
      clip="$2"
      shift 2
      ;;
    -s|--stage)
      stage="$2"
      shift 2
      ;;
    -e|--stop_stage)
      stop_stage="$2"
      shift 2
      ;;
    --start_epoch)
      start_epoch="$2"
      shift 2
      ;;
    --stop_epoch)
      stop_epoch="$2"
      shift 2
      ;;
    --epoch_interval)
      epoch_interval="$2"
      shift 2
      ;;
    --framework)
      TRAINING_FRAMEWORK="$2"
      shift 2
      ;;
    --exp_name)
      EXP_NAME="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

MODEL_SRC_DIR="${COSYVOICE_DIR}/examples/libritts/cosyvoice2/exp/cosyvoice2/llm/${TRAINING_FRAMEWORK}/${EXP_NAME}"
MODEL_DEST_DIR="${COSYVOICE_DIR}/pretrained_models/CosyVoice2-0.5B"
INFER_SCRIPT="${COSYVOICE_DIR}/examples/libritts/cosyvoice2/rl/test_vllm_infer.py"
EVALUATE_SCRIPT="${COSYVOICE_DIR}/examples/libritts/cosyvoice2/rl/evaluate/evaluate.py"
VISUALIZE_SCRIPT="${COSYVOICE_DIR}/examples/libritts/cosyvoice2/rl/evaluate/visualize.py"
CONVERT_SCRIPT="${COSYVOICE_DIR}/examples/libritts/cosyvoice2/rl/preprocess/copy_model.py"

echo "开始时间: $(date)"
echo "目标方法：${METHOD}"
echo "训练语料: ${TRAIN_CORPUS}"
echo "测试语料: ${TEST_CORPUS}"
echo "超参数: beta=${beta}, clip=${clip}"
echo "实验名称: ${EXP_NAME}"
echo "源模型目录: ${MODEL_SRC_DIR}"
echo "目标模型目录: ${MODEL_DEST_DIR}"

# 检查目录是否存在
if [ ! -d "${MODEL_SRC_DIR}" ]; then
    echo "错误: 源模型目录不存在: ${MODEL_SRC_DIR}"
    exit 1
fi

if [ ! -d "${MODEL_DEST_DIR}" ]; then
    echo "错误: 目标模型目录不存在: ${MODEL_DEST_DIR}"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "==== stage 1: Inference ===="
    for ((i=start_epoch; i<=stop_epoch; i+=epoch_interval)); do

        echo "正在推理 ${METHOD} epoch ${i}..."

        # 调用Python脚本处理模型
        echo "处理模型文件..."
        python "${CONVERT_SCRIPT}" \
            --framework "${TRAINING_FRAMEWORK}" \
            --src_dir "${MODEL_SRC_DIR}" \
            --dst_dir "${MODEL_DEST_DIR}" \
            --epoch "${i}" || { echo "模型处理失败"; exit 1; }

        # 执行Python推理脚本
        echo "执行推理脚本..."
        SAMPLING_TEMPERATURE=1 SAMPLING_TOP_P=1 SAMPLING_TOP_K=25 \
        python "${INFER_SCRIPT}" \
                --num_epoch "${i}" \
                --exp_name "${EXP_NAME}" \
                --test_corpus "${TEST_CORPUS}" || { echo "推理失败"; exit 1; }
        
        echo "epoch ${i} 推理完成"
        echo "----------------------------------"
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "==== stage 2: Evaluate ===="
    for ((i=start_epoch; i<=stop_epoch; i+=epoch_interval)); do
        echo "正在评估 ${METHOD} epoch ${i}..."
        
        # 执行Python评估脚本
        echo "执行评估脚本..."
        python "${EVALUATE_SCRIPT}" \
                --num_epoch "${i}" \
                --method "${METHOD}" \
                --train_corpus "${TRAIN_CORPUS}" \
                --test_corpus "${TEST_CORPUS}"

        if [ $? -ne 0 ]; then
            echo "评估失败，但继续执行后续任务"
        fi
        
        echo "epoch ${i} 评估完成"
        echo "----------------------------------"
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "==== stage 3: Visualize ${METHOD} ===="
    echo "执行可视化脚本..."
    python "${VISUALIZE_SCRIPT}" \
            --train_corpus "${TRAIN_CORPUS}" \
            --test_corpus "${TEST_CORPUS}"

    if [ $? -ne 0 ]; then
        echo "可视化失败，但继续执行后续任务"
    else
        echo "${METHOD} 评估结果可视化完成"
    fi
fi

echo "任务结束时间: $(date)"
echo "${METHOD} 推理、评估、可视化任务执行完毕"