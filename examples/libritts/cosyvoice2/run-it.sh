#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
. ./path.sh || exit 1;
set -e

# 定义必须传入的参数列表
required_params=("corpus" "method")

# 训练数据集
train_corpus_name=""

# 训练方法, instruction tuning
method=""

# 存储已处理的参数
processed_params=()
stage=3
stop_stage=5

# 默认参数
lr=1e-5
datasets="train valid"
dpo_datasets="receive"
train_engine="deepspeed"

# 参数解析，支持传入部分超参数
while [[ $# -gt 0 ]]; do
  key="$1"
  processed=false
  
  case $key in
    -c|--corpus)
      train_corpus_name="$2"
      processed_params+=("corpus")
      processed=true
      shift 2
      ;;
    -s|--stage)
      stage="$2"
      processed=true
      shift 2
      ;;
    -e|--stop_stage)
      stop_stage="$2"
      processed=true
      shift 2
      ;;
    -m|--method)
      method="$2"
      processed_params+=("method")
      processed=true
      shift 2
      ;;
    --train_engine)
      train_engine="$2"
      processed=true
      shift 2
      ;;
    --lr)
      lr="$2"
      processed=true
      shift 2
      ;;
    --beta|--clip|--start_epoch|--grpo_datasets)
      # 这些参数对IT方法不需要，但为了兼容run.sh的调用，需要接收它们
      shift 2
      processed=true
      ;;
    *)
      echo "警告: 未知参数 '$key' 将被忽略"
      processed=true
      shift
      ;;
  esac
  
  # 如果参数未被处理，发出警告并跳过
  if ! $processed; then
    echo "警告: 参数 '$key' 处理失败，将被忽略"
    shift
  fi
done

# 检查必须参数是否都已提供
missing_params=()
for param in "${required_params[@]}"; do
  if ! [[ " ${processed_params[@]} " =~ " ${param} " ]]; then
    missing_params+=("$param")
  fi
done

# 如果有缺失参数，输出错误并退出
if [ ${#missing_params[@]} -ne 0 ]; then
  echo "错误: 以下必须参数未提供: ${missing_params[*]}"
  echo "用法示例: bash run-it.sh --corpus esd --method it"
  exit 1
fi

data_dir=/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/data/${train_corpus_name}
pretrained_model_dir=/root/autodl-tmp/CosyVoice_dev/pretrained_models/CosyVoice2-0.5B

# 实验名称（保存目录）
exp_name="${train_corpus_name}_${method}|b0c0s0l${lr}"

echo "训练数据集: $train_corpus_name"
echo "训练方法: $method"
echo "开始阶段: $stage"
echo "结束阶段: $stop_stage"
echo "训练引擎: $train_engine"
echo "实验名称: $exp_name"

pretrained_model_dir=/root/autodl-tmp/CosyVoice_dev/pretrained_models/CosyVoice2-0.5B
init_model_dir=/root/autodl-tmp/CosyVoice_dev/pretrained_models/CosyVoice2-0.5B-bk
cp ${init_model_dir}/llm.pt ${pretrained_model_dir}/llm.pt || exit 1

# 训练前：先修改第三步参数并运行，跑完第三步后再开始训练

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Data preparation, prepare wav.scp/text/utt2spk/spk2utt"
  for x in ${datasets}; do
    mkdir -p $data_dir/$x
    for y in ${dpo_datasets}; do
      python local/prepare_data.py --src_dir $data_dir/$x/$y --des_dir $data_dir/$x/$y
    done
  done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  for x in ${datasets}; do
    echo "Extract campplus speaker embedding, you will get spk2embedding.pt and utt2embedding.pt in $data_dir/$x dir"
    for y in ${dpo_datasets}; do
      tools/extract_embedding.py --dir $data_dir/$x/$y \
        --onnx_path $pretrained_model_dir/campplus.onnx
    done
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Extract discrete speech token, you will get utt2speech_token.pt in $data_dir/$x dir"
  for x in ${datasets}; do
    for y in ${dpo_datasets}; do
      tools/extract_speech_token.py --dir $data_dir/$x/$y \
        --onnx_path $pretrained_model_dir/speech_tokenizer_v2.onnx
    done
  done
fi


# DPO: 只有receive数据子集需要把reject token打包到batch中
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare required parquet format data, you should have prepared wav.scp/text/utt2spk/spk2utt/utt2embedding.pt/spk2embedding.pt/utt2speech_token.pt"
  for x in ${datasets}; do
    mkdir -p $data_dir/$x/receive/parquet
    tools/make_parquet_list_rl.py --num_utts_per_parquet 1000 \
      --num_processes 10 \
      --src_dir $data_dir/$x/receive \
      --des_dir $data_dir/$x/receive/parquet
  done
fi

# train llm
export CUDA_VISIBLE_DEVICES="0"
num_gpus=1
job_id=1986
dist_backend="nccl"
num_workers=2
prefetch=100

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Run train. We only support llm traning for now. If your want to train from scratch, please use conf/cosyvoice.fromscratch.yaml"
  if [ $train_engine == 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
  fi
  cat $data_dir/train/receive/parquet/data.list > $data_dir/train.data.list
  cat $data_dir/valid/receive/parquet/data.list > $data_dir/dev.data.list
  
  for model in llm; do
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
      cosyvoice/bin/train_dpo.py \
      --train_engine $train_engine \
      --config conf/cosyvoice2_it.yaml \
      --train_data $data_dir/train.data.list \
      --cv_data $data_dir/dev.data.list \
      --model $model \
      --checkpoint $pretrained_model_dir/${model}.pt \
      --model_dir `pwd`/exp/cosyvoice2/$model/$train_engine/$exp_name \
      --tensorboard_dir `pwd`/tensorboard/cosyvoice2/$model/$train_engine/$exp_name \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --use_amp \
      --deepspeed_config ./conf/ds_stage2.json \
      --deepspeed.save_states model+optimizer
  done
fi

# average model
average_num=1
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  for model in llm; do
    decode_checkpoint=`pwd`/exp/cosyvoice2/$model/$train_engine/${model}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python cosyvoice/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path `pwd`/exp/cosyvoice2/$model/$train_engine/$exp_name  \
      --num ${average_num} \
      --val_best
  done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Export your model for inference speedup. Remember copy your llm or flow model to model_dir"
  python cosyvoice/bin/export_jit.py --model_dir $pretrained_model_dir
  python cosyvoice/bin/export_onnx.py --model_dir $pretrained_model_dir
fi