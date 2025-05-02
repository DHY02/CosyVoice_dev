#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
. ./path.sh || exit 1;

stage=0
stop_stage=5

# 训练数据集
train_corpus="casia"

# 训练方法
method="dpo"

data_dir=/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/data/${train_corpus}_${method}
pretrained_model_dir=/root/autodl-tmp/CosyVoice_dev/pretrained_models/CosyVoice2-0.5B


datasets="train valid"
dpo_datasets="receive reject"

# 参数
beta=0.01

# 实验名称（保存目录）
exp_name="${train_corpus}_${method}_${beta}"

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
    cp $data_dir/$x/reject/utt2speech_token.pt $data_dir/$x/receive/utt2reject_speech_token.pt
  done
fi


# DPO: 只有receive数据子集需要把reject token打包到batch中
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare required parquet format data, you should have prepared wav.scp/text/utt2spk/spk2utt/utt2embedding.pt/spk2embedding.pt/utt2speech_token.pt"
  for x in ${datasets}; do
    mkdir -p $data_dir/$x/receive/parquet
    tools/make_parquet_list_dpo.py --num_utts_per_parquet 1000 \
      --num_processes 10 \
      --src_dir $data_dir/$x/receive \
      --des_dir $data_dir/$x/receive/parquet \
      --dpo
  done
fi

# inference
# if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
#   echo "Run inference. Please make sure utt in tts_text is in prompt_data"
#   # TODO consider remove bin/inference.py, or use similar initilization method as in readme
#   for mode in instruct; do
#     python cosyvoice/bin/inference.py --mode $mode \
#       --gpu 0 \
#       --config conf/cosyvoice2_dpo_infer.yaml \
#       --prompt_data data/casia/parquet/data.list \
#       --prompt_utt2data data/casia/parquet/utt2data.list \
#       --tts_text `pwd`/wav2tts_text_dpo_1200_test.json \
#       --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
#       --llm_model $pretrained_model_dir/llm_sft_dpo_1.pt \
#       --flow_model $pretrained_model_dir/flow.pt \
#       --hifigan_model $pretrained_model_dir/hift.pt \
#       --result_dir `pwd`/exp/cosyvoice/test_dpo_1200_DPO/$mode
#   done
# fi

# train llm
export CUDA_VISIBLE_DEVICES="0"
num_gpus=1
job_id=1986
dist_backend="nccl"
num_workers=2
prefetch=100
train_engine=torch_ddp
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Run train. We only support llm traning for now. If your want to train from scratch, please use conf/cosyvoice.fromscratch.yaml"
  if [ $train_engine == 'deepspeed' ]; then
    echo "Notice deepspeed has its own optimizer config. Modify conf/ds_stage2.json if necessary"
  fi
  cat $data_dir/train/receive/parquet/data.list > $data_dir/train.data.list
  cat $data_dir/valid/receive/parquet/data.list > $data_dir/dev.data.list
  # NOTE will update llm/hift training later
  # --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
  for model in llm; do
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
      cosyvoice/bin/train_dpo.py \
      --train_engine $train_engine \
      --config conf/cosyvoice2_dpo.yaml \
      --train_data $data_dir/train.data.list \
      --cv_data $data_dir/dev.data.list \
      --model $model \
      --checkpoint $pretrained_model_dir/${model}_ori.pt \
      --model_dir `pwd`/exp/cosyvoice2/$model/$train_engine/$exp_name \
      --tensorboard_dir `pwd`/tensorboard/cosyvoice2/$model/$train_engine/$exp_name \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --use_amp \
      --deepspeed_config ./conf/ds_stage2.json \
      --deepspeed.save_states model+optimizer \
      --dpo \
      --beta ${beta}
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