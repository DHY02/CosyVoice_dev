#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
. ./path.sh || exit 1;

stage=5
stop_stage=5

# data_url=www.openslr.org/resources/60
data_dir=/home/CosyVoice/examples/libritts/cosyvoice2/data
pretrained_model_dir=/home/pretrained_models/CosyVoice2-0.5B

receive="angry_receive_train"
reject="angry_reject_train"
valid="angry_dpo_valid"
test="angry_dpo_test"
datasets="${receive} ${reject} ${valid} ${test}"

# if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
#   echo "Data Download"
#   for part in casia_train casia_test; do
#     local/download_and_untar.sh ${data_dir} ${data_url} ${part}
#   done
# fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Data preparation, prepare wav.scp/text/utt2spk/spk2utt"
  for x in ${datasets}; do
    mkdir -p data/$x
    python local/prepare_data.py --src_dir $data_dir/$x --des_dir data/$x
  done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Extract campplus speaker embedding, you will get spk2embedding.pt and utt2embedding.pt in data/$x dir"
  for x in ${datasets}; do
    tools/extract_embedding.py --dir data/$x \
      --onnx_path $pretrained_model_dir/campplus.onnx
  done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Extract discrete speech token, you will get utt2speech_token.pt in data/$x dir"
  for x in ${datasets}; do
    tools/extract_speech_token.py --dir data/$x \
      --onnx_path $pretrained_model_dir/speech_tokenizer_v2.onnx
  done
  cp data/${reject}/utt2speech_token.pt data/${receive}/utt2reject_speech_token.pt
fi



if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare required parquet format data, you should have prepared wav.scp/text/utt2spk/spk2utt/utt2embedding.pt/spk2embedding.pt/utt2speech_token.pt"
  for x in ${receive}; do
    mkdir -p data/$x/parquet
    tools/make_parquet_list_dpo.py --num_utts_per_parquet 1000 \
      --num_processes 10 \
      --src_dir data/$x \
      --des_dir data/$x/parquet \
      --dpo
  done

  for x in ${valid} ${test}; do
    mkdir -p data/$x/parquet
    tools/make_parquet_list_dpo.py --num_utts_per_parquet 1000 \
      --num_processes 10 \
      --src_dir data/$x \
      --des_dir data/$x/parquet
  done
fi

# inference
# if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
#   echo "Run inference. Please make sure utt in tts_text is in prompt_data"
#   # TODO consider remove bin/inference.py, or use similar initilization method as in readme
#   for mode in sft; do
#     python cosyvoice/bin/inference.py --mode $mode \
#       --gpu 0 \
#       --config conf/cosyvoice2.yaml \
#       --prompt_data data/casia_test/parquet/data.list \
#       --prompt_utt2data data/casia_test/parquet/utt2data.list \
#       --tts_text `pwd`/tts_text_angry50_test.json \
#       --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
#       --llm_model $pretrained_model_dir/llm.pt \
#       --flow_model $pretrained_model_dir/flow.pt \
#       --hifigan_model $pretrained_model_dir/hift.pt \
#       --result_dir `pwd`/exp/cosyvoice/test-angry/$mode
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
  cat data/${receive}/parquet/data.list > data/train.data.list
  cat data/${valid}/parquet/data.list > data/dev.data.list
  # NOTE will update llm/hift training later
  # --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
  for model in llm; do
    torchrun --nnodes=1 --nproc_per_node=$num_gpus \
        --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint="localhost:1234" \
      cosyvoice/bin/train_dpo.py \
      --train_engine $train_engine \
      --config conf/cosyvoice2_dpo.yaml \
      --train_data data/train.data.list \
      --cv_data data/dev.data.list \
      --model $model \
      --checkpoint $pretrained_model_dir/$model.pt \
      --model_dir `pwd`/exp/cosyvoice2/$model/$train_engine \
      --tensorboard_dir `pwd`/tensorboard/cosyvoice2/$model/$train_engine \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --use_amp \
      --deepspeed_config ./conf/ds_stage2.json \
      --deepspeed.save_states model+optimizer \
      --dpo
  done
fi

# average model
average_num=5
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  for model in llm flow hifigan; do
    decode_checkpoint=`pwd`/exp/cosyvoice/$model/$train_engine/${model}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python cosyvoice/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path `pwd`/exp/cosyvoice/$model/$train_engine  \
      --num ${average_num} \
      --val_best
  done
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Export your model for inference speedup. Remember copy your llm or flow model to model_dir"
  python cosyvoice/bin/export_jit.py --model_dir $pretrained_model_dir
  python cosyvoice/bin/export_onnx.py --model_dir $pretrained_model_dir
fi