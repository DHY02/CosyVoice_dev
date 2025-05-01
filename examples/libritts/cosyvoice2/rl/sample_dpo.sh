#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.
cd ..
. ./path.sh || exit 1;

stage=4

stop_stage=4

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(dirname "$(dirname "$(dirname "$script_dir")")")"

cosyvoice2_dir="${root_dir}/examples/libritts/cosyvoice2"

pretrained_model_dir="${root_dir}/pretrained_models/CosyVoice2-0.5B"

corpus="m3ed/whole"
datasets="${corpus}"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Data preparation, prepare wav.scp/text/utt2spk/spk2utt"
  for x in ${datasets}; do
    mkdir -p data/$x
    python local/prepare_data.py --src_dir data/$x --des_dir data/$x
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
fi



if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare required parquet format data, you should have prepared wav.scp/text/utt2spk/spk2utt/utt2embedding.pt/spk2embedding.pt/utt2speech_token.pt"
  for x in ${datasets}; do
    mkdir -p data/$x/parquet
    tools/make_parquet_list_dpo.py --num_utts_per_parquet 1000 \
      --num_processes 10 \
      --src_dir data/$x \
      --des_dir data/$x/parquet
  done
fi

# inference
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Run inference. Please make sure utt in tts_text is in prompt_data"
  num=1
  # TODO consider remove bin/inference.py, or use similar initilization method as in readme
  # zero_shot mode 的tts text不需要angry<|endofprompt|>前缀
  # 17 hours to sample once, 51 hours to sample three times
  for mode in instruct; do
    for sam_conf in cosyvoice2_s2 cosyvoice2_s3 cosyvoice2_s4; do
      echo "sample with conf ${sam_conf}"
      python cosyvoice/bin/inference.py --mode $mode \
        --gpu 0 \
        --config conf/$sam_conf.yaml \
        --prompt_data data/m3ed/whole/parquet/data.list \
        --prompt_utt2data data/m3ed/whole/parquet/utt2data.list \
        --tts_text `pwd`/wav2text_samp.json \
        --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
        --llm_model $pretrained_model_dir/llm_init.pt \
        --flow_model $pretrained_model_dir/flow.pt \
        --hifigan_model $pretrained_model_dir/hift.pt \
        --result_dir `pwd`/exp/cosyvoice/dpo_samp/$mode/samp_$num
      ((num++))
    done
  done
fi