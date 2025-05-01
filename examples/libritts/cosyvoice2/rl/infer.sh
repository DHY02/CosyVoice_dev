#!/bin/bash
# Copyright 2024 Alibaba Inc. All Rights Reserved.

cd ..
. ./path.sh || exit 1;

stage=4
stop_stage=4

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(dirname "$(dirname "$(dirname "$script_dir")")")"

cosyvoice2_dir="${root_dir}/examples/libritts/cosyvoice2"
data_dir="${cosyvoice2_dir}/data"
pretrained_model_dir="${root_dir}/pretrained_models/CosyVoice2-0.5B"

wav2text_name="wav2text_m3ed.json"
llm_name="llm_init.pt"
result_dir_name="test_init"


test="m2ed_test"
ref="casia"
datasets="${test}"

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
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Run inference. Please make sure utt in tts_text is in prompt_data"
  # TODO consider remove bin/inference.py, or use similar initilization method as in readme
  for mode in instruct; do
    python cosyvoice/bin/inference.py --mode $mode \
      --gpu 0 \
      --config conf/cosyvoice2_dpo_infer.yaml \
      --prompt_data data/casia/parquet/data.list \
      --prompt_utt2data data/casia/parquet/utt2data.list \
      --tts_text `pwd`/${wav2text_name} \
      --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
      --llm_model $pretrained_model_dir/${llm_name} \
      --flow_model $pretrained_model_dir/flow.pt \
      --hifigan_model $pretrained_model_dir/hift.pt \
      --result_dir `pwd`/exp/cosyvoice/${result_dir_name}/$mode
  done
fi

