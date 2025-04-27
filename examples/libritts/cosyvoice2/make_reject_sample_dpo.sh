#!/bin/bash

. ./path.sh || exit 1;

stage=4
stop_stage=4

data_dir=/home/CosyVoice/examples/libritts/cosyvoice2/data
pretrained_model_dir=/home/pretrained_models/CosyVoice2-0.5B



# inference
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Run inference. Please make sure utt in tts_text is in prompt_data"
  num=1
  # TODO consider remove bin/inference.py, or use similar initilization method as in readme
  # zero_shot mode 的tts text不需要angry<|endofprompt|>前缀
  for mode in sft; do
    for sam_conf in cosyvoice2 cosyvoice2_s1 cosyvoice2_s2 cosyvoice2_s3 cosyvoice2_s4; do
      python cosyvoice/bin/inference.py --mode $mode \
        --gpu 0 \
        --config conf/$sam_conf.yaml \
        --prompt_data data/casia_train/parquet/data.list \
        --prompt_utt2data data/casia_train/parquet/utt2data.list \
        --tts_text `pwd`/tts_text_angry50.json \
        --qwen_pretrain_path $pretrained_model_dir/CosyVoice-BlankEN \
        --llm_model $pretrained_model_dir/llm.pt \
        --flow_model $pretrained_model_dir/flow.pt \
        --hifigan_model $pretrained_model_dir/hift.pt \
        --result_dir `pwd`/exp/cosyvoice/casia_train_angry50_inf/$mode/samp_$num
      ((num++))
    done
  done
fi