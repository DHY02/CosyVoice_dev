#!/bin/bash

# eva emo-grpo
# bash run.sh \
#     --stop_epoch 2 \
#     --method emo-grpo \
#     --stage 2 \
#     --stop_stage 2 \
#     --te_stage 2 \
#     --te_stop_stage 3 \
#     --emo_train_start_epoch 2

bash run.sh \
    --stop_epoch 5 \
    --method it \
    --stage 1 \
    --stop_stage 2 \
    --tr_stage 5 \
    --tr_stop_stage 5 \
    --emo_train_start_epoch 0 \
    --epoch_interval 3 \
    --beta 0 \
    --clip 0 

bash run.sh \
    --stop_epoch 5 \
    --method emo-dpo \
    --stage 1 \
    --stop_stage 2 \
    --tr_stage 3 \
    --tr_stop_stage 5 \
    --emo_train_start_epoch 2 \
    --beta 0.01 \
    --clip 0 \
    --lr 3e-7 \
    --checkpoint "/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/exp/cosyvoice2/llm/deepspeed/esd_emo-dpo|b0.01cs2l3e-7/epoch_2_whole/mp_rank_00_model_states.pt" || exit 1







