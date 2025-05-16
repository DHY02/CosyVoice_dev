#!/bin/bash

# evaluate
cd "/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/rl" || exit 1
bash run.sh || exit 1
echo "评估结束"

# train emo-grpo
cd /root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2 || exit 1
bash run-emo-grpo-nl.sh || exit 1
echo "emo-grpo 训练结束"

