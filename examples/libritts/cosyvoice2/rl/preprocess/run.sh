#!/bin/bash
set -e
# 	1. 用模型采样4组, rl/run_sampling.sh和sample_dpo.py
# 	2. 修改并运行make_grpo_dataset.py，将采样结果放入对应数据集。
# 	3. 复制一份esd-grpo数据集为esd-emo-grpo，删除其下samp_i文件夹，修改并运行make_emo-grpo_dataset.py，制作emo-grpo数据集
#   4. 训练
#   5. 评估

model_path=/root/autodl-tmp/CosyVoice_dev/pretrained_models/CosyVoice2-0.5B

corpus="esd"
stage=4
stop_stage=5
cp /root/autodl-tmp/CosyVoice_dev/pretrained_models/CosyVoice2-0.5B-bk/llm.pt /root/autodl-tmp/CosyVoice_dev/pretrained_models/CosyVoice2-0.5B/llm.pt || exit 1
# 1
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    bash run_sampling.sh || exit 1
fi

# 2
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    python make_grpo_dataset.py || exit 1
    echo "make_grpo_dataset完成"
fi

# 3
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage3：复制一份esd-grpo数据集为esd-emo-grpo，删除其下samp_i文件夹，运行make_emo-grpo_dataset.py，制作emo-grpo数据集"
    emo_grpo_path="/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/data/esd-emo-grpo"
    mkdir -p ${emo_grpo_path} || exit 1

    src_dir="/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/data/${corpus}"
    cp -r ${src_dir}/* "${emo_grpo_path}/" || exit 1
    find "${emo_grpo_path}" -mindepth 2 -type d -name "samp_*" -exec rm -rf {} + || exit 1

    python make_emo-grpo_dataset.py || exit 1
fi

# 4
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage4: 训练GRPO"
    cd "/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2" || exit 1

    bash run-grpo.sh -c $corpus -s 5 -e 5 || exit 1
fi

# 5
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage5: 评估"
    cd "/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/rl" || exit 1
    bash run.sh || exit 1
fi