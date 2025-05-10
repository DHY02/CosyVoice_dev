#!/bin/bash
# 	1. 用模型采样4组, rl/run_sampling.sh和sample_dpo.py
# 	2. 修改并运行make_grpo_dataset.py，将采样结果放入对应数据集。
# 	3. 复制一份esd-grpo数据集为esd-emo-grpo，删除其下samp_i文件夹，修改并运行make_emo-grpo_dataset.py，制作emo-grpo数据集
#   4. 训练
#   5. 评估

# 1
bash run_sampling.sh || exit 1

# 2
python make_grpo_dataset.py || exit 1

# 3
emo_grpo_path="/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/data/esd-emo-grpo"

mkdir -p ${emo_grpo_path} || exit 1

cp -r "/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/data/esd-grpo/"* "${emo_grpo_path}/" || exit 1
rm -rf "${emo_grpo_path}/samp_"* || exit 1

python make_emo-grpo_dataset.py || exit 1

# 4
cd "/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2"

bash run-grpo.sh || exit 1

cd "/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/rl"

bash run.sh || exit 1