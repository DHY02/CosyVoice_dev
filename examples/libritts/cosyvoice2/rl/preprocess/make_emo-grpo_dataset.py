from pathlib import Path
import os
import shutil
# 将同一文本的其他五种情感作为负样本存为samp_1, ..., samp_5
emotions = ["angry", "fear", "happy", "neutral", "sad", "surprise"]
# 第i个情感e的samp_j = emotions[(e.index + j) % 6]

# id: {e1: path1; e2: path2, ...}

# 对于每个正样本，根据上述算法找到改情感对应所有的samp，然后将这些samp命名为该正样本的wav id，并放入对应的samp文件夹，并给予对应的奖励
dataset_dir = Path("/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/data/casia-emo-grpo")

for cor in ["train", "valid"]:
    for entry in os.listdir(dataset_dir / cor / "receive"):
        if entry.endswith('.wav'):
            reject_list = []
            wav_name = entry.split('.')[0]
            text_name = wav_name + ".normalized.txt"
            emo = wav_name.split('_')[-1].strip()
            aid = "_".join(wav_name.split('_')[:-1])
            i = emotions.index(emo)

            # 遍历每个负样本
            for j in range(1, 6):
                tgt_dir = dataset_dir / cor / f"samp_{j}"
                os.makedirs(tgt_dir, exist_ok=True)
                reject_emo = emotions[(i + j) % 6]
                reject_wav = aid + "_" + reject_emo + ".wav"
                shutil.copy(dataset_dir / cor / "receive" / reject_wav, tgt_dir / entry)
                shutil.copy(dataset_dir / cor / "receive" / text_name, tgt_dir / text_name)
                with open (tgt_dir / f"{wav_name}.adv", "w", encoding="utf-8") as f:
                    f.write("0")