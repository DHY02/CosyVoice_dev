from pathlib import Path
import os
import shutil
import glob

# 将同一文本的其他五种情感作为负样本存为samp_1, ..., samp_5
emotions = ["冷静", "快乐", "伤心", "惊喜", "生气"]
emo_cnt = len(emotions)
samp_cnt = 4
assert samp_cnt + 1 == emo_cnt
corpus = "esd"

# 对于CASIA：第i个情感e的第j个负样本samp_j = emotions[(e.index + j) % emo_cnt]

# id: {e1: path1; e2: path2, ...}

# 对于esd数据集：单个说话人的情感顺序（每个情感350）：
# ["Neutral", "Angry", "Happy", "Sad", "Surprise"]
# 所以找到下一个情感语音需要映射：sid_aid_emo1.wav -> sid_{aid + j * 350}_emo2.wav

# 对于每个正样本，根据上述算法找到该情感对应所有的samp，然后将这些samp命名为该正样本的wav id，并放入对应的samp文件夹，并给予对应的奖励
dataset_dir = Path(f"/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/data/{corpus}-emo-grpo")

for cor in ["train", "valid"]:
    for entry in os.listdir(dataset_dir / cor / "receive"):
        if entry.endswith('.wav'):
            reject_list = []
            wav_name = entry.split('.')[0]
            text_name = wav_name + ".normalized.txt"
            emo = wav_name.split('_')[-1].strip()
            aid = "_".join(wav_name.split('_')[:-1])
            i = emotions.index(emo)
            if corpus == "esd":
                # spk id
                sid = aid.split('_')[0]
                aid = aid.split('_')[1]
            # 遍历每个负样本
            for j in range(1, samp_cnt + 1):
                tgt_dir = dataset_dir / cor / f"samp_{j}"
                os.makedirs(tgt_dir, exist_ok=True)
                reject_emo = emotions[(i + j) % emo_cnt]
                reject_wav = aid + "_" + reject_emo + ".wav"
                if corpus == "esd": 
                    pattern = f"{sid}_{((int(aid) + j * 350 - 1) % 1750)+1:06d}_*.wav"
                    search_path = os.path.join(dataset_dir / cor / "receive", pattern)
                    if len(search_path) < 0:
                        raise Exception(f"未找到{search_path}")
                    reject_wav = glob.glob(search_path)[0]
                shutil.copy(dataset_dir / cor / "receive" / reject_wav, tgt_dir / entry)
                shutil.copy(dataset_dir / cor / "receive" / text_name, tgt_dir / text_name)
                # .adv是Emo ACC reward文件
                with open (tgt_dir / f"{wav_name}.adv", "w", encoding="utf-8") as f:
                    f.write("0")

print(f"make emo-grpo-dataset done.")