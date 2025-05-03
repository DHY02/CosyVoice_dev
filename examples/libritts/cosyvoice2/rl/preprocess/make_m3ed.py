# 从EmoBox/data/m3ed的raw audio和其来自EMOBox的标注json创建libritts-like的cosyvoice2微调数据集
from pathlib import Path
import json
import os
import shutil

pwd = Path(__file__)


dataset="m3ed"

cosyvoice2_dir = pwd.parent.parent.parent
# cosyvoice data文件夹
data_dir = cosyvoice2_dir / "data"

# EmoBox数据集文件夹
emobox_data_dir = data_dir.parent.parent.parent.parent.parent / "EmoBox/data"
raw_dataset_dir = emobox_data_dir / dataset

# 输出目录
dataset_dir = data_dir / dataset

def make_m3ed(mode):
    test_json_path = raw_dataset_dir / "fold_1" / f"{dataset}_{mode}_fold_1.json"
    test_dict = {}
    with open(test_json_path, "r", encoding="utf-8") as f:
        test_dict = json.load(f)

    transcript_path = raw_dataset_dir / "annotation.json"
    trans_dict = {}
    with open(transcript_path, "r", encoding="utf-8") as f_1:
        trans_dict = json.load(f_1)

    target_mode_dir = dataset_dir / mode
    os.makedirs(target_mode_dir, exist_ok=True)
    for k, v in test_dict.items():
        # 复制音频到目标目录下
        sub_path = v['wav'].replace(f"{dataset}/", "")
        audio_path = raw_dataset_dir / sub_path

        # 如A_rucikeaidewomen_1_1.wav
        audio_name = v['wav'].split('/')[-1]
        shutil.copy(audio_path, target_mode_dir / audio_name)

        # 写入音频转录到目标目录下
        # 如A_rucikeaidewomen_1_1
        name_wo_wav = audio_name.split('.')[0]

        # 得到tv名，如rucikeaidewomen
        tv_name = name_wo_wav.split('_')[1]

        # 得到分段名，如rucikeaidewomen_1
        segment_name = name_wo_wav[2:].rsplit('_', 1)[0]

        # 对话名，如rucikeaidewomen_1_1
        dialog_name = name_wo_wav[2:]
        trans = trans_dict[tv_name][segment_name]["Dialog"][dialog_name]["Text"].strip()
        emotion = v['emo'].strip()
        with open(target_mode_dir / f"{name_wo_wav}.normalized.txt", "w", encoding="utf-8") as f_2:
            f_2.write(f"{emotion}<|endofprompt|>{trans}")
    print('done!')


# 合并训练集验证集测试集生成整个数据集的目录
def gen_whole():
    whole_dir = dataset_dir / "whole"
    train_dir = dataset_dir / "train"
    valid_dir = dataset_dir / "valid"
    os.makedirs(whole_dir, exist_ok=True)
    for entry in os.listdir(train_dir):
        if ".txt" in entry or ".wav" in entry:
            shutil.copy(train_dir / entry, whole_dir / entry)
    for entry in os.listdir(valid_dir):
        if ".txt" in entry or ".wav" in entry:
            shutil.copy(valid_dir / entry, whole_dir / entry)
    print("合并完成!")

def main():
    # for mode in ["train", "valid"]:
    #     make_m3ed(mode)
    gen_whole()
if __name__ == "__main__":
    main()
    
