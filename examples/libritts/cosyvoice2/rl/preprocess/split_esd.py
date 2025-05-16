import os
import shutil
import json
# 将esd数据集前10个中文说话人每个情感文件夹下的前20个句子作为验证集，接着30个作为测试集，剩下作为训练集，生成libritts-like数据集文件夹
dataset_dir_path = "/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/data/esd_raw"
tgt_dir_path = "/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/data/esd"
train_wav_scp = {}
valid_wav_scp = {}
test_wav_scp = {}

wav2text = {}
wav2emo = {}

corpus="esd"

for spk_id in sorted(os.listdir(dataset_dir_path)):
    spk_path = os.path.join(dataset_dir_path, spk_id)
    if not os.path.isdir(spk_path):
        continue

    if spk_id < "0011":
        with open(f"{spk_path}/{spk_id}.txt", "r", encoding="utf-8") as f:
            for line in f:
                w_id, text, emo = line.strip().split('\t')
                
                if emo == "中立":
                    emo = "冷静" 
                wav2emo[w_id] = emo
                wav2text[w_id] = [f"{emo}<|endofprompt|>{text}"]
        for emo_dir in os.listdir(spk_path):
            emo_path = os.path.join(spk_path, emo_dir)
            if not os.path.isdir(emo_path):
                continue
            wav_files = sorted([f for f in os.listdir(emo_path) if f.endswith(".wav")])
            for i, wav_file in enumerate(wav_files):
                wav_path = os.path.join(emo_path, wav_file)
                w_id = wav_file.split('.')[0]
                emotion = wav2emo[wav_file.split('.')[0]]
                if i < 20:  
                    valid_wav_scp[f"{w_id}_{emotion}"] = wav_path
                elif i < 50:
                    test_wav_scp[f"{w_id}_{emotion}"] = wav_path
                else:
                    train_wav_scp[f"{w_id}_{emotion}"] = wav_path


for sub_dir in ["train", "valid", "test"]:
    receive_path = os.path.join(tgt_dir_path, f"{sub_dir}/receive")
    os.makedirs(receive_path, exist_ok=True)
    if sub_dir == "train":
        wav_scp = train_wav_scp
    elif sub_dir == "valid":
        wav_scp = valid_wav_scp
    elif sub_dir == "test":
        wav_scp = test_wav_scp
    else:
        raise Exception()
    
    for k, v in wav_scp.items():
        tgt_wav_path = os.path.join(receive_path, f"{k}.wav")
        wid = "_".join(k.split('_')[:-1])
        tgt_txt_path = os.path.join(receive_path, f"{k}.normalized.txt")
        shutil.copy(v, tgt_wav_path)
        with open(tgt_txt_path, "w", encoding="utf-8") as f:
            f.write(wav2text[wid][0])

# 写入采样json
json_path = os.path.join(tgt_dir_path, f"wav2text_samp_{corpus}.json")
with open (json_path, "w", encoding="utf-8") as f:
    json.dump(wav2text, f, ensure_ascii=False, indent=4)



