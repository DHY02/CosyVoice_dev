from pathlib import Path
import os
import random

import shutil

from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import json
from funasr import AutoModel
"""
    对同一audio_name的5个采样音频进行情感预测，并制作dpo数据集
"""
emo_acc = {}

# debug的文件夹名和提示词相关，音频文件名和提示词无关，情感名从文件名中提取
debug_label = "surprise_zeroshot"
way = "zero_shot"

# acc txt的后缀
acc_label = "init"
# acc_label = f"debug_{debug_label}"

wav2text_path = Path("/home/CosyVoice/examples/libritts/cosyvoice2/wav2text_samp_casia_ori.json")
# wav2text_path = f"/home/CosyVoice/examples/libritts/cosyvoice2/tts_text_{debug_label}.json"

# 模型的输出标签
# model_output_emo_list = ["angry", "fear", "happy", "neutral", "sad", "surprise"]
model_output_emo_list = ['angry', 'disgusted', 'fear', 'happy', 'neutral', 'other', 'sad', 'surprise', '<unk>']

# 随机采样的音频位置
samples_dir = Path("/home/CosyVoice/examples/libritts/cosyvoice2/exp/cosyvoice/sampling_casia_ori")

ori_corpus_dir = Path("/home/CosyVoice/examples/libritts/cosyvoice2/data/casia")

pwd = Path(__file__)

# cosyvoice相关文件夹
cosyvoice_dir_path = pwd.parent.parent.parent


# 验证情感分布
def print_distribution(names, title):
    unique, counts = np.unique([name[:-2].split('_')[-1] for name in names], return_counts=True)
    print(f"{title}情感分布: {dict(zip(unique, counts))}")

# 对同一audio_name的5个采样音频进行推理，将最小acc的和最高acc的采样分别作为拒绝样本和接受样本记录下来
def make_dpo_corpus(config, model):
    # samples_dir = f"/home/CosyVoice/examples/libritts/cosyvoice2/exp/cosyvoice/debug_{debug_label}/{way}"
    worst_acc = {}
    worst_acc_path = {}
    best_acc = {}
    best_acc_path = {}

    emo_acc_total = {}
    emo_cnt = {}

    for entry in os.listdir(samples_dir):
        samp_dir_path = os.path.join(samples_dir, entry)
        if os.path.isdir(samp_dir_path) and "samp" in entry:
            for wav in os.listdir(samp_dir_path):
                if ".wav" in wav:
                    samp_wav_path = os.path.join(samp_dir_path, wav)
                    audio_name = wav.split('.')[0].strip()[:-2]
                    emotion_name = audio_name.split('_')[-1].strip()
                    index = model_output_emo_list.index(emotion_name)
                    _, result_prob = predict(config, audio_path=samp_wav_path, model=model)

                    acc = result_prob[index]

                    if emotion_name not in emo_cnt:
                        emo_cnt[emotion_name] = 0
                    else:
                        emo_cnt[emotion_name] += 1
                    if emotion_name not in emo_acc_total:
                        emo_acc_total[emotion_name] = 0
                    else:
                        emo_acc_total[emotion_name] += acc
                    
                    if audio_name not in worst_acc or worst_acc[audio_name] > acc:
                        worst_acc[audio_name] = acc
                        worst_acc_path[audio_name] = samp_wav_path
                    if audio_name not in best_acc or best_acc[audio_name] < acc:
                        best_acc[audio_name] = acc
                        best_acc_path[audio_name] = samp_wav_path
    
    
    os.makedirs(cosyvoice_dir_path, exist_ok=True)
    for k, v in emo_acc_total.items():
        avg_acc = v / emo_cnt[k]
        print(f"***根据{emo_cnt[k]}个样本的统计，{k} 情感的平均acc为{avg_acc}")
        # with open(cosyvoice_dir_path / f"accuracy_{acc_label}.txt", "a", encoding="utf-8") as f:
        #     f.write(f"{k} acc: {avg_acc}\n")
    # assert False
    corpus_dir = cosyvoice_dir_path / "data" / "casia_dpo"
    os.makedirs(corpus_dir, exist_ok=True)

    # 将数据集分为训练集、验证集、测试集，根据情感label进行分层采样
    # 提取wav_id和label进行分层采样
    audio_names = np.array(list(worst_acc_path.keys()))
    emotion_labels = np.array([name[:-2].split('_')[-1] for name in audio_names])
    split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, temp_idx in split1.split(audio_names, emotion_labels):
        train_names = audio_names[train_idx]  # 80%训练集
        temp_names = audio_names[temp_idx]    # 20%临时集
        temp_emotions = emotion_labels[temp_idx]
    
    split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    for val_idx, test_idx in split2.split(temp_names, temp_emotions):
        val_names = temp_names[val_idx]   # 10%验证集
        test_names = temp_names[test_idx] # 10%测试集


    print_distribution(train_names, "训练集")
    print_distribution(val_names, "验证集")
    print_distribution(test_names, "测试集")
    
    # 重组为字典
    train_l_data = {name: worst_acc_path[name] for name in train_names}
    val_l_data = {name: worst_acc_path[name] for name in val_names}
    test_l_data = {name: worst_acc_path[name] for name in test_names}
    
    train_w_data = {name: best_acc_path[name] for name in train_names}
    val_w_data = {name: best_acc_path[name] for name in val_names}
    test_w_data = {name: best_acc_path[name] for name in test_names}


    # 制作训练集
    train_l_dir = corpus_dir / "train" / "reject"
    train_w_dir = corpus_dir / "train" / "receive"

    os.makedirs(train_l_dir, exist_ok=True)
    os.makedirs(train_w_dir, exist_ok=True)

    # w
    for k, v in train_w_data.items():
        wav_name = k + ".wav"
        text_name = k + ".normalized.txt"
        shutil.copy(v, train_w_dir / wav_name)
        with open(train_w_dir / text_name, "w", encoding="utf-8") as f:
            f.write(tts_texts[k][0])

    # l
    for k, v in train_l_data.items():
        wav_name = k + ".wav"
        text_name = k + ".normalized.txt"
        shutil.copy(v, train_l_dir / wav_name)
        with open(train_l_dir / text_name, "w", encoding="utf-8") as f:
            f.write(tts_texts[k][0])

    # 验证集
    valid_l_dir = corpus_dir / "valid" / "reject"
    valid_w_dir = corpus_dir / "valid" / "receive"
    os.makedirs(valid_l_dir, exist_ok=True)
    os.makedirs(valid_w_dir, exist_ok=True)
    
    # w
    for k, v in val_w_data.items():
        wav_name = k + ".wav"
        text_name = k + ".normalized.txt"
        shutil.copy(v, valid_w_dir / wav_name)
        with open(valid_w_dir / text_name, "w", encoding="utf-8") as f:
            f.write(tts_texts[k][0])

    # l
    for k, v in val_l_data.items():
        wav_name = k + ".wav"
        text_name = k + ".normalized.txt"
        shutil.copy(v, valid_l_dir / wav_name)
        with open(valid_l_dir / text_name, "w", encoding="utf-8") as f:
            f.write(tts_texts[k][0])
    
    # 测试集

    test_l_dir = corpus_dir / "test" / "reject"
    test_w_dir = corpus_dir / "test" / "receive"
    os.makedirs(test_l_dir, exist_ok=True)
    os.makedirs(test_w_dir, exist_ok=True)
    
    # w
    for k, v in test_w_data.items():
        wav_name = k + ".wav"
        text_name = k + ".normalized.txt"
        shutil.copy(v, test_w_dir / wav_name)
        with open(test_w_dir / text_name, "w", encoding="utf-8") as f:
            f.write(tts_texts[k][0])

    # l
    for k, v in test_l_data.items():
        wav_name = k + ".wav"
        text_name = k + ".normalized.txt"
        shutil.copy(v, test_l_dir / wav_name)
        with open(test_l_dir / text_name, "w", encoding="utf-8") as f:
            f.write(tts_texts[k][0])


    print("done!")


# 对同一audio_name的5个采样音频进行推理，将最小acc的采样和原数据集音频分别作为拒绝样本和接受样本记录下来
def make_dpo_corpus_2(model):
    # samples_dir = f"/home/CosyVoice/examples/libritts/cosyvoice2/exp/cosyvoice/debug_{debug_label}/{way}"
    worst_acc = {}
    worst_acc_path = {}
    best_acc = {}
    best_acc_path = {}

    emo_acc_total = {}
    emo_cnt = {}

    for entry in os.listdir(samples_dir):
        samp_dir_path = os.path.join(samples_dir, entry)
        if os.path.isdir(samp_dir_path) and "samp" in entry:
            for wav in os.listdir(samp_dir_path):
                if ".wav" in wav:
                    samp_wav_path = os.path.join(samp_dir_path, wav)
                    audio_name = wav.split('.')[0].strip()
                    emotion_name = audio_name.split('_')[-1].strip()
                    index = model_output_emo_list.index(emotion_name)

                    res = model.generate(samp_wav_path, granularity="utterance", extract_embedding=False)

                    acc = res[0]['scores'][index]
                    # acc = 1
                    if emotion_name not in emo_cnt:
                        emo_cnt[emotion_name] = 0
                    else:
                        emo_cnt[emotion_name] += 1

                    if emotion_name not in emo_acc_total:
                        emo_acc_total[emotion_name] = 0
                    else:
                        emo_acc_total[emotion_name] += acc
                    
                    if audio_name not in worst_acc or worst_acc[audio_name] > acc:
                        worst_acc[audio_name] = acc
                        worst_acc_path[audio_name] = samp_wav_path
                    # if audio_name not in best_acc or best_acc[audio_name] < acc:
                    #     best_acc[audio_name] = acc
                    #     best_acc_path[audio_name] = samp_wav_path
    
    os.makedirs(cosyvoice_dir_path, exist_ok=True)
    for k, v in emo_acc_total.items():
        avg_acc = v / emo_cnt[k]
        print(f"***根据{emo_cnt[k]}个样本的统计，{k} 情感的平均acc为{avg_acc}")
        # with open(cosyvoice_dir_path / f"accuracy_{acc_label}.txt", "a", encoding="utf-8") as f:
        #     f.write(f"{k} acc: {avg_acc}\n")
    assert False
    corpus_dir = cosyvoice_dir_path / "data" / "casia_dpo"
    os.makedirs(corpus_dir, exist_ok=True)

    # 将数据集分为训练集、验证集、测试集，根据情感label进行分层采样
    # 提取wav_id和label进行分层采样
    audio_names = np.array(list(worst_acc_path.keys()))
    emotion_labels = np.array([name[:-2].split('_')[-1] for name in audio_names])
    split1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, temp_idx in split1.split(audio_names, emotion_labels):
        train_names = audio_names[train_idx]  # 80%训练集
        temp_names = audio_names[temp_idx]    # 20%临时集
        temp_emotions = emotion_labels[temp_idx]
    
    split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    for val_idx, test_idx in split2.split(temp_names, temp_emotions):
        val_names = temp_names[val_idx]   # 10%验证集
        test_names = temp_names[test_idx] # 10%测试集


    print_distribution(train_names, "训练集")
    print_distribution(val_names, "验证集")
    print_distribution(test_names, "测试集")
    
    # 重组为字典
    train_l_data = {name: worst_acc_path[name] for name in train_names}
    val_l_data = {name: worst_acc_path[name] for name in val_names}
    test_l_data = {name: worst_acc_path[name] for name in test_names}
    
    # train_w_data = {name: best_acc_path[name] for name in train_names}
    # val_w_data = {name: best_acc_path[name] for name in val_names}
    # test_w_data = {name: best_acc_path[name] for name in test_names}


    # 制作训练集
    train_l_dir = corpus_dir / "train" / "reject"
    train_w_dir = corpus_dir / "train" / "receive"

    os.makedirs(train_l_dir, exist_ok=True)
    os.makedirs(train_w_dir, exist_ok=True)

    # l, w
    for k, v in train_l_data.items():
        wav_name = k + ".wav"
        text_name = k + ".normalized.txt"
        shutil.copy(v, train_l_dir / wav_name)
        with open(train_l_dir / text_name, "w", encoding="utf-8") as f:
            f.write(tts_texts[k][0])
        w_path = ori_corpus_dir / wav_name
        shutil.copy(w_path, train_w_dir / wav_name)
        with open(train_w_dir / text_name, "w", encoding="utf-8") as f:
            f.write(tts_texts[k][0])

    # 验证集
    valid_l_dir = corpus_dir / "valid" / "reject"
    valid_w_dir = corpus_dir / "valid" / "receive"
    os.makedirs(valid_l_dir, exist_ok=True)
    os.makedirs(valid_w_dir, exist_ok=True)
    
    # l
    for k, v in val_l_data.items():
        wav_name = k + ".wav"
        text_name = k + ".normalized.txt"
        shutil.copy(v, valid_l_dir / wav_name)
        with open(valid_l_dir / text_name, "w", encoding="utf-8") as f:
            f.write(tts_texts[k][0])
        w_path = ori_corpus_dir / wav_name
        shutil.copy(w_path, valid_w_dir / wav_name)
        with open(valid_w_dir / text_name, "w", encoding="utf-8") as f:
            f.write(tts_texts[k][0])
    
    # 测试集

    test_l_dir = corpus_dir / "test" / "reject"
    test_w_dir = corpus_dir / "test" / "receive"
    os.makedirs(test_l_dir, exist_ok=True)
    os.makedirs(test_w_dir, exist_ok=True)
    
    # l
    for k, v in test_l_data.items():
        wav_name = k + ".wav"
        text_name = k + ".normalized.txt"
        shutil.copy(v, test_l_dir / wav_name)
        with open(test_l_dir / text_name, "w", encoding="utf-8") as f:
            f.write(tts_texts[k][0])
            w_path = ori_corpus_dir / wav_name
        shutil.copy(w_path, test_w_dir / wav_name)
        with open(test_w_dir / text_name, "w", encoding="utf-8") as f:
            f.write(tts_texts[k][0])

    print("done!")


# 获得emo-dpo的emo_dpo_reject音频，向附近选的14个相同text音频中随机选一个不是该情感的音频作为emo_dpo_reject
def make_emo_dpo_reject_corpus():
    random.seed(42)
    corpus_dir = Path("/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/data/casia-emo-dpo")
    # 获得receive样本列表并排序
    for subdir in ["train", "valid"]:
        src_dir = corpus_dir / subdir / "receive"
        receives = [entry for entry in os.listdir(src_dir) if entry.endswith('.wav')]
        receives.sort()
        # receive id: reject id
        reject_dict = {}
        for i, wav in enumerate(receives):
            emotion = wav.split('.')[0].split('_')[-1].strip()
            audio_id = "_".join(wav.split('.')[0].split('_')[:-1])
            delta_s = -14
            delta_e = 14
            sample_list = []
            for j in range(max(i + delta_s, 0), min(i + delta_e, len(receives) - 1)):
                # print(receives[j])
                emo_j = receives[j].split('.')[0].split('_')[-1].strip()
                audio_id_j = "_".join(receives[j].split('.')[0].split('_')[:-1])
                if emo_j != emotion and audio_id_j == audio_id:
                    sample_list.append(receives[j])
            if sample_list:  # 确保列表不为空
                selected_reject = random.choice(sample_list)
                reject_dict[wav] = selected_reject
            else:
                # print(wav)
                raise Exception
        target_dir = corpus_dir / subdir / "emo_dpo_reject"
        for k, v in reject_dict.items():
            os.makedirs(target_dir, exist_ok=True)
            text_name = "_".join(k.split('.')[:-1]) + ".normalized.txt"
            # wav
            shutil.copy(src_dir / v, target_dir / k)
            # text
            shutil.copy(src_dir / text_name, target_dir / text_name)
    print("Make_emo_dpo_corpus done!")


# 转移receive到casia-emo-dpo
def move_from_dpo2emo_dpo():
    src_dir = Path("/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/data/casia_dpo")
    tgt_dir = Path("/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/data/casia-emo-dpo")
    
    # 存id: path
    receive_id2path = {}
    reject_id2path = {}
    for i in ["train", "valid", "test"]:
        for j in ["receive", "reject"]:
            os.makedirs(tgt_dir / i / j, exist_ok=True)
            for entry in os.listdir(src_dir / i / j):
                if entry.endswith(".wav"):
                    audio_id = "_".join(entry.split('.')[:-1])
                    if j == "receive":
                        receive_id2path[audio_id] = (src_dir / i / j / entry, src_dir / i / j / (audio_id + ".normalized.txt"))
                    elif j == "reject":
                        reject_id2path[audio_id] = (src_dir / i / j / entry, src_dir / i / j / (audio_id + ".normalized.txt"))
    # 对于receive中的id，从id对应的reject 的 path复制到目标reject目录下
    for i in ["train", "valid", "test"]:
        for entry in os.listdir(tgt_dir / i / "receive"):
            if entry.endswith(".wav"):
                audio_id = "_".join(entry.split('.')[:-1])
                rej_wav = reject_id2path[audio_id][0]
                rej_txt = reject_id2path[audio_id][1]
                shutil.copy(rej_wav, tgt_dir / i / "reject" / (audio_id + ".wav"))
                shutil.copy(rej_txt, tgt_dir / i / "reject" / (audio_id + ".normalized.txt"))
    print("move_from_dpo2emo_dpo is done!")


if __name__ == '__main__':
    # config = utils.parse_opt()
    # model = models.load(config)
    # model_name = "emotion2vec_base_finetuned"
    # model = AutoModel(model=f"iic/{model_name}")
    # tts_texts = {}
    # with open(wav2text_path, "r", encoding="utf-8") as f:
    #     tts_texts = json.load(f)
    # make_dpo_corpus_2(model)
    move_from_dpo2emo_dpo()
    # make_emo_dpo_reject_corpus()