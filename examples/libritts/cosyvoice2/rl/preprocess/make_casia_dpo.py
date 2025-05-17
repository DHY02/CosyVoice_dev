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


def make_esd_dpo_corpus():
    """
        对于esd文件夹下的train和valid, 对于同一个wav_id, 从若干个采样中选择wav_id.adv最小的作为reject,
        创建reject文件夹将对应的wav和txt放入该文件夹
    """
    # 数据源路径
    esd_root_dir = Path("/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/data/esd")
    
    # 遍历train和valid目录
    for split in ["train", "valid"]:
        print(f"处理 {split} 数据集...")
        split_dir = esd_root_dir / split
        
        # 创建reject目录
        reject_dir = split_dir / "reject"
        os.makedirs(reject_dir, exist_ok=True)
        
        # 获取所有samp_*目录
        samp_dirs = [d for d in os.listdir(split_dir) if d.startswith('samp_')]
        
        # 存储每个音频ID的最小adv值及其路径
        min_adv_values = {}
        min_adv_paths = {}
        
        # 遍历所有采样目录，找出每个音频ID的最小adv值
        for samp_dir in samp_dirs:
            samp_path = split_dir / samp_dir
            
            for file in os.listdir(samp_path):
                if file.endswith('.adv'):
                    audio_id = file[:-4]  # 去掉.adv后缀
                    adv_file_path = samp_path / file
                    
                    # 读取adv值
                    try:
                        with open(adv_file_path, 'r') as f:
                            adv_value = float(f.readline().strip())
                    except Exception as e:
                        print(f"读取{adv_file_path}失败: {e}")
                        continue
                    
                    # 更新最小adv值
                    if audio_id not in min_adv_values or adv_value < min_adv_values[audio_id]:
                        min_adv_values[audio_id] = adv_value
                        min_adv_paths[audio_id] = samp_path / audio_id
        
        # 将最小adv值对应的音频和文本复制到reject目录
        count = 0
        for audio_id, min_path in min_adv_paths.items():
            wav_src = f"{min_path}.wav"
            txt_src = f"{min_path}.normalized.txt"
            
            if os.path.exists(wav_src) and os.path.exists(txt_src):
                wav_dest = reject_dir / f"{audio_id}.wav"
                txt_dest = reject_dir / f"{audio_id}.normalized.txt"
                
                try:
                    shutil.copy(wav_src, wav_dest)
                    shutil.copy(txt_src, txt_dest)
                    count += 1
                except Exception as e:
                    print(f"复制文件失败: {e}")
            else:
                print(f"警告: 文件不存在 {wav_src} 或 {txt_src}")
        
        print(f"{split} 数据集处理完成，共复制 {count} 个reject样本")
    
    print("ESD DPO语料库创建完成!")


def make_esd_emo_dpo_reject_corpus():
    """
        对于esd文件夹下的train和valid, 对于同一个wav_id，寻找同一个文本的不同情感版本作为emo_dpo_reject
        规则：对于同一说话人的每一个audio_id，其audio_id +/- 350就是同一个文本的不同情感版本
        注：同一个音频ID只对应一个情感且是唯一的，说话人ID范围是0001-0010，audio_id范围是000001-001750
        创建emo_dpo_reject文件夹将对应的wav和txt放入该文件夹
    """
    random.seed(42)
    # 数据源路径
    esd_root_dir = Path("/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/data/esd")
    
    # 定义audio_id范围和增量
    audio_id_max = 1750  # 范围是000001-001750
    audio_id_offset = 350  # 同一文本不同情感的偏移量
    
    # 遍历train和valid目录
    for split in ["train", "valid"]:
        print(f"处理 {split} 数据集...")
        split_dir = esd_root_dir / split
        
        # 创建emo_dpo_reject目录
        emo_dpo_reject_dir = split_dir / "emo_dpo_reject"
        os.makedirs(emo_dpo_reject_dir, exist_ok=True)
        
        # 获取receive目录中的所有wav文件
        receive_dir = split_dir / "receive"
        if not os.path.exists(receive_dir):
            print(f"警告: {receive_dir} 目录不存在，跳过处理")
            continue
            
        receives = [entry for entry in os.listdir(receive_dir) if entry.endswith('.wav')]
        if not receives:
            print(f"警告: {receive_dir} 目录中没有wav文件，跳过处理")
            continue
        
        # 建立映射：话者ID和音频ID到文件名的映射
        speaker_audio_to_file = {}
        for wav_file in receives:
            parts = wav_file.split('_')
            if len(parts) < 3:
                continue
                
            speaker_id = parts[0]
            audio_id_str = parts[1]
            
            try:
                audio_id = int(audio_id_str)
                key = (speaker_id, audio_id)
                speaker_audio_to_file[key] = wav_file
            except ValueError:
                continue
        
        # 存储接收-拒绝音频对应关系
        reject_dict = {}
        skipped_count = 0
        
        # 遍历所有receive音频文件
        for wav in receives:
            parts = wav.split('_')
            if len(parts) < 3:
                print(f"警告: 无法解析文件名 {wav}，跳过")
                skipped_count += 1
                continue
                
            speaker_id = parts[0]
            audio_id_str = parts[1]
            current_emotion = parts[2].split('.')[0]
            
            try:
                audio_id = int(audio_id_str)
            except ValueError:
                print(f"警告: 无法将 {audio_id_str} 转换为整数，跳过 {wav}")
                skipped_count += 1
                continue
            
            # 尝试可能的偏移量
            possible_offsets = [-350*2, -350, 350, 350*2, 350*3]
            random.shuffle(possible_offsets)  # 随机化选择顺序
            
            found_match = False
            for offset in possible_offsets:
                target_audio_id = audio_id + offset
                
                # 确保在有效范围内
                if 1 <= target_audio_id <= audio_id_max:
                    key = (speaker_id, target_audio_id)
                    if key in speaker_audio_to_file:
                        target_wav = speaker_audio_to_file[key]
                        target_emotion = target_wav.split('_')[2].split('.')[0]
                        
                        # 确保情感不同
                        if target_emotion != current_emotion:
                            reject_dict[wav] = target_wav
                            found_match = True
                            break
            
            if not found_match:
                print(f"警告: {wav} 没有找到合适的emo_dpo_reject音频，跳过")
                skipped_count += 1
        
        print(f"跳过的文件总数: {skipped_count}")
        
        # 复制选中的拒绝样本到emo_dpo_reject目录
        count = 0
        for k, v in reject_dict.items():
            # 文本文件名（与音频同名但扩展名不同）
            text_name = k.replace('.wav', '.normalized.txt')
            
            # 源文件路径
            src_wav = receive_dir / v
            src_txt = receive_dir / text_name
            
            # 目标文件路径（注意：保持与原receive音频相同的文件名）
            dst_wav = emo_dpo_reject_dir / k
            dst_txt = emo_dpo_reject_dir / text_name
            
            # 确保源文件存在
            if os.path.exists(src_wav) and os.path.exists(src_txt):
                try:
                    # 复制文件
                    shutil.copy(src_wav, dst_wav)
                    shutil.copy(src_txt, dst_txt)
                    count += 1
                except Exception as e:
                    print(f"复制文件失败: {e}")
            else:
                print(f"警告: 文件不存在 {src_wav} 或 {src_txt}")
        
        print(f"{split} 数据集处理完成，共创建 {count} 个emo_dpo_reject样本")
    
    print("ESD emo-dpo-reject语料库创建完成!")


if __name__ == '__main__':
    # config = utils.parse_opt()
    # model = models.load(config)
    # model_name = "emotion2vec_base_finetuned"
    # model = AutoModel(model=f"iic/{model_name}")
    # tts_texts = {}
    # with open(wav2text_path, "r", encoding="utf-8") as f:
    #     tts_texts = json.load(f)
    # make_dpo_corpus_2(model)
    # move_from_dpo2emo_dpo()
    # make_emo_dpo_reject_corpus()
    make_esd_emo_dpo_reject_corpus()