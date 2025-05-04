# 从casia事先随机提取出的和测试集数量相同的casia音频作为推理的ref音频
import json
import os
from pathlib import Path
import random

random.seed(1)
pwd = Path(__file__)

dataset="m3ed"

cosyvoice2_dir = pwd.parent.parent.parent
# cosyvoice data文件夹
data_dir = cosyvoice2_dir / "data"

dataset_dir = data_dir / dataset


# 生成测试集的wav2text以及对应的promtwav2refwav，
# wav2text中的wav是其他数据集的prompt音频，text是测试集的text
# promtwav2refwav中refwav是text对应的ground truth
def make_infer_text(emo_cnt, mode):
    # 测试集的wav2text
    target_json_path_1 = cosyvoice2_dir / f"wav2text_{dataset}_test.json"

    # promtwav2refwav：记录prompt音频和ref音频路径的映射
    target_json_path_2 = cosyvoice2_dir / f"promtwav2refwav_{dataset}_test.json"

    print(f"数据集中最少情感数为{emo_cnt}，以此设为测试集中每个情感的数量")
    trans_list = []
    num_emo = {}
    # text数据集
    corpus_dir = dataset_dir / mode

    # prompt音频数据集
    prompt_dir = data_dir / "casia"

    # 划分测试集后的text总数
    cnt = 0
    # 读取测试集音频数量n
    for entry in os.listdir(corpus_dir):
        # 读取测试集每个情感emo_cnt个转录文本
        if ".txt" in entry:
            with open(corpus_dir / entry, "r", encoding="utf-8") as f:
                trans = f.readline()
            emotion = trans.split('<|endofprompt|>')[0]
            ref_wav_path = entry.split('.')[0] + ".wav"
            trans = (trans, ref_wav_path)
            # 排除Disgust
            if emotion != "Disgust":
                if emotion not in num_emo:
                    num_emo[emotion] = 1
                    trans_list.append(trans)
                    cnt += 1
                elif num_emo[emotion] < emo_cnt:
                    num_emo[emotion] += 1
                    trans_list.append(trans)
                    cnt += 1
    print("trans_list len:")
    print(len(trans_list))
    # 随机提取出的和测试集数量相同的casia音频
    all_wav_files = [f.split('.')[0] for f in os.listdir(prompt_dir) if f.endswith(".wav") and os.path.isfile(os.path.join(prompt_dir, f))]
    random.shuffle(all_wav_files)
    selected_prom = all_wav_files[:cnt]

    # 将抽取的名称和转录文本一一映射写入json
    wav2text_dict = {}
    promtwav2refwav_dict = {}
    for i, (trans, ref_path) in enumerate(trans_list):
        prompt_name = selected_prom[i]
        wav2text_dict[prompt_name] = [trans]
        promtwav2refwav_dict[prompt_name] = ref_path

    with open(target_json_path_1, "w", encoding="utf-8") as f:
        json.dump(wav2text_dict, f, ensure_ascii=False, indent=2)
    with open(target_json_path_2, "w", encoding="utf-8") as f:
        json.dump(promtwav2refwav_dict, f, ensure_ascii=False, indent=2)


# 统计数据集中情感分类及其数量，返回最小数量
def emo_status(mode):
    emo_cnt = {}
    corpus_dir = dataset_dir / mode
    for entry in os.listdir(corpus_dir):
        if entry.endswith("txt"):
            with open(corpus_dir / entry, "r", encoding="utf-8") as f:
                trans = f.readline()
            emotion = trans.split('<|endofprompt|>')[0]
            if emotion not in emo_cnt:
                emo_cnt[emotion] = 1
            else:
                emo_cnt[emotion] += 1
    print(f"{mode}中情感分布：{emo_cnt}")
    min_key = min(emo_cnt, key=lambda k: emo_cnt[k])
    min_value = emo_cnt[min_key]
    return min_value


# 使用自身的音频作为参考音频生成采样的wav2text
def make_train_valid_wav2text():
    output_dict = {}
    # 读取m3ed train和valid 文件夹下的标注文本，生成字典，写入json
    for sub in ["train", "valid"]:
        corpus_dir = f"/home/CosyVoice/examples/libritts/cosyvoice2/data/m3ed/{sub}"
        for entry in os.listdir(corpus_dir):
            if ".txt" in entry:
                audio_name = entry.split('.')[0]
                with open(f"{corpus_dir}/{entry}", "r", encoding="utf-8") as f:
                    trans = f.readline().strip()
                output_dict[audio_name] = [trans]
    
    target_json_path = cosyvoice2_dir / f"wav2text_samp.json"
    with open(target_json_path, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=2)

    print(f"JSON 文件已生成：{target_json_path}")


if __name__ == "__main__":
    emo_cnt = emo_status("test")
    make_infer_text(emo_cnt, "test")
    # make_train_valid_wav2text()