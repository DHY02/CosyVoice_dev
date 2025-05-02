# 读取两组test音频，比较acc
from collections import OrderedDict
from pathlib import Path
import os
# import utils
# import shutil
# import models
# from predict import predict
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import json
from funasr import AutoModel
"""
    使用测试集计算accuracy做评估
"""
emo_acc = {}

# debug的文件夹名和提示词相关，音频文件名和提示词无关，情感名从文件名中提取
debug_label = "surprise_zeroshot"


# acc txt的后缀
acc_label = "after_sft_woDPO_1"
# acc_label = f"debug_{debug_label}"

# wav2text_path = Path("/home/CosyVoice/examples/libritts/cosyvoice2/wav2tts_text_dpo_ori.json")
# wav2text_path = f"/home/CosyVoice/examples/libritts/cosyvoice2/tts_text_{debug_label}.json"

# 模型的输出标签
# model_output_emo_list = ["angry", "fear", "happy", "neutral", "sad", "surprise"]
model_output_emo_list = ['angry', 'disgusted', 'fear', 'happy', 'neutral', 'other', 'sad', 'surprise', '<unk>']

# test音频根目录
test_root_dir = Path("/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/exp/cosyvoice")

# 要测试的test音频子目录列表
test_sub_dir = ["test_dpo_1200_DPO"]

# 测试集的推理方法
way = "sft"

# 原始数据集目录
ori_corpus_dir = Path("/home/CosyVoice/examples/libritts/cosyvoice2/data/casia")

script_path = Path(__file__)

# cosyvoice相关文件夹
cosyvoice_dir_path = script_path.parent / "cosyvoice"

# 对同一audio_name的5个采样音频进行推理，将最小acc的采样和原数据集音频分别作为拒绝样本和接受样本记录下来
def solve(model, subdir):
    test_dir = test_root_dir / subdir / way

    emo_acc_total = {}
    emo_cnt = {}

    for wav in os.listdir(test_dir):
        if ".wav" in wav:
            samp_wav_path = os.path.join(test_dir, wav)
            audio_name = wav.split('.')[0].strip()[:-2]
            emotion_name = audio_name.split('_')[-1].strip()
            index = model_output_emo_list.index(emotion_name)
            # result, result_prob = predict(config, audio_path=samp_wav_path, model=model)
            res = model.generate(samp_wav_path, granularity="utterance", extract_embedding=False)
            # acc_total = 1 if config.class_labels[int(result)] == emotion_name else 0
            acc = res[0]['scores'][index]
            # acc = result_prob[index]


            # acc = 1
            if emotion_name not in emo_cnt:
                emo_cnt[emotion_name] = 0
            else:
                emo_cnt[emotion_name] += 1

            if emotion_name not in emo_acc_total:
                emo_acc_total[emotion_name] = 0
            else:
                emo_acc_total[emotion_name] += acc

    # 按 键排序
    emo_acc_total = OrderedDict(sorted(emo_acc_total.items()))
    os.makedirs(cosyvoice_dir_path, exist_ok=True)
    for k, v in emo_acc_total.items():
        avg_acc = v / emo_cnt[k]
        print(f"***根据{emo_cnt[k]}个样本的统计，{k} 情感的平均acc为{avg_acc}")
        with open(cosyvoice_dir_path / f"test_accuracy_{subdir}_{way}_emo2vec.txt", "a", encoding="utf-8") as f:
            f.write(f"{k} acc: {avg_acc}\n")
    # assert False



if __name__ == '__main__':
    # config = utils.parse_opt()
    # model = models.load(config)
    model_name = "emotion2vec_base_finetuned"
    model = AutoModel(model=f"iic/{model_name}")
    for subdir in test_sub_dir:
        solve(model, subdir)
