# 读取两组test音频，比较acc
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
    使用测试集做评估
"""
emo_acc = {}

# debug的文件夹名和提示词相关，音频文件名和提示词无关，情感名从文件名中提取
debug_label = "surprise_zeroshot"
way = "instruct"

# acc txt的后缀
# acc_label = "afterDPO_1"
# acc_label = f"debug_{debug_label}"

cosyvoice2_dir = Path(__file__).parent.parent.parent
wav2text_path = cosyvoice2_dir / "wav2text_m3ed.json"
# wav2text_path = f"/home/CosyVoice/examples/libritts/cosyvoice2/tts_text_{debug_label}.json"

# 模型的输出标签
# model_output_emo_list = ["angry", "fear", "happy", "neutral", "sad", "surprise"]
model_output_emo_list = ['anger', 'disgusted', 'fear', 'happy', 'neutral', 'other', 'sad', 'surprise', '<unk>']

# test音频根目录
test_root_dir = cosyvoice2_dir / "exp/cosyvoice"

ori_corpus_dir = cosyvoice2_dir / "cosyvoice2/data/casia"

script_path = Path(__file__)

# 评测结果文件夹
result_dir_path = script_path.parent / "result"

# SER ACC评估
def serACC(model, subdir):
    test_dir = test_root_dir / subdir / way

    emo_acc_total = {}
    emo_cnt = {}
    trans_dict = {}
    with open(wav2text_path, "r", encoding="utf-8") as f:
        trans_dict = json.load(f)
    for wav in os.listdir(test_dir):
        if ".wav" in wav:
            samp_wav_path = os.path.join(test_dir, wav)
            ref_name = wav.split('.')[0].strip()[:-2]
            instruct_emotion_name = trans_dict[ref_name][0].split('<|endofprompt|>')[0].lower()

            res = model.generate(samp_wav_path, granularity="utterance", extract_embedding=False)

            scores = res[0]['scores']
            max_score = max(scores)
            max_i = scores.index(max_score)

            if model_output_emo_list[max_i] == instruct_emotion_name:
                acc = 1 
            else:
                acc = 0

            if instruct_emotion_name not in emo_cnt:
                emo_cnt[instruct_emotion_name] = 0
            else:
                emo_cnt[instruct_emotion_name] += 1

            if instruct_emotion_name not in emo_acc_total:
                emo_acc_total[instruct_emotion_name] = 0
            else:
                emo_acc_total[instruct_emotion_name] += acc


    os.makedirs(result_dir_path, exist_ok=True)
    avg_acc_result = {}
    for k, v in emo_acc_total.items():
        avg_acc = v / emo_cnt[k]
        avg_acc_result[k] = avg_acc
        print(f"***根据该情感{emo_cnt[k]}个样本的统计，{k} 情感的平均acc为{avg_acc}")
    with open(result_dir_path / f"eva_accuracy_{subdir}.txt", "w", encoding="utf-8") as f:
        for k, v in avg_acc_result.items():
            f.write(f"{k} acc: {v}\n")
    # assert False



if __name__ == '__main__':
    # config = utils.parse_opt()
    # model = models.load(config)
    model_name = "emotion2vec_base_finetuned"
    model = AutoModel(model=f"iic/{model_name}")
    for subdir in ["test_init"]:
        serACC(model, subdir)
