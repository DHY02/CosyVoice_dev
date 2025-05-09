# 读取两组test音频，比较acc
from pathlib import Path
import os
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import json
from funasr import AutoModel
from faster_whisper import WhisperModel
from jiwer import wer
from utils import remove_punctuation, cosine_similarity, run_autoPCP, sort_dict
import pandas as pd
import argparse
"""
    使用测试集做评估
"""


pwd = Path(__file__).parent
cosyvoice2_dir = pwd.parent.parent
# 测试集名称
test_corpus = "m3ed"

# 设置wav2text，来获取ground truth的情感label
wav2text_path = cosyvoice2_dir / f"wav2text_{test_corpus}_test.json"

# GT音频目录
prompt2ref_path = cosyvoice2_dir / f"promtwav2refwav_{test_corpus}_test.json"

# 模型的输出标签
model_output_emo_list = ['anger', 'disgusted', 'fear', 'happy', 'neutral', 'other', 'sad', 'surprise', '<unk>']

# test音频根目录
test_root_dir = cosyvoice2_dir / "exp/cosyvoice"

# 测试集文本参考音频目录
ref_corpus_dir = cosyvoice2_dir / f"data/{test_corpus}/test"

# 评测结果文件夹
result_dir_path = pwd / "result"

# 评估结果命名方法：
"""
    内容：
    anger acc: 0.375
    fear acc: 0.015625
    happy acc: 0.384615
    neutral acc: 0.40625
    sad acc: 0.5538461538461539
    surprise acc: 0.323076
    EMO SIM: 0.499060
    Prosody SIM: 2.0
    WER: 0.28940568475452194
"""



# 各个指标评估
def evaluate(subdir, ser_model=None, asr_model=None):
    test_dir = test_root_dir / subdir / "instruct"
    # 实验目录命名：test_方法|参数_epoch_数字
    methodAndhyp = subdir.split('_')[1]
    num_epoch = subdir.split('_')[3]
    result_txt_name = f"{test_corpus}_{methodAndhyp}_epoch_{num_epoch}"
    emo_acc_total = {}
    emo_cnt = {}

    utt_cnt = 0

    emo_sim_total = 0

    wer_total = 0
    
    trans_dict = {}
    prompt2ref = {}

    # autoPCP input
    src_audio = []
    tgt_audio = []

    with open(wav2text_path, "r", encoding="utf-8") as f:
        trans_dict = json.load(f)
    with open(prompt2ref_path, "r", encoding="utf-8") as f:
        prompt2ref = json.load(f)
    for wav in os.listdir(test_dir):
        if ".wav" in wav:
            samp_wav_path = os.path.join(test_dir, wav)
            
            # prompt_name = wav.split('.')[0].strip()[:-2]
            prompt_name = wav.split('.')[0].strip()
            if prompt_name not in trans_dict:
                continue
            gt_wav_path = os.path.join(ref_corpus_dir, prompt2ref[prompt_name])
            instruct_emotion_name = trans_dict[prompt_name][0].split('<|endofprompt|>')[0].lower()

            # SER ACC and Emotion Similarity
            if ser_model:
                ser_res = ser_model.generate(samp_wav_path, granularity="utterance", extract_embedding=True)
                
                gt_res = ser_model.generate(gt_wav_path, granularity="utterance", extract_embedding=True)
                
                # SER ACC
                scores = ser_res[0]['scores']
                max_score = max(scores)
                max_i = scores.index(max_score)
                if model_output_emo_list[max_i] == instruct_emotion_name:
                    acc = 1 
                else:
                    acc = 0
                if instruct_emotion_name not in emo_cnt:
                    emo_cnt[instruct_emotion_name] = 1
                else:
                    emo_cnt[instruct_emotion_name] += 1
                if instruct_emotion_name not in emo_acc_total:
                    emo_acc_total[instruct_emotion_name] = 0
                else:
                    emo_acc_total[instruct_emotion_name] += acc
                
                # Emotion Similarity
                predict_feat = ser_res[0]['feats']
                gt_feat = gt_res[0]['feats']
                emo_sim_total += cosine_similarity(predict_feat, gt_feat)
                
            # WER
            if asr_model:
                segments, _ = asr_model.transcribe(samp_wav_path, beam_size=5)
                segments = list(segments)
                text = remove_punctuation("".join([seg.text for seg in segments]), "chinese")
                ref_text = remove_punctuation(trans_dict[prompt_name][0].split('<|endofprompt|>')[1].strip(), "chinese")
                print(f"wav: {prompt_name}")
                print(f"text: {text}")
                print(f"ref_text: {ref_text}")
                utt_wer = wer(ref_text, text)
                wer_total += utt_wer
            utt_cnt += 1

            # AutoPCP
            src_audio.append(samp_wav_path)
            tgt_audio.append(gt_wav_path)

    os.makedirs(result_dir_path, exist_ok=True)
    # SER acc
    if ser_model:
        avg_acc_result = {}
        print(emo_acc_total)
        print(emo_cnt)
        for k, v in emo_acc_total.items():
            avg_acc = v / emo_cnt[k]
            avg_acc_result[k] = avg_acc
            print(f"***根据该情感{emo_cnt[k]}个样本的统计，{k} 情感的平均acc为{avg_acc}")
        avg_acc_result = sort_dict(avg_acc_result)
    
        # EMO SIM
        emo_sim_result = emo_sim_total / utt_cnt 
        print(f"EMO SIM: {emo_sim_result}")
    
    # WER
    if asr_model:
        wer_result = wer_total / utt_cnt
        print(f"WER: {wer_result}")
    
    # AutoPCP
    df = pd.DataFrame({"src_audio": src_audio, "tgt_audio": tgt_audio})

    df.to_csv(pwd / "input.tsv", sep="\t", index=False, encoding='utf-8')
    if os.path.exists(pwd / "output.txt"):
        os.remove(pwd / "output.txt")
    run_autoPCP()
    prosody_sim = 0
    with open(pwd / "output.txt", "r", encoding="utf-8") as f:
        for line in f:
            l_sim = float(line.strip())
            prosody_sim += l_sim
    prosody_sim = prosody_sim / utt_cnt
    print(f"prosody_sim: {prosody_sim}")

    with open(result_dir_path / result_txt_name, "w", encoding="utf-8") as f:
        for k, v in avg_acc_result.items():
            f.write(f"{k} acc: {v}\n")

        f.write(f"EMO SIM: {emo_sim_result}\n")
        f.write(f"WER: {wer_result}\n")
        f.write(f"Prosody SIM: {prosody_sim}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate advantage functions for audio files.")
    parser.add_argument("--num_epoch", required=True, help="epoch number")
    parser.add_argument("--method", required=True, help="method")
    args = parser.parse_args()
    
    ser_model_name = "emotion2vec_base_finetuned"
    ser_model = AutoModel(model=f"iic/{ser_model_name}")
    asr_model_size = "large-v3"
    asr_model = WhisperModel(asr_model_size, device="cuda", compute_type="float16", download_root=str(pwd / "eva_model/models"), local_files_only=True)
    # assert False

    # 实验目录命名：test_方法|参数_epoch_数字
    subdir_list = [f'test_{args.method}_epoch_{args.num_epoch}']
    # for i in range(1, 10, 2):
    #     subdir = f"test_emo_dpo_epoch_{i}"
    #     if os.path.exists(test_root_dir / subdir / method):
    #         subdir_list.append(subdir)
    print(f"对以下目录音频做评估：{subdir_list}")
    
    for subdir in subdir_list:
        evaluate(subdir, ser_model=ser_model, asr_model=asr_model)
