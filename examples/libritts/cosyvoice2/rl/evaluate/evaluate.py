# 读取两组test音频，比较acc
from pathlib import Path
import os
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import json
from funasr import AutoModel
from faster_whisper import WhisperModel, BatchedInferencePipeline
from jiwer import wer
from utils import char_level, detect_language, remove_punctuation, cosine_similarity, run_autoPCP, sort_dict
import pandas as pd
import argparse
"""
    使用测试集做评估
"""


pwd = Path(__file__).parent
cosyvoice2_dir = pwd.parent.parent

# 模型的输出标签
model_output_emo_list = ['angry', 'disgusted', 'fear', 'happy', 'neutral', 'other', 'sad', 'surprise', '<unk>']

# test音频根目录
test_root_dir = cosyvoice2_dir / "exp/cosyvoice"

# 评测结果文件夹
result_dir_path = pwd / "result"
zhEmo2enEmo = {"冷静": "neutral", "生气": "angry", "快乐": "happy", "伤心": "sad", "惊喜": "surprise"}

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
def evaluate(subdir, ser_model=None, asr_model=None, skip_prosody=False):
    # 声明全局变量，以便在函数内部可以修改它们
    global test_corpus, train_corpus, wav2text_path, prompt2ref_path, ref_corpus_dir
    
    test_dir = test_root_dir / subdir / "instruct"
    dir_len = len(subdir.split('_'))
    # esd_esd_grpo_epoch_0 like
    if dir_len == 5:
        methodAndhyp = subdir.split('_')[2]
        num_epoch = subdir.split('_')[-1]
    elif dir_len == 4:
        # test_方法|参数_epoch_数字 like
        methodAndhyp = subdir.split('_')[1]
        num_epoch = subdir.split('_')[3]
    else:
        raise Exception("未知的文件夹名")
    # 测试集_训练集_方法_epoch_numOfEpoch
    result_txt_name = f"{test_corpus}_{train_corpus}_{methodAndhyp}_epoch_{num_epoch}"
    print(f"即将写入{result_txt_name}")
    emo_acc_total = {}
    emo_cnt = {}

    utt_cnt = 0

    emo_sim_total = 0

    wer_total = 0
    
    trans_dict = {}
    # prompt2ref = {}

    # autoPCP input
    src_audio = []
    tgt_audio = []

    with open(wav2text_path, "r", encoding="utf-8") as f:
        trans_dict = json.load(f)
    # with open(prompt2ref_path, "r", encoding="utf-8") as f:
    #     prompt2ref = json.load(f)
    for wav in os.listdir(test_dir):
        if ".wav" in wav:
            samp_wav_path = os.path.join(test_dir, wav)
            
            # prompt_name = wav.split('.')[0].strip()[:-2]
            # prompt_name = wav.split('.')[0].strip()
            # if prompt_name not in trans_dict:
            #     continue
            wav_id = wav.split('.')[0]
            gt_wav_path = os.path.join(ref_corpus_dir, wav)
            if test_corpus == "esd":
                instruct_emotion_name = trans_dict[wav_id][0].split('<|endofprompt|>')[0]
            else:
                instruct_emotion_name = trans_dict[wav_id][0].split('<|endofprompt|>')[0].lower()
            if detect_language(instruct_emotion_name) == "中文":
                instruct_emotion_name = zhEmo2enEmo[instruct_emotion_name]
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
                segments, _ = asr_model.transcribe(samp_wav_path, beam_size=5, language="zh", initial_prompt="以下是普通话的句子。", batch_size=16)
                segments = list(segments)
                text = remove_punctuation("".join([seg.text for seg in segments]), "chinese")
                ref_text = remove_punctuation(trans_dict[wav_id][0].split('<|endofprompt|>')[1].strip(), "chinese")
                print(f"wav: {wav_id}")
                print(f"trans_text: {text}")
                print(f"ref_text: {ref_text}")
                utt_wer = wer(char_level(ref_text), char_level(text))
                wer_total += utt_wer
                print(f"WER: {utt_wer}")
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
    
    if not skip_prosody:
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

    # 有None则先读取再写入，相当于修改
    if not ser_model or not asr_model or skip_prosody:
        avg_acc_result = {}
        with open(result_dir_path / result_txt_name, "r", encoding="utf-8") as f:
            for line in f:
                if not line:
                    continue
                else:
                    line = line.strip()
                if "acc" in line:
                    if not ser_model:
                        metric, value = line.split('acc: ')
                        avg_acc_result[metric] = value
                elif "EMO SIM" in line:
                    if not ser_model:
                        metric, value = line.split(': ')
                        emo_sim_result = value
                elif "WER" in line:
                    if not asr_model:
                        metric, value = line.split(': ')
                        wer_result = value
                elif "Prosody" in line:
                    if skip_prosody:
                        metric, value = line.split(': ')
                        prosody_sim = value
                else:
                    raise Exception(f"未知的指标: {line}")
    if len(avg_acc_result.keys()) > 0 and emo_sim_result and wer_result and prosody_sim:
        with open(result_dir_path / result_txt_name, "w", encoding="utf-8") as f:
            for k, v in avg_acc_result.items():
                f.write(f"{k} acc: {v}\n")

            f.write(f"EMO SIM: {emo_sim_result}\n")
            f.write(f"WER: {wer_result}\n")
            f.write(f"Prosody SIM: {prosody_sim}\n")
    else:
        print(f"写入失败：{subdir} 因缺少指标放弃写入！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate advantage functions for audio files.")
    parser.add_argument("--num_epoch", required=False, help="epoch number")
    parser.add_argument("--method", required=False, help="method")
    parser.add_argument("--train_corpus", required=False, help="train_corpus")
    parser.add_argument("--test_corpus", required=False, help="test_corpus")
    parser.add_argument("--skip_ser", action='store_true',
                        default=False, help="skip_ser")
    parser.add_argument("--skip_asr", action='store_true',
                        default=False, help="skip_asr")
    args = parser.parse_args()
    
    # 更新全局变量
    global test_corpus, train_corpus, wav2text_path, prompt2ref_path, ref_corpus_dir
    if args.test_corpus:
        test_corpus = args.test_corpus
        # 更新依赖于test_corpus的路径
        wav2text_path = cosyvoice2_dir / f"wav2text_{test_corpus}_test.json"
        prompt2ref_path = cosyvoice2_dir / f"promtwav2refwav_{test_corpus}_test.json"
        # 测试集文本参考音频目录
        ref_corpus_dir = cosyvoice2_dir / f"data/{test_corpus}/test/receive"

    if args.train_corpus:
        train_corpus = args.train_corpus
    
    ser_model_name = "emotion2vec_base_finetuned"
    ser_model = AutoModel(model=f"iic/{ser_model_name}") if not args.skip_ser else None
    asr_model_size = "large-v3"
    asr_model = WhisperModel(asr_model_size, device="cuda", compute_type="float16", 
        download_root=str(pwd / "eva_model/models"), local_files_only=True) if not args.skip_asr else None
    batched_model = BatchedInferencePipeline(model=asr_model) if not args.skip_asr else None
    asr_model = batched_model if batched_model else asr_model 
    # assert False

    # 实验目录命名：test_方法|参数_epoch_数字
    subdir_list = [f'{test_corpus}_{train_corpus}_{args.method}_epoch_{args.num_epoch}']
    
    # for i in range(1, 10, 2):
    #     subdir = f"test_emo_dpo_epoch_{i}"
    #     if os.path.exists(test_root_dir / subdir / method):
    #         subdir_list.append(subdir)


    # 对exp下所有没有做过评估的音频目录做评估
    # done_methods = []
    # result_dir = "/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/rl/evaluate/result"
    # for result in os.listdir(result_dir):
    #     res_path = os.path.join(result_dir, result)
    #     if not os.path.isdir(res_path):
    #         method = result.split('_')[1]
    #         done_methods.append(method)
    # for exp_dir in os.listdir(test_root_dir):
    #     exp_path = os.path.join(test_root_dir, exp_dir, "instruct")
        
    #     if os.path.isdir(exp_path):
    #         method = exp_dir.split('_')[1]
    #         if "DPO" in exp_dir or method not in done_methods:
    #             continue
    #         subdir_list.append(exp_dir)

    print(f"对以下目录音频做评估：{subdir_list}")
    for subdir in subdir_list:
        evaluate(subdir, ser_model=ser_model, asr_model=asr_model, skip_prosody=False)
