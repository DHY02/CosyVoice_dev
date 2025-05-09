import json
import os
import pathlib
import shutil
from pathlib import Path
from funasr import AutoModel
import torch
import argparse
import pandas as pd
from examples.libritts.cosyvoice2.rl.evaluate.utils import cosine_similarity, run_autoPCP

# Emotion list
model_output_emo_list = ['angry', 'disgusted', 'fear', 'happy', 'neutral', 'other', 'sad', 'surprise', '<unk>']
ser_model_name = "emotion2vec_base_finetuned"
pwd = Path(__file__).parent

def main(src_dir, tgt_dir, wav2text_path, new_loss):
    # Initialize SER model
    ser_model = AutoModel(model=f"iic/{ser_model_name}")
    
    # Load transcription dictionary
    with open(wav2text_path, "r", encoding="utf-8") as f:
        trans_dict = json.load(f)

    

    # Convert string paths to Path objects
    src_path = Path(src_dir)
    tgt_path = Path(tgt_dir)
    
    # Ensure target directory exists
    tgt_path.mkdir(parents=True, exist_ok=True)

    # 嵌套字典{utt: {{"1": adv}, {"2": adv}, …}}，"1"代表samp_1文件夹，"-1"表示receive
    utt2advs = {}

    # AutoPCP
    src_audio = []
    tgt_audio = []

    reward_ps_list = []
    if not src_path.exists():
        raise Exception()

    # 对于每个utt，找到receive、samp_i 文件夹下所有adv文件内的advantage，写入utt2advs，如果没有adv文件则输入模型得到
    for sub_dir in os.listdir(src_path):
        if "receive" in sub_dir or "samp" in sub_dir:
            sub_dir_path = src_path / sub_dir
            group_id = "-1" if "receive" in sub_dir else sub_dir.split('_')[-1]
            for entry in os.listdir(sub_dir_path):
                if entry.endswith('.wav'):
                    aid = entry.split('.')[0].strip()
                    adv_file = sub_dir_path / f"{aid}.adv"
                    wav_path =  os.path.join(sub_dir_path, entry)
                    ser_res = None

                    # 读取reward_a
                    if adv_file.exists():
                        with open(adv_file, "r", encoding="utf-8") as f:
                            line = f.readline().strip()
                            if aid not in utt2advs:
                                utt2advs[aid] = {}
                            utt2advs[aid][group_id] = round(float(line), 2)
                    else:
                        # Process with SER model       
                        ser_res = ser_model.generate(
                            wav_path, 
                            granularity="utterance", 
                            extract_embedding=True
                        )
                        
                        # Get emotion from transcription
                        try:
                            instruct_emotion_name = trans_dict[aid][0].split('<|endofprompt|>')[0].lower()
                            scores = ser_res[0]['scores']
                            ind = model_output_emo_list.index(instruct_emotion_name)
                            adv = round(scores[ind], 2)
                            if aid not in utt2advs:
                                utt2advs[aid] = {}
                            utt2advs[aid][group_id] = adv
                        except (KeyError, IndexError, ValueError) as e:
                            print(f"Error processing {aid}: {e}")
                            continue
                    
                    if new_loss:
                        # emo-grpo的奖励函数改为λ_A Reward_A+λ_ES Reward_ES+λ_PS Reward_PS, 其中负样本的Reward_A 设为0
                        gt_wav_path = os.path.join(src_path / "receive", f"{aid}.wav")
                        reward_es = None
                        es_adv_file = sub_dir_path / f"{aid}.esadv"

                        # 读取 Reward_es, 即Emo-SIM
                        if es_adv_file.exists():
                            with open(es_adv_file, "r", encoding="utf-8") as f:
                                line = f.readline().strip()
                        else:
                            line = None

                        if not line:
                            if group_id == "-1":
                                reward_es = 1
                            else:
                                if not ser_res:
                                    ser_res = ser_model.generate(
                                        wav_path, 
                                        granularity="utterance", 
                                        extract_embedding=True
                                    )
                                predict_feat = ser_res[0]['feats']
                                gt_res = ser_model.generate(gt_wav_path, granularity="utterance", extract_embedding=True)
                                gt_feat = gt_res[0]['feats']
                                reward_es = cosine_similarity(predict_feat, gt_feat)
                                with open(es_adv_file, "w", encoding="utf-8") as f:
                                    f.write(str(reward_es))
                        else:
                            reward_es = round(float(line), 2)
                        reward_a = utt2advs[aid][group_id]
                        utt2advs[aid][group_id] = reward_a * 0.5 + reward_es * 0.2

                        # 读取 Reward_ps
                        ps_adv_file = wav_path.split('.')[0] + ".psadv"
                        if os.path.exists(ps_adv_file):
                            with open(ps_adv_file, "r", encoding="utf-8") as f:
                                line = f.readline().strip()
                            reward_ps = round(float(line), 2)
                            reward_ps_list.append(reward_ps)
                            utt2advs[aid][group_id] = round(utt2advs[aid][group_id] + reward_ps * 0.3, 2)
                        else:
                            src_audio.append(str(wav_path))
                            tgt_audio.append(str(gt_wav_path))
                        

    # 读取reward_ps
    if new_loss and src_audio != []:
        # AutoPCP
        df = pd.DataFrame({"src_audio": src_audio, "tgt_audio": tgt_audio})
        input_file = pwd.parent / "evaluate" / "input.tsv"
        output_file = pwd.parent / "evaluate" / "output.txt"
        df.to_csv(input_file, sep="\t", index=False, encoding='utf-8')
        
        if os.path.exists(output_file):
            os.remove(output_file)
        run_autoPCP()

        # 读取output.txt在每个src的Prosody SIM分数
        with open(output_file, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        assert df.shape[0] == len(lines)
        float_list = list(map(float, lines))
        reward_ps_list.extend(float_list)
        min_reward_ps = min(reward_ps_list)
        max_reward_ps = max(reward_ps_list)
        src_audio_col = df.columns.get_loc("src_audio")
        for i, p_sim in enumerate(lines):
            # /root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/data/casia-emo-grpo/train/samp_1/zhaoquanyin_250_neutral.wav
            src_audio_col = df.columns.get_loc("src_audio")
            ps_adv_file = df.iloc[i, src_audio_col].split('.')[0] + ".psadv"
            aid = df.iloc[i, src_audio_col].split('/')[-1].split('.')[0]
            group = df.iloc[i, src_audio_col].split('/')[-2]
            group_id = "-1" if group == "receive" else group.split('_')[-1]
            normalized_ps = (float(p_sim) - min_reward_ps) / (max_reward_ps - min_reward_ps)
            utt2advs[aid][group_id] = round(utt2advs[aid][group_id] + normalized_ps * 0.3, 2)
            with open (ps_adv_file, "w", encoding="utf-8") as f:
                f.write(str(normalized_ps))

    if new_loss:
        torch.save(utt2advs, f"{tgt_dir}/utt2emo-grpoadvs_dict.pt")
    else:
        torch.save(utt2advs, f"{tgt_dir}/utt2advs_dict.pt")
                        

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate advantage functions for audio files.")
    parser.add_argument("--src_dir", required=True, help="Source directory like train/valid data")
    parser.add_argument("--tgt_dir", required=True, help="Target directory to save utt2advs_dict.pt")
    parser.add_argument("--wav2text_path", required=True, help="Path to wav2tts_text_dpo_ori.json file for calculating emo acc scores")
    parser.add_argument("--new_loss", action='store_true', default=False, help="Emo-GRPO new_loss")
    args = parser.parse_args()
    
    main(args.src_dir, args.tgt_dir, args.wav2text_path, args.new_loss)
