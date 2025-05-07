import json
import os
import pathlib
import shutil
from pathlib import Path
from funasr import AutoModel
import torch
import argparse

# Emotion list
model_output_emo_list = ['angry', 'disgusted', 'fear', 'happy', 'neutral', 'other', 'sad', 'surprise', '<unk>']
ser_model_name = "emotion2vec_base_finetuned"

def main(src_dir, tgt_dir, wav2text_path):
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
                    
                    # Check if .adv file exists
                    if adv_file.exists():
                        with open(adv_file, "r", encoding="utf-8") as f:
                            line = f.readline().strip()
                            if aid not in utt2advs:
                                utt2advs[aid] = {}
                            utt2advs[aid][group_id] = round(float(line), 2)
                    else:
                        # Process with SER model
                        wav_path = sub_dir_path / entry
                        res = ser_model.generate(
                            str(wav_path), 
                            granularity="utterance", 
                            extract_embedding=True
                        )
                        
                        # Get emotion from transcription
                        try:
                            instruct_emotion_name = trans_dict[aid][0].split('<|endofprompt|>')[0].lower()
                            scores = res[0]['scores']
                            ind = model_output_emo_list.index(instruct_emotion_name)
                            adv = round(scores[ind], 2)
                            if aid not in utt2advs:
                                utt2advs[aid] = {}
                            utt2advs[aid][group_id] = adv
                        except (KeyError, IndexError, ValueError) as e:
                            print(f"Error processing {aid}: {e}")
                            continue
    torch.save(utt2advs, f"{tgt_dir}/utt2advs_dict.pt")
                        

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate advantage functions for audio files.")
    parser.add_argument("--src_dir", required=True, help="Source directory like train/valid data")
    parser.add_argument("--tgt_dir", required=True, help="Target directory to save utt2advs_dict.pt")
    parser.add_argument("--wav2text_path", required=True, help="Path to wav2tts_text_dpo_ori.json file for calculating emo acc scores")
    
    args = parser.parse_args()
    
    main(args.src_dir, args.tgt_dir, args.wav2text_path)
