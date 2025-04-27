"""
    从aishell2随机提取出n个tts text。根据这n个tts text，
    找到casia数据集中6种情感的6n个音频作为prompt wav生成wav_id: emotion <||>tts text 字典，生成对应的wav2tts_text.json文件
"""

import random
import json
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent

emotions = ["angry", "fear", "happy", "sad", "surprise", "neutral"]

# 每种情感的tts text数量
num_each_emotion = 200
output_dict = {}

def make_emotion_corpus(emotion):
    cnt = 0
    corpus_dir = root_dir / "data" / "casia"
    # 3. 读取以emotion为后缀的num_each_emotion个prompt音频name，映射到对应情感的tts text
    for fi in corpus_dir.glob(f"*{emotion}.wav"):
        audio_name = fi.stem
        output_dict[audio_name] = [emotion + "<|endofprompt|>" + random_samples_text[cnt]]
        cnt += 1
        if cnt >= num_each_emotion: 
            break



if __name__ == "__main__":
    tts_text = []
    with open(f"{root_dir}/data/aishell2_tts_text/trans.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip() 
            if not line:
                continue
            id_, text = line.split("\t")
            tts_text.append(text)
    
    assert len(tts_text) >= num_each_emotion
    random_samples_text = random.sample(tts_text, min(num_each_emotion, len(tts_text)))
    for emo in emotions:
        make_emotion_corpus(emo)

    target_path = root_dir / f"wav2tts_text_dpo_{num_each_emotion * len(emotions)}.json"

    with open(f"{target_path}", "w", encoding="utf-8") as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=2)

    print(f"JSON 文件已生成：{target_path}")