"""
    从aishell2数据集中提取出json格式的TTS text
"""

import random
import json
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent

emotion = "angry"

# 创建一个和训练集不重复的测试集（待完成）
with open(root_dir / "tts_text_angry50.json", "r", encoding="utf-8") as f:
    old_train_text = json.load(f)

# 1. 读取数据
with open(f"{root_dir}/data/aishell2_tts_text/trans.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# 2. 解析数据：文本
data = []
for line in lines:
    line = line.strip()  # 去除换行符
    if not line:
        continue
    id_, text = line.split("\t")  # 按制表符分割
    text = emotion + "<|endofprompt|>" + text
    data.append(text)

# 3. 随机抽取 50 个样本（如果数据不足 50 个，则全部抽取）
random_samples = random.sample(data, min(50, len(data)))

output_dict = {}
cnt = 0
train_dir = root_dir / "data" / "casia_train"
# 4. 读取50个prompt音频name
for fi in train_dir.glob(f"*{emotion}.wav"):
    audio_name = fi.stem
    output_dict[audio_name] = [random_samples[cnt]]
    cnt += 1
    if cnt >= 50: 
        break

target_path = root_dir / f"tts_text_{emotion}50.json"
# 5. 保存为 JSON 文件
with open(f"{target_path}", "w", encoding="utf-8") as f:
    json.dump(output_dict, f, ensure_ascii=False, indent=2)

print(f"JSON 文件已生成：{target_path}")
