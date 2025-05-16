# 根据训练集和验证集，将几次采样的音频复制到对应的数据集下，train/sample_{i}
import os
import pathlib
import shutil
from pathlib import Path

corpus = "esd"
src_path = Path(f"/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/data/{corpus}")

sample_dir = Path(f"/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/exp/cosyvoice/data/sampling_{corpus}")
if __name__ == "__main__":
    for sub_corpus in ["train", "valid"]:
        # 打开训练集，找到所有的文件名称，找到对应的几个采样文件夹，复制其下相同名称的文件到目标目录
        for entry in os.listdir(src_path / sub_corpus / "receive"):
            if entry.endswith(".wav"):
                # 采样得到的音频名为aid.wav，原数据集的音频名为aid_emotion.wav
                entry_no_emo = "_".join(entry.split('_')[:-1]) + ".wav"
                emo = entry.split('.')[0].split('_')[-1]
                for dir in os.listdir(sample_dir):
                    for samp in os.listdir(sample_dir / dir):
                        if samp == entry_no_emo:
                            os.makedirs(src_path / sub_corpus / dir, exist_ok=True)
                            a_id = samp.split('.')[0]
                            t_name = a_id + f"_{emo}" + '.normalized.txt'
                            shutil.copy(sample_dir / dir / samp, src_path / sub_corpus / dir / f"{a_id}_{emo}.wav")
                            shutil.copy(src_path / sub_corpus / "receive" / t_name, src_path / sub_corpus / dir)

