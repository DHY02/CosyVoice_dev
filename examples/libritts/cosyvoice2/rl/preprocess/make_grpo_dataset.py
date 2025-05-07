# 根据训练集和验证集，将几次采样的音频复制到对应的数据集下，train/sample_{i}
import os
import pathlib
import shutil
from pathlib import Path


src_path = Path("/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/data/casia-emo-dpo")

sample_dir = Path("/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/data/sampling_casia_ori")
if __name__ == "__main__":
    for corpus in ["train", "valid"]:
        # 打开训练集，找到所有的文件名称，找到对应的几个采样文件夹，复制其下相同名称的文件到目标目录
        for entry in os.listdir(src_path / corpus / "receive"):
            if entry.endswith(".wav"):
                for dir in os.listdir(sample_dir):
                    for samp in os.listdir(sample_dir / dir):
                        if samp == entry:
                            os.makedirs(src_path / corpus / dir, exist_ok=True)
                            a_id = samp.split('.')[0]
                            t_name = a_id + '.normalized.txt'
                            shutil.copy(sample_dir / dir / samp, src_path / corpus / dir)
                            shutil.copy(src_path / corpus / "receive" / t_name, src_path / corpus / dir)

