import string
import numpy as np
import subprocess
import os

def remove_punctuation(text, lang):
    special_punc = {"arabic": "؟،؛«»…", "french": "«»…¿¡", "spanish": "¿¡«»…", "chinese": "、。！？，；：“”‘’（）【】…《》·"}
    assert lang in special_punc.keys(), f"错误: {lang}不在特殊标点列表中"
    punctuation = string.punctuation + special_punc[lang]
    translator = str.maketrans("", "", punctuation)
    return text.translate(translator)


def cosine_similarity(a, b):
    # 确保输入为 float32 且形状一致
    assert a.dtype == np.float32 and b.dtype == np.float32
    assert a.shape == b.shape, "数组形状必须相同"

    dot_product = np.dot(a, b)
    
    # 计算 L2 范数
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    # 避免除以零
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    # 返回余弦相似度（范围 [-1, 1]）
    return dot_product / (norm_a * norm_b)

def sort_dict(x):
    return {k: x[k] for k in sorted(x)}

def run_autoPCP():
    # 获取当前工作目录
    current_dir = os.getcwd()
    # 设置输入输出文件路径
    input_tsv = os.path.join(current_dir, "input.tsv")
    output_txt = os.path.join(current_dir, "output.txt")
    # 构建命令列表
    cmd = [
        "python", "-m", "stopes.modules",
        "+compare_audios=AutoPCP_multilingual_v2",
        f"+compare_audios.input_file={input_tsv}",
        "++compare_audios.src_audio_column=src_audio",  
        "++compare_audios.tgt_audio_column=tgt_audio",  
        "+compare_audios.named_columns=true",
        f"+compare_audios.output_file={output_txt}",
        "launcher=local"
    ]
    # 执行命令
    result = subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=True
    )