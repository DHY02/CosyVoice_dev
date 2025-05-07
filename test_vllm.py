import argparse
import os
from pathlib import Path
import random
import time
import asyncio
import torch
import torchaudio

import sys
import json

sys.path.append('third_party/Matcha-TTS')

from async_cosyvoice.async_cosyvoice import AsyncCosyVoice2
from cosyvoice.utils.file_utils import load_wav
import random
import numpy as np
import torch

# 训练集
corpus = "casia_ori"

# 参考音频数据集
ref_corpus = "m3ed/whole"

def set_seeds(base_seed):
    """设置所有随机种子"""
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed_all(base_seed)
    # 为了确保可复现性，添加以下设置
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def get_dynamic_seed(base_seed=1986):
    """获取动态种子"""
    # 使用当前小时数作为偏移量 (0-23)
    hour_offset = int(time.strftime("%S"))  
    return base_seed + hour_offset

async def main():
    # # cosyvoice = AsyncCosyVoice2('./pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=True)
    # cosyvoice = AsyncCosyVoice2('/home/CosyVoice2-0.5B', load_jit=True, load_trt=True, fp16=True)

    # # instruct 不能使用 Generater 模式传入text
    # task_id = 0

    # wav2text = {}
    # with open(f"examples/libritts/cosyvoice2/wav2text_samp_{corpus}.json", "r", encoding="utf-8") as f:
    #     wav2text = json.load(f)

    # # 选择ref_corpus数据集的所有说话人的所有情感音频各一条，作为参考音频列表
    # ref_list = []
    # combo_list = []
    # ref_dir = Path(f"examples/libritts/cosyvoice2/data/{ref_corpus}")
    # for entry in os.listdir(ref_dir):
    #     if ".wav" in entry:
    #         # 当该说话人的该情感没有收入ref_list时
    #         if ref_corpus == "casia":
    #             spk = entry.split('_')[0]
    #             emo = entry.split('.')[0].split('_')[-1]
    #         elif ref_corpus == "m3ed/whole":
    #             text_path = entry.replace(".wav", ".normalized.txt")
    #             spk = entry.split('_')[0] + '_' +entry.split('_')[1]
    #             with open(ref_dir / text_path, "r", encoding="utf-8") as f:
    #                 emo = f.readline().strip().split('<|endofprompt|>')[0]
    #         else:
    #             assert False, "未知的参考数据集"
            
    #         combo = f"{spk}-{emo}"
    #         if combo not in combo_list and emo != "Disgust":
    #             wav_path = ref_dir / entry
    #             audio = load_wav(wav_path, 16000)
    #             ref_list.append(audio)
    #             combo_list.append(combo)

    # print(f"combo_list of {ref_list} len: {len(combo_list)}")
    # # assert False

    # output_dir = Path("examples/libritts/cosyvoice2/exp/cosyvoice")
    # (output_dir / f"sampling_{corpus}" / args.output_subdir).mkdir(parents=True, exist_ok=True)
    # for k, v in wav2text.items():
    #     tts_text = v[0].split('<|endofprompt|>')[1]
    #     instruct_text = v[0].split('<|endofprompt|>')[0]
    #     sample = random.choices(ref_list, k=1)[0]
    #     audio_data: torch.Tensor = None
    #     async for chunk in cosyvoice.inference_instruct2(tts_text, instruct_text, sample, stream=False):
    #         if chunk['tts_speech'] != None:
    #             chunk_data = chunk['tts_speech'].cpu()
    #         audio_data = torch.concat([audio_data, chunk_data], dim=1) if audio_data is not None else chunk_data
    #         del chunk_data
    #     if audio_data != None:
    #         audio_data = audio_data.cpu()
    #         torchaudio.save(str(output_dir / f"sampling_{corpus}" / args.output_subdir / f"{k}.wav"), audio_data, cosyvoice.sample_rate)
        
        # del audio_data
        # torch.cuda.empty_cache()
    prompt_text = '希望你以后能够做得比我还好哟'
    prompt_speech_16k = load_wav('/root/autodl-tmp/CosyVoice_dev/asset/zero_shot_prompt.wav', 16000)
    tts_text = "收到好友从远方寄来的生日礼物，真是太好了"
    tgt_dir = "test_vllm_wavs"
    os.makedirs(tgt_dir, exist_ok=True)
    # cosyvoice = AsyncCosyVoice2('./pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=True)
    cosyvoice = AsyncCosyVoice2('/root/autodl-tmp/CosyVoice_dev/pretrained_models/CosyVoice2-0.5B', load_jit=True, load_trt=False, fp16=True)
    for i in range(10):
        set_seeds(i)
        audio_data: torch.Tensor = None
        async for chunk in cosyvoice.inference_instruct2(tts_text, 'angry', prompt_speech_16k, stream=False):
            if chunk['tts_speech'] != None:
                chunk_data = chunk['tts_speech'].cpu()
            audio_data = torch.concat([audio_data, chunk_data], dim=1) if audio_data is not None else chunk_data
            if audio_data != None:
                audio_data = audio_data.cpu()
        torchaudio.save(f'{tgt_dir}/instruct2_{i}.wav', audio_data, cosyvoice.sample_rate)


        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_subdir', type=str, default="")
    args = parser.parse_args()

    asyncio.run(main())