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
sys.path.append('/root/autodl-tmp/CosyVoice_dev') 

from async_cosyvoice.async_cosyvoice import AsyncCosyVoice2
from cosyvoice.utils.file_utils import load_wav
random.seed(1)
cosyvoice_dir = Path("/root/autodl-tmp/CosyVoice_dev")

# 参考音频数据集
ref_corpus = "casia"

async def main(method, num_epoch):
    global train_corpus, test_corpus
    result_exp_name=f"{test_corpus}_{train_corpus}_{method}_epoch_{num_epoch}"
    # cosyvoice = AsyncCosyVoice2('./pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=True)
    cosyvoice = AsyncCosyVoice2(str(cosyvoice_dir / 'pretrained_models/CosyVoice2-0.5B'), load_jit=True, load_trt=False, fp16=True)

    # instruct 不能使用 Generater 模式传入text
    task_id = 0

    wav2text = {}
    with open(cosyvoice_dir / f"examples/libritts/cosyvoice2/wav2text_{test_corpus}_test.json", "r", encoding="utf-8") as f:
        wav2text = json.load(f)

    # 选择ref_corpus数据集的所有说话人的所有情感音频各一条，作为参考音频列表
    ref_list = []
    combo_list = []
    ref_dir = Path(cosyvoice_dir / f"examples/libritts/cosyvoice2/data/{ref_corpus}")
    for entry in sorted(os.listdir(ref_dir)):
        if ".wav" in entry:
            # 当该说话人的该情感没有收入ref_list时
            if ref_corpus == "casia":
                spk = entry.split('_')[0]
                emo = entry.split('.')[0].split('_')[-1]
            elif ref_corpus == "m3ed/whole":
                text_path = entry.replace(".wav", ".normalized.txt")
                spk = entry.split('_')[0] + '_' +entry.split('_')[1]
                with open(ref_dir / text_path, "r", encoding="utf-8") as f:
                    emo = f.readline().strip().split('<|endofprompt|>')[0]
            else:
                assert False, "未知的参考数据集"
            
            combo = f"{spk}-{emo}"
            if combo not in combo_list and emo != "Disgust":
                wav_path = ref_dir / entry
                audio = load_wav(wav_path, 16000)
                ref_list.append(audio)
                combo_list.append(combo)

    print(f"combo_list of {ref_list} len: {len(combo_list)}")
    # assert False

    output_dir = Path(cosyvoice_dir / "examples/libritts/cosyvoice2/exp/cosyvoice")
    (output_dir / result_exp_name / "instruct").mkdir(parents=True, exist_ok=True)
    for k, v in wav2text.items():
        tts_text = v[0].split('<|endofprompt|>')[1]
        instruct_text = v[0].split('<|endofprompt|>')[0]
        sample = random.choices(ref_list, k=1)[0]
        audio_data: torch.Tensor = None
        async for chunk in cosyvoice.inference_instruct2(tts_text, instruct_text, sample, stream=False):
            if chunk['tts_speech'] != None:
                chunk_data = chunk['tts_speech'].cpu()
            audio_data = torch.concat([audio_data, chunk_data], dim=1) if audio_data is not None else chunk_data
            del chunk_data
        if audio_data != None:
            audio_data = audio_data.cpu()
            torchaudio.save(str(output_dir / result_exp_name / "instruct" / f"{k}.wav"), audio_data, cosyvoice.sample_rate)
        
        # del audio_data
        # torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate advantage functions for audio files.")
    parser.add_argument("--num_epoch", required=True, help="epoch number")
    parser.add_argument("--method", required=True, help="method")
    parser.add_argument("--train_corpus", required=True, help="train_corpus")
    parser.add_argument("--test_corpus", required=True, help="test_corpus")
    args = parser.parse_args()
    global train_corpus, test_corpus
    if args.test_corpus:
        test_corpus = args.test_corpus 
    if args.train_corpus:
        train_corpus = args.train_corpus 
    asyncio.run(main(args.method, args.num_epoch))