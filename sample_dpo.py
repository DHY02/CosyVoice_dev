import time
import asyncio
import torch
import torchaudio

import sys
import json
sys.path.append('third_party/Matcha-TTS')

from async_cosyvoice.async_cosyvoice import AsyncCosyVoice2
from cosyvoice.utils.file_utils import load_wav

async def main():
    prompt_text = '希望你以后能够做得比我还好哟'
    prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)

    # cosyvoice = AsyncCosyVoice2('./pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=True)
    cosyvoice = AsyncCosyVoice2('/home/CosyVoice2-0.5B', load_jit=True, load_trt=True, fp16=True)

    # instruct 不能使用 Generater 模式传入text
    task_id = 0

    wav2text = {}
    with open("examples/libritts/cosyvoice2/wav2text_samp.json", "r", encoding="utf-8") as f:
        wav2text = json.load(f)

    # tts_text = '收到好友从远方寄来的生日礼物[breath]，那份意外的惊喜与深深的祝福[breath]让我心中充满了甜蜜的快乐，笑容如花儿般绽放。'

    for k, v in wav2text.items():
        tts_text = v.split('<|endofprompt|>')
        chunk_num = 0
        audio_data: torch.Tensor = None
        async for chunk in cosyvoice.inference_instruct2(tts_text, '用四川话说这句话', prompt_speech_16k, stream=False):
            audio_data = torch.concat([audio_data, chunk['tts_speech']], dim=1) if audio_data is not None else chunk['tts_speech']
            chunk_num += 1
        torchaudio.save('instruct2_{}.wav'.format(task_id), audio_data, cosyvoice.sample_rate)
        # print(f'任务完成，生成 {chunk_num} 个片段')

if __name__ == "__main__":
    asyncio.run(main())