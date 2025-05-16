import torch

model_path = "/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/data/esd-emo-grpo/train/receive/utt2emo-grpoadvs_dict.pt"
checkpoint = torch.load(model_path, map_location='cpu')
keys = sorted(checkpoint.keys())[-5:]
for key in keys:
    print(f"{key}: {checkpoint[key]}")