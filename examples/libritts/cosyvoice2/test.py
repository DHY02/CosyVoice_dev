import torch
import math
# model_path = "/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/data/esd-emo-grpo/train/receive/utt2emo-grpoadvs_dict.pt"
# checkpoint = torch.load(model_path, map_location='cpu')
# keys = sorted(checkpoint.keys())[-5:]
# for key in keys:
#     print(f"{key}: {checkpoint[key]}")

# jsd = math.log(1 + 1.1) - math.log(1 + 0.9)
# beta=0.1
# jsd = torch.tensor(jsd)
# smooth_jsd = torch.tanh(jsd / beta)
# print(jsd)

# print(smooth_jsd)

pt_path = "/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/data/esd/valid/receive/utt2speech_token.pt"
d = torch.load(pt_path, map_location='cpu')
keys = sorted(d.keys())[-10:]
for key in keys:
    print(f"key: {key}, len: {len(d[key])}")