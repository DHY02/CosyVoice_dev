import torch

a = torch.load("/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/data/esd/train/receive/utt2emo-grpoadvs_dict.pt")

# Get the last 3 keys of the dictionary
keys = list(a.keys())[-3:]

# Print the last 3 keys and their values
print("Last 3 keys of the dictionary:")
for key in keys:
    print(f"Key: {key}, Value: {a[key]}")

