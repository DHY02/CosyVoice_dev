import sys
import os

import torch
import torchaudio
import librosa

lower_sr = 16000
high_sr = 22050


def get_spk_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 构建 speakers 目录的绝对路径
    speakers_dir = os.path.join(current_dir, "..", "speakers")
    os.makedirs(speakers_dir, exist_ok=True)
    return speakers_dir

def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    max_val = 0.8

    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )

    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val

    zeros = torch.zeros(1, int(high_sr * 0.2))

    # print(speech, zeros)

    speech = torch.concat([speech, zeros], dim=1)
    
    return speech

def load_spk_from_wav(waveform, sr, cosyvoice, prompt_text_token, prompt_text_token_len):
    assert sr == 24000
    

    # target_wav_high = torchaudio.transforms.Resample(sample_rate, high_sr)(target_wav)
    # target_wav_high = postprocess(target_wav_high)
    # target_wav_lower = torchaudio.transforms.Resample(high_sr, lower_sr)(target_wav_high)

    prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=sr)(waveform)
    speech_feat, speech_feat_len = cosyvoice.frontend._extract_speech_feat(prompt_speech_resample)
    speech_token, speech_token_len = cosyvoice.frontend._extract_speech_token(waveform)
    if sr == 24000:
        # cosyvoice2, force speech_feat % speech_token = 2
        token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
        speech_feat, speech_feat_len[:] = speech_feat[:, :2 * token_len], 2 * token_len
        speech_token, speech_token_len[:] = speech_token[:, :token_len], token_len
    embedding = cosyvoice.frontend._extract_spk_embedding(waveform)

    return {
        "speech_feat": speech_feat,
        "speech_feat_len": speech_feat_len,
        "speech_token": speech_token,
        "speech_token_len": speech_token_len,
        "embedding": embedding,
        "prompt_text_len": prompt_text_token_len,
        "prompt_text": prompt_text_token
    }

def load_spk_from_pt(spk_id, spk_dir):
    spk_pt = os.path.join(spk_dir, f"{spk_id}.pt")

    if os.path.exists(spk_pt) and os.path.isfile(spk_pt):
        return torch.load(spk_pt)

    return None

def scan_spks_from_file(spk_dir = "./speakers"):
    spks = []

    for spk_pt in os.listdir(spk_dir):
        if not spk_pt.endswith('.pt'):
            continue

        full_spk_pt = os.path.join(spk_dir, spk_pt)
        spk_id = spk_pt.replace(".pt", "")

        if os.path.exists(full_spk_pt) and os.path.isfile(full_spk_pt):
            spks.append(spk_id)

    return spks