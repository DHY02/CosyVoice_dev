#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import logging
import os
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import multiprocessing
import time
import torch
import numpy as np

def job(utt_list, parquet_file, utt2parquet_file, spk2parquet_file):
    global utt2wav, utt2text, utt2spk
    global utt2embedding, spk2embedding, utt2speech_token
    global utt2reject_speech_token, utt2emo_dpo_reject_speech_token
    global utt2advs_dict, utt2reject_speech_token_dict
    start_time = time.time()
    data_list = []
    for utt in tqdm(utt_list):
        data = open(utt2wav[utt], 'rb').read()
        data_list.append(data)
    wav_list = [utt2wav[utt] for utt in utt_list]
    text_list = [utt2text[utt] for utt in utt_list]
    spk_list = [utt2spk[utt] for utt in utt_list]
    uttembedding_list = [utt2embedding[utt] for utt in utt_list]
    spkembedding_list = [spk2embedding[utt2spk[utt]] for utt in utt_list]
    speech_token_list = [utt2speech_token[utt] for utt in utt_list]

    if utt2reject_speech_token:
        reject_speech_token_list = [utt2reject_speech_token[utt] for utt in utt_list]

    if utt2emo_dpo_reject_speech_token:
        utt2emo_dpo_reject_speech_token_list = [utt2emo_dpo_reject_speech_token[utt] for utt in utt_list]

    if utt2reject_speech_token_dict:
        reject_speech_tokens_dict_list = [utt2reject_speech_token_dict[utt] for utt in utt_list]
    if utt2advs_dict:
        advantages_dict_list = [utt2advs_dict[utt] for utt in utt_list]

    # 保存到parquet,utt2parquet_file,spk2parquet_file
    df = pd.DataFrame()

    # df的每一列是一种数据，每一行是一个样本
    df['utt'] = utt_list
    df['wav'] = wav_list
    df['audio_data'] = data_list
    df['text'] = text_list
    df['spk'] = spk_list
    df['utt_embedding'] = uttembedding_list
    df['spk_embedding'] = spkembedding_list
    df['speech_token'] = speech_token_list
    if utt2reject_speech_token:
        df['reject_speech_token'] = reject_speech_token_list
    if utt2emo_dpo_reject_speech_token:
        print("emo_dpo_reject_speech_token已保存")
        df['emo_dpo_reject_speech_token'] = utt2emo_dpo_reject_speech_token_list
    if utt2reject_speech_token_dict:
        print("grpo reject_speech_tokens_dict已保存")
        df['reject_speech_tokens_dict'] = reject_speech_tokens_dict_list
    if utt2advs_dict:
        print("grpo advantages_dict_list已保存")
        df['advantages_dict'] = advantages_dict_list
   
    df.to_parquet(parquet_file)
    with open(utt2parquet_file, 'w') as f:
        json.dump({k: parquet_file for k in utt_list}, f, ensure_ascii=False, indent=2)
    with open(spk2parquet_file, 'w') as f:
        json.dump({k: parquet_file for k in list(set(spk_list))}, f, ensure_ascii=False, indent=2)
    logging.info('spend time {}'.format(time.time() - start_time))



def error_cb(e):
    print(f"子进程报错: {repr(e)}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_utts_per_parquet',
                        type=int,
                        default=1000,
                        help='num utts per parquet')
    parser.add_argument('--num_processes',
                        type=int,
                        default=1,
                        help='num processes for make parquets')
    parser.add_argument('--src_dir',
                        type=str)
    parser.add_argument('--des_dir',
                        type=str)
    parser.add_argument('--dpo',
                        action='store_true',
                        default=False,
                        help='Use Direct Preference Optimization')
    parser.add_argument('--emo_dpo',
                        action='store_true',
                        default=False,
                        help='Use Emo-dpo')
    parser.add_argument('--grpo',
                        action='store_true',
                        default=False,
                        help='Use GRPO')
    parser.add_argument('--use_grpo_new_loss',
                        default=False,
                        help='Use grpo_new_loss')
    args = parser.parse_args()
    utt2wav, utt2text, utt2spk = {}, {}, {}
    with open('{}/wav.scp'.format(args.src_dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2wav[l[0]] = l[1]
    with open('{}/text'.format(args.src_dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2text[l[0]] = ' '.join(l[1:])
    with open('{}/utt2spk'.format(args.src_dir)) as f:
        for l in f:
            l = l.replace('\n', '').split()
            utt2spk[l[0]] = l[1]
    utt2embedding = torch.load('{}/utt2embedding.pt'.format(args.src_dir))
    spk2embedding = torch.load('{}/spk2embedding.pt'.format(args.src_dir))
    utt2speech_token = torch.load('{}/utt2speech_token.pt'.format(args.src_dir))
    if args.dpo:
        utt2reject_speech_token = torch.load('{}/utt2reject_speech_token.pt'.format(args.src_dir))
        # print(f"utt2reject_speech_token: {utt2reject_speech_token}")
        if args.emo_dpo:
            utt2emo_dpo_reject_speech_token = torch.load('{}/utt2emo_dpo_reject_speech_token.pt'.format(args.src_dir)) 
            # print(f"utt2emo_dpo_reject_speech_token: {utt2emo_dpo_reject_speech_token}")
        else:
            utt2emo_dpo_reject_speech_token = None
    else:
        utt2reject_speech_token = None
        utt2emo_dpo_reject_speech_token = None

    if args.grpo:
        # 要得到List[reject_speech_tokens_dict]: [{samp_i: reject_speech_tokens_i}]作为df新的一列，只需拿到嵌套的字典utt: reject_speech_tokens_dict即可
        # 遍历utt2reject_speech_token{1,2,3…}.pt，得到嵌套字典
        utt2reject_speech_token_dict = {}
        for entry in os.listdir(args.src_dir):
            if "utt2reject_speech_token" in entry and entry != "utt2reject_speech_token.pt":
                group_id = entry.split('.')[0][-1]
                utt2reject_speech_token = torch.load(f'{args.src_dir}/{entry}')
                for utt, tokens in utt2reject_speech_token.items():
                    if utt not in utt2reject_speech_token_dict:
                        utt2reject_speech_token_dict[utt] = {}
                    utt2reject_speech_token_dict[utt][group_id] = tokens

        # 要得到List[advantages_dict]: [{{1: adv}, {2: adv}, …}]作为df新的一列，只需要拿到嵌套的字典utt: {{1: adv}, {2: adv}, …}
        if args.use_grpo_new_loss:
            utt2advs_dict = torch.load('{}/utt2emo-grpoadvs_dict.pt'.format(args.src_dir))
        else:
            utt2advs_dict = torch.load('{}/utt2advs_dict.pt'.format(args.src_dir))

    else:
        utt2reject_speech_token_dict = None
        utt2advs_dict = None
    utts = list(utt2wav.keys())

    # Using process pool to speedup
    pool = multiprocessing.Pool(processes=args.num_processes)
    parquet_list, utt2parquet_list, spk2parquet_list = [], [], []
    for i, j in enumerate(range(0, len(utts), args.num_utts_per_parquet)):
        parquet_file = os.path.join(args.des_dir, 'parquet_{:09d}.tar'.format(i))
        utt2parquet_file = os.path.join(args.des_dir, 'utt2parquet_{:09d}.json'.format(i))
        spk2parquet_file = os.path.join(args.des_dir, 'spk2parquet_{:09d}.json'.format(i))
        parquet_list.append(parquet_file)
        utt2parquet_list.append(utt2parquet_file)
        spk2parquet_list.append(spk2parquet_file)
        pool.apply_async(job, (utts[j: j + args.num_utts_per_parquet], parquet_file, utt2parquet_file, spk2parquet_file), error_callback=error_cb)
    pool.close()
    pool.join()

    with open('{}/data.list'.format(args.des_dir), 'w', encoding='utf8') as f1, \
            open('{}/utt2data.list'.format(args.des_dir), 'w', encoding='utf8') as f2, \
            open('{}/spk2data.list'.format(args.des_dir), 'w', encoding='utf8') as f3:
        for name in parquet_list:
            f1.write(name + '\n')
        for name in utt2parquet_list:
            f2.write(name + '\n')
        for name in spk2parquet_list:
            f3.write(name + '\n')
