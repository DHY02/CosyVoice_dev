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
import os
import sys
from concurrent import futures
import argparse
import cosyvoice_pb2
import cosyvoice_pb2_grpc
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import grpc
import torch
import numpy as np
import psutil
import signal
import time
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/../../..'.format(ROOT_DIR))
sys.path.append('{}/../../../third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')


class CosyVoiceServiceImpl(cosyvoice_pb2_grpc.CosyVoiceServicer):
    def __init__(self, args):
        try:
            self.cosyvoice = CosyVoice(args.model_dir)
        except Exception:
            try:
                self.cosyvoice = CosyVoice2(args.model_dir)
            except Exception:
                raise TypeError('no valid model_type!')
        logging.info('grpc service initialized')

    def AddSpeaker(self, request, context):
        """实现添加 speaker 的 RPC 方法"""
        try:
            logging.info(f'get add speaker request for speaker: {request.spk_name}')
            
            # 检查 speaker 名称是否有效
            if not request.spk_name:
                raise ValueError("Speaker name cannot be empty")
            
            # 检查音频数据是否有效
            if not request.prompt_audio:
                raise ValueError("Prompt audio cannot be empty")

            prompt_speech_16k = torch.from_numpy(np.array(np.frombuffer(request.prompt_audio, dtype=np.int16))).unsqueeze(dim=0)
            prompt_speech_16k = prompt_speech_16k.float() / (2**15)
            # 调用 cosyvoice 的 save_spk 方法
            self.cosyvoice.save_spk(prompt_speech_16k, request.spk_name, request.prompt_text)
            
            # 返回成功响应
            return cosyvoice_pb2.AddSpeakerResponse(
                success=True, 
                message=f"Speaker {request.spk_name} added successfully"
            )
            
        except Exception as e:
            logging.error(f"Error in AddSpeaker: {e}")
            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # 向客户端返回错误信息
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"AddSpeaker failed: {str(e)}")
            return cosyvoice_pb2.AddSpeakerResponse(
                success=False, 
                message=f"AddSpeaker failed: {str(e)}"
            )

    def RemoveSpeaker(self, request, context):
        """实现删除 speaker 的 RPC 方法"""
        try:
            logging.info(f'get remove speaker request for speaker: {request.spk_name}')
            
            # 检查 speaker 名称是否有效
            if not request.spk_name:
                raise ValueError("Speaker name cannot be empty")
            
            # 调用 cosyvoice 的 save_spk 方法
            self.cosyvoice.remove_spk(request.spk_name)
            
            # 返回成功响应
            return cosyvoice_pb2.RemoveSpeakerResponse(
                success=True, 
                message=f"Speaker {request.spk_name} removed successfully"
            )
            
        except Exception as e:
            logging.error(f"Error in RemoveSpeaker: {e}")
            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # 向客户端返回错误信息
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"RemoveSpeaker failed: {str(e)}")
            return cosyvoice_pb2.RemoveSpeakerResponse(
                success=False, 
                message=f"RemoveSpeaker failed: {str(e)}"
            )

    def Inference(self, request, context):
        try:
            if request.HasField('sft_request'):
                logging.info('get sft inference request')
                logging.info(f'{request.sft_request.tts_text}, {request.sft_request.spk_id}')
                model_output = self.cosyvoice.inference_sft(request.sft_request.tts_text, request.sft_request.spk_id)
            elif request.HasField('zero_shot_request'):
                logging.info('get zero_shot inference request')
                prompt_speech_16k = torch.from_numpy(np.array(np.frombuffer(request.zero_shot_request.prompt_audio, dtype=np.int16))).unsqueeze(dim=0)
                prompt_speech_16k = prompt_speech_16k.float() / (2**15)
                model_output = self.cosyvoice.inference_zero_shot(request.zero_shot_request.tts_text,
                                                                request.zero_shot_request.prompt_text,
                                                                prompt_speech_16k)
            elif request.HasField('cross_lingual_request'):
                logging.info('get cross_lingual inference request')
                prompt_speech_16k = torch.from_numpy(np.array(np.frombuffer(request.cross_lingual_request.prompt_audio, dtype=np.int16))).unsqueeze(dim=0)
                prompt_speech_16k = prompt_speech_16k.float() / (2**15)
                model_output = self.cosyvoice.inference_cross_lingual(request.cross_lingual_request.tts_text, prompt_speech_16k)
            elif request.HasField('cross_lingual_spk_request'):
                logging.info('get cross_lingual_spk inference request')
                model_output = self.cosyvoice.inference_cross_lingual_spk(request.cross_lingual_spk_request.tts_text, request.cross_lingual_spk_request.spk_id)
            elif request.HasField('instruct_spk_request'):
                logging.info('get instruct_spk inference request')
                if isinstance(self.cosyvoice, CosyVoice2):
                    model_output = self.cosyvoice.inference_instruct3(request.instruct_spk_request.tts_text, 
                                                                    request.instruct_spk_request.instruct_text,
                                                                    request.instruct_spk_request.spk_id)
                else:
                    model_output = self.cosyvoice.inference_instruct(request.instruct_spk_request.tts_text,
                                                                request.instruct_spk_request.instruct_text,
                                                                request.instruct_spk_request.spk_id)
            elif request.HasField('instruct_request'):
                logging.info('get instruct inference request')
                prompt_speech_16k = torch.from_numpy(np.array(np.frombuffer(request.instruct_request.prompt_audio, dtype=np.int16))).unsqueeze(dim=0)
                prompt_speech_16k = prompt_speech_16k.float() / (2**15)
                if isinstance(self.cosyvoice, CosyVoice2):
                    model_output = self.cosyvoice.inference_instruct2(request.instruct_request.tts_text, 
                                                                    request.instruct_request.instruct_text,
                                                                    prompt_speech_16k)
                else:
                    raise RuntimeError(f"请求失败，当前请求只支持cosyvoice 2.0!")


            logging.info('send inference response')
            for i in model_output:
                response = cosyvoice_pb2.Response()
                response.tts_audio = (i['tts_speech'].numpy() * (2 ** 15)).astype(np.int16).tobytes()
                yield response
        
        except Exception as e:
            logging.error(f"Error in Inference: {e}")
            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # 向客户端返回错误信息
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Inference failed: {str(e)}")
            return


def main():
    grpcServer = grpc.server(futures.ThreadPoolExecutor(max_workers=args.max_conc), maximum_concurrent_rpcs=args.max_conc)
    cosyvoice_pb2_grpc.add_CosyVoiceServicer_to_server(CosyVoiceServiceImpl(args), grpcServer)
    grpcServer.add_insecure_port('0.0.0.0:{}'.format(args.port))
    grpcServer.start()
    logging.info("server listening on 0.0.0.0:{}".format(args.port))
    grpcServer.wait_for_termination()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=50000)
    parser.add_argument('--max_conc',
                        type=int,
                        default=2)
    parser.add_argument('--model_dir',
                        type=str,
                        default='iic/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    args = parser.parse_args()
    main()
