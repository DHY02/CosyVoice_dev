#!/usr/bin/env python3

import os
import sys
import torch
import argparse
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_deepspeed_to_pytorch(checkpoint_path, output_path, epoch):
    """
    将DeepSpeed检查点转换为单个PyTorch模型文件
    
    Args:
        checkpoint_path: DeepSpeed检查点目录路径 (如 epoch_0_whole)
        output_path: 输出的PyTorch模型文件路径
        epoch: 模型轮次
    """
    try:
        # 检查检查点目录是否存在
        if not os.path.exists(checkpoint_path):
            logger.error(f"检查点目录不存在: {checkpoint_path}")
            return False
            
        # 获取模型状态文件路径
        model_file = os.path.join(checkpoint_path, "mp_rank_00_model_states.pt")
        if not os.path.exists(model_file):
            logger.error(f"模型文件不存在: {model_file}")
            return False
            
        logger.info(f"加载DeepSpeed模型文件: {model_file}")
        checkpoint = torch.load(model_file, map_location="cpu")
        
        # 检查并记录模型结构
        logger.info(f"模型checkpoint键: {list(checkpoint.keys())}")
        
        # 提取模型权重
        if "module" in checkpoint and "model" in checkpoint["module"]:
            logger.info("从 checkpoint['module']['model'] 提取权重")
            model_state = checkpoint["module"]["model"]
        elif "module" in checkpoint:
            logger.info("从 checkpoint['module'] 提取权重")
            model_state = checkpoint["module"]
        else:
            logger.info("直接使用整个checkpoint作为模型状态")
            model_state = checkpoint
        
        # 保存为单一PyTorch模型文件
        logger.info(f"保存模型到: {output_path}")
        torch.save(model_state, output_path)
        temp_path = os.path.join(checkpoint_path, f"epoch_{epoch}_whole.pt")
        shutil.copy(output_path, temp_path)
        logger.info("模型转换成功!")
        return True
        
    except Exception as e:
        logger.exception(f"模型转换过程中出错: {str(e)}")
        
        # 尝试简单提取和保存
        try:
            logger.info("尝试简单提取方法...")
            checkpoint = torch.load(model_file, map_location="cpu")
            
            # 尝试进一步查看结构
            if isinstance(checkpoint, dict):
                logger.info(f"顶级键: {list(checkpoint.keys())}")
                if "module" in checkpoint:
                    module_keys = list(checkpoint["module"].keys())
                    logger.info(f"module键: {module_keys}")
                    
                    # 如果"model"是其中一个键
                    if "model" in checkpoint["module"]:
                        model_keys = list(checkpoint["module"]["model"].keys())[:10]
                        logger.info(f"model键(前10个): {model_keys}")
            
            # 直接保存整个检查点
            logger.info(f"直接保存整个检查点到: {output_path}")
            torch.save(checkpoint, output_path)
            
            if os.path.exists(output_path):
                logger.info("备用方法成功")
                return True
        except Exception as backup_error:
            logger.exception(f"备用提取方法失败: {str(backup_error)}")
            
        return False


def copy_torch_model(src_path, dst_path):
    """
    复制已有的PyTorch模型文件
    
    Args:
        src_path: 源模型文件路径
        dst_path: 目标模型文件路径
    """
    try:
        logger.info(f"正在复制模型: {src_path} -> {dst_path}")
        if not os.path.exists(src_path):
            logger.error(f"源模型文件不存在: {src_path}")
            return False
            
        shutil.copy2(src_path, dst_path)
        logger.info("模型复制成功!")
        return True
    except Exception as e:
        logger.exception(f"模型复制失败: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="转换或复制模型文件")
    parser.add_argument("--framework", required=True, choices=["torch_ddp", "deepspeed"], 
                        help="训练框架类型")
    parser.add_argument("--src_dir", required=True, help="源模型目录")
    parser.add_argument("--dst_dir", required=True, help="目标模型目录")
    parser.add_argument("--epoch", required=True, help="模型轮次")
    
    args = parser.parse_args()
    
    if args.framework == "torch_ddp":
        src_path = os.path.join(args.src_dir, f"epoch_{args.epoch}_whole.pt")
        dst_path = os.path.join(args.dst_dir, "llm.pt")
        success = copy_torch_model(src_path, dst_path)
    else:  # deepspeed
        src_path = os.path.join(args.src_dir, f"epoch_{args.epoch}_whole")
        dst_path = os.path.join(args.dst_dir, "llm.pt")
        success = convert_deepspeed_to_pytorch(src_path, dst_path, args.epoch)
    
    sys.exit(0 if success else 1)