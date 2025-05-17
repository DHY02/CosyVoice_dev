#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import sys


def load_config(config_path):
    """加载DeepSpeed配置文件"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到配置文件 {config_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"错误：配置文件 {config_path} 不是有效的JSON格式")
        sys.exit(1)


def save_config(config, config_path):
    """保存DeepSpeed配置文件"""
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"已将配置保存到 {config_path}")
    except Exception as e:
        print(f"保存配置文件时出错: {e}")
        sys.exit(1)


def set_learning_rate(config_path, lr):
    """修改配置文件中的学习率"""
    config = load_config(config_path)
    
    # 检查配置结构
    if 'optimizer' not in config:
        print("错误：配置文件中没有 optimizer 部分")
        sys.exit(1)
    
    if 'params' not in config['optimizer']:
        print("错误：配置文件中没有 optimizer.params 部分")
        sys.exit(1)
    
    # 获取当前学习率
    current_lr = config['optimizer']['params'].get('lr', "未设置")
    print(f"当前学习率: {current_lr}")
    
    # 更新学习率
    config['optimizer']['params']['lr'] = lr
    print(f"已将学习率更新为: {lr}")
    
    # 保存配置
    save_config(config, config_path)
    return True


def main():
    parser = argparse.ArgumentParser(description="设置DeepSpeed配置文件中的学习率")
    parser.add_argument('--config', type=str, required=True,
                       help='DeepSpeed配置文件路径')
    parser.add_argument('--lr', type=float, required=True, 
                       help='新的学习率值')
    
    args = parser.parse_args()
    
    # 如果提供的是相对路径，转换为绝对路径
    config_path = os.path.abspath(args.config)
    
    set_learning_rate(config_path, args.lr)


if __name__ == "__main__":
    main()
