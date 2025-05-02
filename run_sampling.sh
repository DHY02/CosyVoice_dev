#!/bin/bash

# 定义三组不同的配置

# 第一次运行
echo "Applying config 1..."
TEMPERATURE=0.85 TOP_P=0.9 TOP_K=20 python3 sample_dpo.py --output_subdir "samp_1" || exit 1;
# 第二次运行 
echo "Applying config 2..."
TEMPERATURE=1.0 TOP_P=1.0 TOP_K=25 python3 sample_dpo.py --output_subdir "samp_2" || exit 1;
# 第三次运行
echo "Applying config 3..."
TEMPERATURE=1.2 TOP_P=1.0 TOP_K=30 python3 sample_dpo.py --output_subdir "samp_3" || exit 1;

echo "All configurations completed successfully"
exit 0


