#!/bin/bash

# 定义三组不同的配置

# 第一次运行
# echo "c 1..."
# python3 sample_dpo.py --output_subdir "samp_1" || exit 1;

# 第二次运行 
echo "Applying config 2..."
SAMPLING_TEMPERATURE=1.3 SAMPLING_TOP_P=0.9 SAMPLING_TOP_K=40 \
python3 sample_dpo.py --output_subdir "samp_2" || exit 1;

# 第三次运行
echo "Applying config 3..."
SAMPLING_TEMPERATURE=1.5 SAMPLING_TOP_P=0.85 SAMPLING_TOP_K=50 \
python3 sample_dpo.py --output_subdir "samp_3" || exit 1;

# 第四次运行
echo "Applying config 4..."
SAMPLING_TEMPERATURE=1.1 SAMPLING_TOP_P=0.95 SAMPLING_TOP_K=30 \
python3 sample_dpo.py --output_subdir "samp_4" || exit 1;

echo "All configurations completed successfully"
exit 0


