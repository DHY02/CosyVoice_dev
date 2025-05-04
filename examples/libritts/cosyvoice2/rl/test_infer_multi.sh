#!/bin/bash
# 批量对checkpoint模型进行推理
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(dirname "$(dirname "$(dirname "$(dirname "$script_dir")")")")"
echo $root_dir
cosyvoice2_dir="${root_dir}/examples/libritts/cosyvoice2"
data_dir="${cosyvoice2_dir}/data"
pretrained_model_dir="${root_dir}/pretrained_models/CosyVoice2-0.5B"
checkpoint_model_dir="${cosyvoice2_dir}/exp/cosyvoice2/llm/torch_ddp/casia_dpo_0.01"

# 创建包含1到9的奇数epoch的模型路径数组
llm_model_paths=()
for num_epoch in {1..9..2}; do
    epoch_model_name="epoch_${num_epoch}_whole.pt"
    model_path="${checkpoint_model_dir}/${epoch_model_name}"
    llm_model_paths+=("${model_path}")
done

# 打印所有要测试的模型路径
echo "将要测试的模型路径:"
printf '%s\n' "${llm_model_paths[@]}"

for model_path in "${llm_model_paths[@]}"; do
    echo "正在测试模型: ${model_path}"
    # 检查模型文件是否存在
    if [ ! -f "${model_path}" ]; then
        echo "错误: 模型文件不存在: ${model_path}" 
        continue
    fi
    
    # 设置结果目录名称
    if [[ "${model_path}" == *"epoch_"* ]]; then
        epoch_num=$(echo "${model_path}" | grep -oP 'epoch_\K\d+')
        result_dir_name="test_dpo_epoch_${epoch_num}"
    else
        echo "不接受的模型名称"
        exit 1;
    fi
    
    bash "${script_dir}/test_infer.sh" "${root_dir}" "${model_path}" "${result_dir_name}"
done
