#!/bin/bash
# train + evaluate

stage=1
stop_stage=1

# 训练stage
tr_stage=3
tr_stop_stage=5

# 评估stage
te_stage=1
te_stop_stage=3


method="emo-dpo"

train_corpus_name="esd"
test_corpus_name="esd"
framework="deepspeed"

# grpo/dpo超参数
beta=0.04
clip=0.2
lr=1e-5
grpo_datasets="receive samp_1 samp_2 samp_3 samp_4"

# checkpoint to start training
checkpoint=""

# the epoch that emo-grpo or emo-dpo start
emo_train_start_epoch=2

# evaluate args
start_epoch=0
# stop_epoch必须由命令行参数提供
epoch_interval=1

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stop_epoch)
            stop_epoch="$2"
            shift 2
            ;;
        --method)
            method="$2"
            shift 2
            ;;
        --stage)
            stage="$2"
            shift 2
            ;;
        --stop_stage)
            stop_stage="$2"
            shift 2
            ;;
        --tr_stage)
            tr_stage="$2"
            shift 2
            ;;
        --tr_stop_stage)
            tr_stop_stage="$2"
            shift 2
            ;;
        --te_stage)
            te_stage="$2"
            shift 2
            ;;
        --te_stop_stage)
            te_stop_stage="$2"
            shift 2
            ;;
        --beta)
            beta="$2"
            shift 2
            ;;
        --clip)
            clip="$2"
            shift 2
            ;;
        --lr)
            lr="$2"
            shift 2
            ;;
        --emo_train_start_epoch)
            emo_train_start_epoch="$2"
            shift 2
            ;;
        --epoch_interval)
            epoch_interval="$2"
            shift 2
            ;;
        --checkpoint)
            checkpoint="$2"
            processed=true
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# 检查stop_epoch是否已提供
if [ -z "${stop_epoch}" ]; then
    echo "错误：必须提供评估的结束epoch值--stop_epoch参数"
    echo "用法：bash run.sh --stop_epoch <value>"
    exit 1
fi

exp_name="${train_corpus_name}_${method}|b${beta}c${clip}s${emo_train_start_epoch}l${lr}"

if [ "${framework}" == "deepspeed" ]; then
    python tools/set_ds_lr.py --lr ${lr} --config "conf/ds_stage2.json"
else
    echo "当前不支持torch_ddp自动设置lr"
    exit 1
fi


# train
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "==== stage 1: Train ===="
    pretrained_model_dir=/root/autodl-tmp/CosyVoice_dev/pretrained_models/CosyVoice2-0.5B
    init_model_dir=/root/autodl-tmp/CosyVoice_dev/pretrained_models/CosyVoice2-0.5B-bk
    cp ${init_model_dir}/llm.pt ${pretrained_model_dir}/llm.pt || exit 1
    # 传入参数
    bash "run-${method}.sh" \
        --corpus ${train_corpus_name} \
        --stage ${tr_stage} \
        --stop_stage ${tr_stop_stage} \
        --method ${method} \
        --beta ${beta} \
        --clip ${clip} \
        --lr ${lr} \
        --grpo_datasets "${grpo_datasets}" \
        --train_engine "${framework}" \
        --start_epoch "${emo_train_start_epoch}" \
        --dpo_ref_model "${init_model_dir}/llm.pt" \
        --checkpoint "${checkpoint}" || {
        echo "错误：训练阶段执行失败！"
        exit 1
    }
    
fi

# evaluate
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "==== stage 2: Evaluate ===="
    cd "/root/autodl-tmp/CosyVoice_dev/examples/libritts/cosyvoice2/rl"
    # 传入参数
    bash run.sh \
        --train_corpus ${train_corpus_name} \
        --test_corpus ${test_corpus_name} \
        --method ${method} \
        --beta ${beta} \
        --clip ${clip} \
        --stage ${te_stage} \
        --stop_stage ${te_stop_stage} \
        --start_epoch ${start_epoch} \
        --stop_epoch ${stop_epoch} \
        --epoch_interval ${epoch_interval} \
        --framework ${framework} \
        --exp_name ${exp_name} || {
        echo "错误：验证阶段执行失败！"
    }
    
fi