#!/bin/bash

###==========================================================================
### Usage: bash run_distribute_train_ascend.sh [OPTIONS]...
### Description:
###     Run distributed train for YOLOv5 model.
###     Note that long option should use '--option=value' format, short option should use '-o value'
### Options:
###   -d  --data                path to dataset config yaml file
###   -D, --device              device id for standalone train, or start device id for distribute train
###   -H, --help                print this help message
###   -h  --hyp                 path to hyper-parameter config yaml file
###   -c, --config              path to model config file
###   -r, --rank_table_file     path to rank table config file
###   -n, --number              number of devices
### Example:
### 1. Train models with config. Configs in [] are optional.
###     bash run_distribute_train_ascend.sh --rank_table_file=hccl.json [-c config.yaml -d coco.yaml --hyp=hyp.config.yaml]
###==========================================================================

source common.sh
parse_args "$@"
get_default_config

echo "CONFIG PATH: $CONFIG_PATH"
echo "DATA PATH: $DATA_PATH"
echo "HYP PATH: $HYP_PATH"
echo "RANK TABLE FILE: $RANK_TABLE_FILE"
echo "START DEVICE ID: $DEVICE_ID"

if [ ! -f "$RANK_TABLE_FILE" ]
then
    echo "error: RANK_TABLE_FILE=$RANK_TABLE_FILE is not a file"
exit 1
fi

export DEVICE_NUM=$DEVICE_NUM
export RANK_SIZE=$DEVICE_NUM
export OPENBLAS_NUM_THREADS=1
export RANK_TABLE_FILE=$RANK_TABLE_FILE
export MINDSPORE_HCCL_CONFIG_PATH=$RANK_TABLE_FILE

cpus=$(grep -c "processor" /proc/cpuinfo)
avg=$((cpus / RANK_SIZE))
gap=$((avg - 1))
train_exp=$(get_work_dir "train_exp")
train_exp=$(realpath "${train_exp}")
echo "Make directory ${train_exp}"
mkdir "${train_exp}"
start_device_id="$DEVICE_ID"
cur_dir=$(pwd)
build_third_party_files "$cur_dir" "../third_party"
for((i=0; i < DEVICE_NUM; i++))
do
    start=$((i * avg))
    end=$((start + gap))
    cmdopt="$start-$end"

    export DEVICE_ID=$((i + start_device_id))
    export RANK_ID=$i
    sub_dir="${train_exp}/train_parallel${i}"
    copy_files_to "$sub_dir"
    cd "${sub_dir}" || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env > env.log

    taskset -c $cmdopt python train.py \
        --ms_strategy="StaticShape" \
        --ms_amp_level="O0" \
        --ms_loss_scaler="static" \
        --ms_loss_scaler_value=1024 \
        --ms_optim_loss_scale=1 \
        --ms_grad_sens=1024 \
        --overflow_still_update=True \
        --clip_grad=False \
        --sync_bn=False \
        --optimizer="momentum" \
        --cfg=$CONFIG_PATH \
        --data=$DATA_PATH \
        --hyp=$HYP_PATH \
        --device_target=Ascend \
        --is_distributed=True \
        --epochs=300 \
        --recompute=False \
        --recompute_layers=5 \
        --batch_size=256 \
        --img_size=640 \
        --conf_thres=0.001 \
        --iou_thres=0.65 \
        --run_eval=True \
        --eval_epoch_interval=10 \
        --project="${train_exp}/eval_results" > log.txt 2>&1 &
    cd "${cur_dir}" || exit
done
