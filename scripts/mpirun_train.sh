#!/bin/bash

###==========================================================================
### Usage: bash run_standalone_train_ascend.sh [OPTIONS]...
### Description:
###     Run train for YOLOv5 model.
###     Note that long option should use '--option=value' format, short option should use '-o value'
### Options:
###   -d  --data                path to dataset config yaml file
###   -D, --device              device id for standalone train, or start device id for distribute train
###   -H, --help                print this help message
###   -h  --hyp                 path to hyper-parameter config yaml file
###   -c, --config              path to model config file
###   -n, --number              number of devices
### Example:
### 1. train models with config. Configs in [] are optional.
###     bash mpirun_train.sh [-c config.yaml -d coco.yaml --hyp=hyp.config.yaml]
###==========================================================================

source common.sh
parse_args "$@"
get_default_config

export DEVICE_ID=$DEVICE_ID
export DEVICE_NUM=$DEVICE_NUM
echo "CONFIG PATH: $CONFIG_PATH"
echo "DATA PATH: $DATA_PATH"
echo "HYP PATH: $HYP_PATH"
echo "DEVICE ID: $DEVICE_ID"


cur_dir=$(pwd)
build_third_party_files "$cur_dir" "../third_party"

train_exp=$(get_work_dir "train_exp")
train_exp=$(realpath "${train_exp}")
echo "Make directory ${train_exp}"
copy_files_to "$train_exp"
cd "${train_exp}" || exit
env > env.log
[ "$DEVICE_NUM" -gt 1 ] && distributed="True" || distributed="False"

mpirun --allow-run-as-root -n "$DEVICE_NUM" \
       --output-filename log_output \
       --merge-stderr-to-stdout \
python train.py \
        --distributed_train="$distributed" \
        --distributed_eval="$distributed" \
        --clip_grad=False \
        --optimizer="momentum" \
        --cfg="$CONFIG_PATH" \
        --data="$DATA_PATH" \
        --hyp="$HYP_PATH" \
        --device_target=Ascend \
        --profiler=False \
        --accumulate=False \
        --epochs=300 \
        --iou_thres=0.65 \
        --batch_size=32  > log.txt 2>&1 &
cd ..
