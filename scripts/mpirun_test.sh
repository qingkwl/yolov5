#!/bin/bash

###==========================================================================
### Usage: bash run_standalone_train_ascend.sh [OPTIONS]...
### Description:
###     Run test for YOLOv5 model.
###     Note that long option should use '--option=value' format, short option should use '-o value'
### Options:
###   -d  --data                path to dataset config yaml file
###   -D, --device              device id for standalone train, or start device id for distribute train
###   -H, --help                print this help message
###   -h  --hyp                 path to hyper-parameter config yaml file
###   -c, --config              path to model config file
###   -n, --number              number of devices
###   -w, --weights              path to checkpoint weights
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
echo "WEIGHTS: $WEIGHTS"

cur_dir=$(pwd)
build_third_party_files "$cur_dir" "../third_party"

eval_exp=$(get_work_dir "eval_exp")
eval_exp=$(realpath "${eval_exp}")
echo "Make directory ${eval_exp}"
copy_files_to "$eval_exp"
cd "${eval_exp}" || exit
env > env.log
[ "$DEVICE_NUM" -gt 1 ] && distributed="True" || distributed="False"

mpirun --allow-run-as-root -n "$DEVICE_NUM" \
       --output-filename log_output \
       --merge-stderr-to-stdout \
python test.py \
        --is_distributed="$distributed" \
        --weights="$WEIGHTS" \
        --cfg="$CONFIG_PATH" \
        --data="$DATA_PATH" \
        --hyp="$HYP_PATH" \
        --device_target=Ascend \
        --img_size=640 \
        --conf=0.001 \
        --iou_thres=0.65 \
        --project="${eval_exp}/eval_results" \
        --batch_size=32 > log.txt 2>&1 &
cd ..
