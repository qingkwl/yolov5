#!/bin/bash

###==========================================================================
### Usage: bash run_distribute_test_ascend.sh [OPTIONS]...
### Description:
###     Run distributed test for YOLOv5 model.
###     Note that long option should use '--option=value' format, short option should use '-o value'
### Options:
###   -d  --data                path to dataset config yaml file
###   -D, --device              device id for standalone train, or start device id for distribute train
###   -H, --help                print this help message
###   -h  --hyp                 path to hyper-parameter config yaml file
###   -c, --config              path to model config file
###   -w, --weights             path to checkpoint weights
###   -r, --rank_table_file     path to rank table config file
###   -n, --number              number of devices
### Example:
### 1. Test checkpoint with config. Configs in [] are optional.
###     bash run_distribute_test_ascend.sh -w weights.ckpt --rank_table_file=hccl.json [-c config.yaml -d coco.yaml --hyp=hyp.config.yaml]
###==========================================================================

source common.sh
parse_args "$@"
get_default_config

echo "WEIGHTS: $WEIGHTS"
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

if [ -z "$WEIGHTS" ]; then
    echo "ERROR: weights argument path is empty, which is required."
    exit 1
fi

export DEVICE_NUM=$DEVICE_NUM
export RANK_SIZE=$DEVICE_NUM
export RANK_TABLE_FILE=$RANK_TABLE_FILE
export MINDSPORE_HCCL_CONFIG_PATH=$RANK_TABLE_FILE

cpus=$(grep -c "processor" /proc/cpuinfo)
avg=$((cpus / RANK_SIZE))
gap=$((avg - 1))
eval_exp=$(get_work_dir "eval_exp")
eval_exp=$(realpath "${eval_exp}")
echo "Make directory ${eval_exp}"
mkdir "${eval_exp}"
start_device_id="$DEVICE_ID"
cur_dir=$(pwd)
build_third_party_files "$cur_dir" "../third_party"
for((i=0; i < DEVICE_NUM; i++))
do
    start=$((i * avg))
    end=$((start + gap))
    cmdopt="${start}-${end}"
    export DEVICE_ID=$((i + start_device_id))
    export RANK_ID=$i
    sub_dir="${eval_exp}/eval_parallel${i}"
    copy_files_to "$sub_dir"
    cd "${sub_dir}" || exit
    echo "start testing for rank $RANK_ID, device $DEVICE_ID"
    env > env.log
    taskset -c $cmdopt python val.py \
        --weights=$WEIGHTS \
        --cfg=$CONFIG_PATH \
        --data=$DATA_PATH \
        --hyp=$HYP_PATH \
        --device_target=Ascend \
        --distributed_eval=True \
        --img_size=640 \
        --conf=0.001 \
        --iou_thres=0.65 \
        --rect=False \
        --project="${eval_exp}/eval_results" \
        --batch_size=32 > log.txt 2>&1 &
    cd "${cur_dir}" || exit
done
