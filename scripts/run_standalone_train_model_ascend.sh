#!/bin/bash

if [ $# != 1 ] && [ $# != 4 ]
then
    echo "Usage: sh run_distribute_train.sh [CONFIG_PATH] [DATA_PATH] [HYP_PATH] [DEVICE_ID]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

if [ $# == 1 ]
then
  export DEVICE_ID=$1
  CONFIG_PATH=$"./config/network/yolov5s.yaml"
  DATA_PATH=$"./config/data/coco.yaml"
  HYP_PATH=$"./config/data/hyp.scratch-low.yaml"
fi

if [ $# == 4 ]
then
  CONFIG_PATH=$(get_real_path $1)
  DATA_PATH=$(get_real_path $2)
  HYP_PATH=$(get_real_path $3)
  export DEVICE_ID=$4
fi

echo "CONFIG_PATH: "$CONFIG_PATH
echo "DATA_PATH: "$DATA_PATH
echo "HYP_PATH: "$HYP_PATH


export RANK_ID=0
rm -rf ./train$DEVICE_ID
mkdir ./train$DEVICE_ID
cp ../*.py ./train$DEVICE_ID
cp -r ../config ./train$DEVICE_ID
cp -r ../src ./train$DEVICE_ID
mkdir ./train$DEVICE_ID/scripts
cp -r ../scripts/*.sh ./train$DEVICE_ID/scripts/
cd ./train$DEVICE_ID || exit
env > env.log

python train_model.py \
        --ms_strategy="StaticShape" \
        --ms_amp_level="O0" \
        --ms_loss_scaler="static" \
        --ms_loss_scaler_value=1024 \
        --ms_optim_loss_scale=1 \
        --ms_grad_sens=1024 \
        --overflow_still_update=True \
        --clip_grad=False \
        --optimizer="momentum" \
        --cfg=$CONFIG_PATH \
        --data=$DATA_PATH \
        --hyp=$HYP_PATH \
        --device_target=Ascend \
        --profiler=True \
        --accumulate=False \
        --epochs=300 \
        --recompute=False \
        --recompute_layers=5 \
        --batch_size=32  > log.txt 2>&1 &
cd ..

