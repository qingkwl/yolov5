# Contents

- [Contents](#contents)
- [YOLOv5 Description](#YOLOv5-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
  - [Dataset Download](#Dataset-Download)
  - [Dataset Conversion](#Dataset-Conversion)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
        - [Distributed Training](#distributed-training)
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
- [Model Description](#model-description)
- [Performance](#performance)  

- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)


# [YOLOv5 Description](#contents)

Published in April 2020, YOLOv5 achieved state-of-the-art performance on the COCO dataset for object detection. 
It is an important improvement of YoloV3, the implementation of a new architecture in the **Backbone** and 
the modifications in the **Neck** have improved the **mAP**(mean Average Precision) by **10%** and 
the number of **FPS**(Frame per Second) by **12%**.

[code](https://github.com/ultralytics/yolov5)


# [Model Architecture](#contents)

The YOLOv5 network is mainly composed of CSP and Focus as a backbone, spatial pyramid pooling(SPP) additional module, 
PANet path-aggregation neck and YOLOv3 head. [CSP](https://arxiv.org/abs/1911.11929) is a novel backbone 
that can enhance the learning capability of CNN. 
The [spatial pyramid pooling](https://arxiv.org/abs/1406.4729) block is added over CSP to increase the receptive field 
and separate out the most significant context features. 
Instead of Feature pyramid networks (FPN) for object detection used in YOLOv3, the PANet is used as the method 
for parameter aggregation for different detector levels. 
To be more specific, CSPDarknet53 contains 5 CSP modules which use the convolution **C** with kernel size k=3x3, 
stride s = 2x2; Within the PANet and SPP, 1x1, 5x5, 9x9, 13x13 max poolings are applied.


# [Dataset](#contents)

## Dataset Download
Dataset: 
- Raw data
  - [Train set](http://images.cocodataset.org/zips/train2017.zip)
  - [Validation set](http://images.cocodataset.org/zips/val2017.zip)
  - [Test set](http://images.cocodataset.org/zips/test2017.zip)
- YOLO format labels [coco2017labels](https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip)
- YOLO format segmentation labels [coco2017labels-segments](https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels-segments.zip)

Download the raw images data, then download the labels files according to target model:
- YOLOv5n：YOLO format labels
- YOLOv5s：YOLO format labels
- YOLOv5m：YOLO format segmentation labels
- YOLOv5l：YOLO format segmentation labels
- YOLOv5x：YOLO format segmentation labels

## Dataset Conversion
If you want to use customized data with coco or labelme format, you can use conversion script to convert them to YOLO format.

Conversion steps:
1. Change directory to `config/data_conversion`. The names of the files in this folder stand for configs of corresponding dataset.
2. Modify the config files of the original format and the conversion target format.
3. Afer edit of config files, run `convert_data.py` script. For example, `python convert_data.py coco yolo` means convert dataset from coco format to yolo.

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

```bash
# Run training example(1p) on Ascend/GPU by python command
python train.py \
    --ms_strategy="StaticShape" \
    --ms_amp_level="O0" \
    --ms_loss_scaler="static" \
    --ms_loss_scaler_value=1024 \
    --ms_optim_loss_scale=1 \
    --ms_grad_sens=1024 \
    --overflow_still_update=True \
    --clip_grad=False \
    --optimizer="momentum" \
    --cfg="../config/network/yolov5s.yaml" \
    --data="../config/data/coco.yaml" \
    --hyp="../config/data/hyp.scratch-low.yaml" \
    --device_target=Ascend \
    --profiler=False \
    --accumulate=False \
    --epochs=300 \
    --recompute=False \
    --recompute_layers=5 \
    --batch_size=32  > log.txt 2>&1 &
```

```bash
# Run 1p by shell script, please change `device_target` in config file to run on Ascend/GPU, and change `T_max`, `max_epoch`, `warmup_epochs` refer to contents of notes
bash run_standalone_train_ascend.sh -c ../config/network/yolov5s.yaml -d ../config/data/coco.yaml \
     -h ../config/data/hyp.scratch-low.yaml

# For Ascend device, distributed training example(8p) by shell script
bash run_distribute_train_ascend.sh -c ../config/network/yolov5s.yaml -d ../config/data/coco.yaml \
     -h ../config/data/hyp.scratch-low.yaml -r hccl_8p_xx.json

# For GPU device, distributed training example(8p) by shell script
bash run_distribute_train_gpu.sh ../config/network/yolov5s.yaml ../config/data/coco.yaml \
     ../config/data/hyp.scratch-low.yaml
```

To see usage information, you could pass `--help` or `-H` to shell script.

```bash
# Run evaluation on Ascend/GPU by python command
python test.py \
  --weights="path/to/weights.ckpt" \
  --cfg="../config/network/yolov5s.yaml" \
  --data="../config/data/coco.yaml" \
  --hyp="../config/data/hyp.scratch-low.yaml" \
  --device_target=Ascend \
  --img_size=640 \
  --conf=0.001 \
  --rect=False \
  --iou_thres=0.60 \
  --batch_size=32 > log.txt 2>&1 &
```

```bash
# Run distributed evaluation by shell script
bash run_distribute_test_ascend.sh -w path/to/weights.ckpt -c ../config/network/yolov5s.yaml -d ../config/data/coco.yaml \
     -h ../config/data/hyp.scratch-low.yaml -r hccl_8p_xx.json

# Run standalone evaluation by shell script
bash run_standalone_test_ascend.sh -w path/to/weights.ckpt -c ../config/network/yolov5s.yaml -d ../config/data/coco.yaml \
     -h ../config/data/hyp.scratch-low.yaml
```

The corresponding config files are in `config` folder. The `coco.yaml` in `config/data` folder is about dataset configs. 
The `hyp.scratch-low.yaml` are hyperparameters settings. The `yolov5s.yaml` saves model architecture configs.

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
yolov5
├── README.md                                      // descriptions about yolov5
├── __init__.py
├── config
│   ├── args.py                                    // get config parameters from command line
│   ├── data
│   │   ├── coco.yaml                              // configs about dataset
│   │   └── hyp.scratch-low.yaml                   // configs about hyperparameters
│   ├── data_conversion
│   │   ├── coco.yaml                              // config of coco format dataset 
│   │   ├── labelme.yaml                           // config of labelme format dataset
│   │   └── yolo.yaml                              // config of yolo format dataset
│   └── network
│       └── yolov5s.yaml                           // configs about model architecture
├── convert_data.py                                // convert dataset format
├── export.py
├── preprocess.py
├── scripts
│   ├── common.sh                                  // common functions used in shell scripts 
│   ├── get_coco.sh
│   ├── hccl_tools.py                              // generate rank table files for distributed training or evaluation
│   ├── run_distribute_test_ascend.sh              // launch distributed evaluation(8p) on Ascend
│   ├── run_distribute_train_ascend.sh             // launch distributed training(8p) on Ascend
│   ├── run_distribute_train_gpu.sh                // launch distributed training(8p) on GPU
│   ├── run_distribute_train_thor_ascend.sh
│   ├── run_standalone_test_ascend.sh              // launch 1p evaluation on Ascend
│   ├── run_standalone_test_gpu.sh                 // launch 1p evaluation on GPU
│   ├── run_standalone_train_ascend.sh             // launch 1p training on Ascend
│   ├── run_standalone_train_gpu.sh                // launch 1p training on GPU
│   └── run_standalone_train_model_ascend.sh
├── src
│   ├── __init__.py
│   ├── all_finite.py
│   ├── augmentations.py                           // data augmentations
│   ├── autoanchor.py
│   ├── boost.py
│   ├── callback.py
│   ├── checkpoint_fuse.py
│   ├── dataset.py                                 // create dataset
│   ├── general.py                                 // general functions used in other scripts
│   ├── loss_scale.py
│   ├── metrics.py
│   ├── modelarts.py
│   ├── ms2pt.py                                   // transfer weights from MindSpore to PyTorch
│   ├── network
│   │   ├── __init__.py
│   │   ├── common.py                              // common code for building network
│   │   ├── loss.py                                // loss
│   │   └── yolo.py                                // YOLOv5 network
│   ├── optimizer.py                               // optimizer
│   ├── plots.py
│   └── pt2ms.py                                   // transfer weights from PyTorch to MindSpore
├── test.py                                        // script for evaluation
├── third_party                                    // third-party code
│   ├── __init__.py
│   ├── fast_coco                                  // faster coco mAP computation
│   │   ├── __init__.py
│   │   ├── build.sh
│   │   ├── cocoeval
│   │   │   ├── cocoeval.cpp
│   │   │   └── cocoeval.h
│   │   ├── fast_coco_eval_api.py
│   │   └── setup.py
│   ├── fast_nms                                   // faster nms computation
│   │   ├── __init__.py
│   │   ├── build.sh
│   │   ├── nms.pyx
│   │   └── setup.py
│   └── yolo2coco                                  // yolo data format to coco format converter
│       ├── __init__.py
│       └── yolo2coco.py
└── train.py                                       // script for training
```


## [Script Parameters](#contents)

```text
Major parameters in train.py are:

optional arguments:
  --ms_strategy           Training strategy. Default: "StaticShape"
  --is_distributed        Distribute training or not. Default: False
  --device_target         Device where the code will be executed. Default: "Ascend"
  --cfg                   Model architecture yaml config file path. Default: "./config/network/yolov5s.yaml"
  --data                  Dataset yaml config file path. Default: "./config/data/data.yaml"
  --hyp                   Hyperparameters yaml config file path. Default: "./config/data/hyp.scratch-low.yaml"
  --epochs                Training epochs. Default: 300
  --batch_size            Total batch size for all devices. Default: 32
  --save_checkpoint       Whether save checkpoint. Default: True
  --start_save_epoch      Epoch index after which checkpoint will be saved. Default: 1
  --save_interval         Epoch interval to save checkpoints. Default: 1
  --max_ckpt_num          Maximum number of saved checkpoints. Default: 10
  --cache_images          Whether cache images for faster training. Default: False
  --optimizer             Optimizer used for training. Default: "sgd"
  --sync_bn               Whether use SyncBatchNorm, only available in DDP mode. Default: False
  --project               Folder path to save output data. Default: "runs/train"
  --linear_lr             Whether use linear learning rate. Default: True
  --run_eval              Whether do evaluation after a training epoch. Default: True
  --eval_start_epoch      Epoch index after which model will do evaluation. Default: 200
  --eval_epoch_interval   Epoch interval to do evaluation. Default: 10
```


## [Training Process](#contents)

### Training

For Ascend device, standalone training can be started like this:

```shell
# Run training example(1p) on Ascend/GPU by python command
python train.py \
    --ms_strategy="StaticShape" \
    --ms_amp_level="O0" \
    --ms_loss_scaler="static" \
    --ms_loss_scaler_value=1024 \
    --ms_optim_loss_scale=1 \
    --ms_grad_sens=1024 \
    --overflow_still_update=True \
    --clip_grad=False \
    --optimizer="momentum" \
    --cfg="../config/network/yolov5s.yaml" \
    --data="../config/data/coco.yaml" \
    --hyp="../config/data/hyp.scratch-low.yaml" \
    --device_target=Ascend \
    --profiler=False \
    --accumulate=False \
    --epochs=300 \
    --recompute=False \
    --recompute_layers=5 \
    --batch_size=32  > log.txt 2>&1 &
```

You should fine tune the parameters when run training for custom dataset.

The python command above will run in the background.

### Distributed Training

Distributed training example(8p) by shell script:

```bash
# For Ascend device, distributed training example(8p) by shell script
bash run_distribute_train_ascend.sh -c ../config/network/yolov5s.yaml -d ../config/data/coco.yaml \
     -h ../config/data/hyp.scratch-low.yaml -r hccl_8p_xx.json

# For GPU device, distributed training example(8p) by shell script
bash run_distribute_train_gpu.sh ../config/network/yolov5s.yaml ../config/data/coco.yaml \
     ../config/data/hyp.scratch-low.yaml
```


## [Evaluation Process](#contents)

### Evaluation

Before running the command below, please check the checkpoint path used for evaluation.

```shell
# Run evaluation by python command
python test.py \
  --weights="path/to/weights.ckpt" \
  --cfg="../config/network/yolov5s.yaml" \
  --data="../config/data/coco.yaml" \
  --hyp="../config/data/hyp.scratch-low.yaml" \
  --device_target=Ascend \
  --img_size=640 \
  --conf=0.001 \
  --rect=False \
  --iou_thres=0.60 \
  --batch_size=32 > log.txt 2>&1 &
# OR
# Run evaluation(8p) by shell script
bash run_distribute_test_ascend.sh -w path/to/weights.ckpt -c ../config/network/yolov5s.yaml -d ../config/data/coco.yaml \
     -h ../config/data/hyp.scratch-low.yaml -r hccl_8p_xx.json
# OR
# Run standalone evaluation by shell script
bash run_standalone_test_ascend.sh --w path/to/weights.ckpt -c ../config/network/yolov5s.yaml -d ../config/data/coco.yaml \
     -h ../config/data/hyp.scratch-low.yaml
```

The above python command will run in the background. You can view the results through the file "log.txt".

# [Model Description](#contents)

## [Performance](#contents)

| Model   | size<br><sup>(pixels) | mAP<sup>val<br>50-95<br>rect=True | mAP<sup>val<br>50<br>rect=True | mAP<sup>val<br>50-95<br>rect=False | mAP<sup>val<br>50<br>rect=False | Epoch Time(s) |
|---------|-----------------------|-----------------------------------|--------------------------------|------------------------------------|---------------------------------|---------------|
| YOLOv5n | 640                   |                                   |                                |                                    |                                 | 66            |
| YOLOv5s | 640                   | 0.375                             | 0.572                          | 0.373                              | 0.57                            | 79            |
| YOLOv5m | 640                   | 0.452                             | 0.638                          | 0.451                              | 0.636                           | 133           |
| YOLOv5l | 640                   |                                   |                                |                                    |                                 | 163           |
| YOLOv5x | 640                   |                                   |                                |                                    |                                 | 221           |
