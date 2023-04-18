# Contents

<details>
<summary>Click to unfold/fold</summary>

- [Contents](#contents)
- [YOLOv5 Description](#yolov5-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
  - [Dataset Download](#dataset-download)
  - [Dataset Structure](#dataset-structure)
  - [Dataset Conversion](#dataset-conversion)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
  - [Script and Sample Code](#script-and-sample-code)
  - [Script Parameters](#script-parameters)
  - [Training Process](#training-process)
    - [Training](#training)
    - [Distributed Training](#distributed-training)
  - [Evaluation Process](#evaluation-process)
    - [Evaluation](#evaluation)
  - [Infer Process](#infer-process)
    - [Infer](#infer)
- [Model Description](#model-description)
  - [Performance](#performance)

</details>

# [YOLOv5 Description](#contents)

Published in April 2020 by [Ultralytics](https://ultralytics.com/), YOLOv5 achieved state-of-the-art performance on the COCO dataset for object detection.
It is an important improvement of YoloV3, the implementation of a new architecture in the **Backbone** and
the modifications in the **Neck** have improved the **mAP**(mean Average Precision) by **10%** and 
the number of **FPS**(Frame per Second) by **12%**.

Repository of official implementation by `PyTorch`：https://github.com/ultralytics/yolov5

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

`YOLOv5` is trained on `COCO` dataset with labels of `YOLO` format.

## Dataset Download

Dataset:

- Raw data
  - Train set: http://images.cocodataset.org/zips/train2017.zip
  - Validation set: http://images.cocodataset.org/zips/val2017.zip
  - Test set: http://images.cocodataset.org/zips/test2017.zip
- YOLO format labels `coco2017labels`: https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip
- YOLO format segmentation labels `coco2017labels-segments`: https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels-segments.zip

Download the raw images data, and the labels files according to target model:

| Model   | Label                   |
|---------|-------------------------|
| YOLOv5n | coco2017labels          |
| YOLOv5s | coco2017labels          |
| YOLOv5m | coco2017labels-segments |
| YOLOv5l | coco2017labels-segments |
| YOLOv5x | coco2017labels-segments |

## Dataset Structure

After downloading the dataset and labels, you should put them in correct position
as the following text shows. The `images` folder saves the images and `labels`
folder saves the corresponding labels. The text files like `train2017.txt` saves
the image paths of the corresponding subset of dataset.

```txt
YOLO
├── images
|   ├── train2017
|   ├── val2017
|   └── test2017
├── labels
|   ├── train2017
|   ├── val2017
├── images
|   ├── train2017
|   ├── val2017
|   └── test2017
├── train2017.txt
├── val2017.txt
└── test2017.txt
```

## Dataset Conversion

If you want to use customized data with `COCO` or `labelme` format, you can use conversion script to convert them to `YOLO` format.

Conversion steps:

1. Change directory to `config/data_conversion`. The names of the files in this folder stand for configs of corresponding dataset.
2. Modify the config files of the original format and the conversion target format. Change the path in config files.
3. After edit of config files, run `convert_data.py` script. For example, `python convert_data.py coco yolo` means convert dataset from coco format to yolo.

# [Quick Start](#contents)

<details>
<summary>Installation</summary>

Follow the tutorial in `MindSpore` [official website](https://www.mindspore.cn/install) to install `mindspore`.
Then use the following command to install other required packages:

```shell
pip install -r requirements.txt
```

</details>

<details>
<summary>Training</summary>

You can use the following command to train on a single device:

```bash
# Run training example(1p) on Ascend/GPU by python command
python train.py \
    --ms_strategy="StaticShape" \
    --overflow_still_update=True \
    --optimizer="momentum" \
    --cfg="../config/network/yolov5s.yaml" \
    --data="../config/data/coco.yaml" \
    --hyp="../config/data/hyp.scratch-low.yaml" \
    --device_target=Ascend \
    --epochs=300 \
    --batch_size=32  > log.txt 2>&1 &
```

Or you can use `shell` scripts. The scripts support training on single or multiple devices.
The command are in the following:

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

You could pass `--help` or `-H` to shell script to see usage in detail.

</details>

<details>
<summary>Evaluation</summary>

You can use the following command to evaluate a model:

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

The `rect` switch can increase mAP of evaluation result.
The results in official repository is evaluated with this switch on.
Please note this difference when you compare evaluation results of two repositories.

Or you can also use `shell` scripts to do evaluation:

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

</details>

# [Script Description](#contents)

## [Script and Sample Code](#contents)

<details>
    <summary>Click to unfold/fold</summary>

```text
yolov5
├── README.md                                      // descriptions about yolov5
├── README_CN.md                                   // Chinese descriptions about yolov5
├── __init__.py
├── config
│   ├── args.py                                    // get config parameters from command line
│   ├── data
│   │   ├── coco.yaml                              // configs about dataset
│   │   ├── hyp.scratch-high.yaml                   // configs about hyper-parameters
│   │   ├── hyp.scratch-low.yaml
│   │   └── hyp.scratch-med.yaml
│   ├── data_conversion
│   │   ├── coco.yaml                              // config of coco format dataset 
│   │   ├── labelme.yaml                           // config of labelme format dataset
│   │   └── yolo.yaml                              // config of yolo format dataset
│   └── network                                    // configs of model architecture
│       ├── yolov5l.yaml
│       ├── yolov5m.yaml
│       ├── yolov5n.yaml
│       ├── yolov5s.yaml
│       └── yolov5x.yaml
├── convert_data.py                                // convert dataset format
├── deploy                                         // code for inference
│   ├── __init__.py
│   └── infer_engine
│       ├── __init__.py
│       ├── lite.py                                // code for inference with MindSporeLite
│       ├── mindx.py                               // code for inference with mindx
│       └── model_base.py
├── export.py
├── preprocess.py
├── scripts
│   ├── common.sh                                  // common functions used in shell scripts 
│   ├── get_coco.sh
│   ├── hccl_tools.py                              // generate rank table files for distributed training or evaluation
│   ├── mpirun_test.sh                             // launch evaluation with OpenMPI
│   ├── mpirun_train.sh                            // launch training with OpenMPI
│   ├── run_distribute_test_ascend.sh              // launch distributed evaluation(8p) on Ascend
│   ├── run_distribute_train_ascend.sh             // launch distributed training(8p) on Ascend
│   ├── run_standalone_test_ascend.sh              // launch 1p evaluation on Ascend
│   └── run_standalone_train_ascend.sh             // launch 1p training on Ascend
├── src
│   ├── __init__.py
│   ├── all_finite.py
│   ├── augmentations.py                           // data augmentations
│   ├── autoanchor.py
│   ├── boost.py
│   ├── callback.py
│   ├── checkpoint_fuse.py
│   ├── coco_visual.py
│   ├── data                                       // code for dataset format conversion
│   │   ├── __init__.py
│   │   ├── base.py                                // base class for data conversion
│   │   ├── coco.py                                // transfer dataset with coco format to others
│   │   ├── labelme.py                             // transfer dataset with labelme format to others
│   │   └── yolo.py                                // transfer dataset with yolo format to others
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

</details>

## [Script Parameters](#contents)

```text
Major parameters in train.py are:

optional arguments:
  --ms_strategy           Training strategy. Default: "StaticShape"
  --is_distributed        Distribute training or not. Default: False
  --device_target         Device where the code will be executed. Default: "Ascend"
  --cfg                   Model architecture yaml config file path. Default: "./config/network/yolov5s.yaml"
  --data                  Dataset yaml config file path. Default: "./config/data/data.yaml"
  --hyp                   Hyper-parameters yaml config file path. Default: "./config/data/hyp.scratch-low.yaml"
  --epochs                Training epochs. Default: 300
  --batch_size            Batch size per device. Default: 32
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
    --overflow_still_update=True \
    --optimizer="momentum" \
    --cfg="../config/network/yolov5s.yaml" \
    --data="../config/data/coco.yaml" \
    --hyp="../config/data/hyp.scratch-low.yaml" \
    --device_target=Ascend \
    --epochs=300 \
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

You can also use OpenMPI to run distributed training. You should follow the [official tutorial](https://www.mindspore.cn/tutorials/experts/en/r2.0.0-alpha/parallel/train_gpu.html#configuring-distributed-environment)
to configure OpenMPI environment，then execute the following command：

```bash
bash mpirun_train.sh -c ../config/network/yolov5s.yaml -d ../config/data/coco.yaml \
     -h ../config/data/hyp.scratch-low.yaml
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
  --iou_thres=0.65 \
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

You can also use OpenMPI to run distributed test. You should follow the [official tutorial](https://www.mindspore.cn/tutorials/experts/en/r2.0.0-alpha/parallel/train_gpu.html#configuring-distributed-environment)
to configure OpenMPI environment，then execute the following command：

```bash
bash mpirun_test.sh --w path/to/weights.ckpt -c ../config/network/yolov5s.yaml -d ../config/data/coco.yaml \
     -h ../config/data/hyp.scratch-low.yaml
```

## [Infer Process](#contents)

### [Infer](#contents)

The model of `ckpt` format can be transformed to `om` format by `atc` tool
for doing inference on inference server. The following are steps:

1. Export model with `AIR` format：
  `python export.py --weights /path/to/model.ckpt --file_format AIR`;
2. Transform model with `AIR` format to `om` format by `atc` tool：
  `/usr/local/Ascend/latest/atc/bin/atc --model=yolov5s.om --framework=1 --output=./yolov5s --input_format=NCHW --input_shape="Inputs:1,3,640,640" --soc_version=Ascend310`,
  the `--soc_version` option can be got by `npu-smi info` command. Supported option choices are `Ascend310`，`Ascend310P3`;
3. Infer by executing `infer.py` script：`python infer.py --batch_size 1 --om yolov5s.om`

# [Model Description](#contents)

## [Performance](#contents)

| Model   | size<br><sup>(pixels) | mAP<sup>val<br>50-95<br>rect=True | mAP<sup>val<br>50<br>rect=True | mAP<sup>val<br>50-95<br>rect=False | mAP<sup>val<br>50<br>rect=False | Epoch Time(s) | Throughput<br>(images/s) |
|---------|-----------------------|-----------------------------------|--------------------------------|------------------------------------|---------------------------------|---------------|--------------------------|
| YOLOv5n | 640                   |                                   |                                |                                    |                                 | 66            | 224.00                   |
| YOLOv5s | 640                   | 0.375                             | 0.572                          | 0.373                              | 0.57                            | 79            | 187.14                   |
| YOLOv5m | 640                   | 0.453                             | 0.637                          | 0.451                              | 0.637                           | 133           | 111.16                   |
| YOLOv5l | 640                   | 0.489                             | 0.675                          | 0.486                              | 0.671                           | 163           | 90.70                    |
| YOLOv5x | 640                   |                                   |                                |                                    |                                 | 221           | 66.90                    |

<details>
<summary>Note</summary>

- All models are trained to 300 epochs with default settings. Nano and Small models use hyp.scratch-low.yaml hyper-parameters, all others use hyp.scratch-high.yaml.
- The following are settings used for different models:

```bash
--data coco.yaml --epochs 300 --weights '' --cfg yolov5n.yaml  --batch-size 
                                                 yolov5s.yaml                32
                                                 yolov5m.yaml                24
                                                 yolov5l.yaml                24
                                                 yolov5x.yaml                    
```

- The result of `Epoch Time` is evaluated on 8 Ascend 910A with batch_size 32 per device.
- The result of `Throughput` is of single Ascend 910A device.
- **mAP<sup>val</sup>** values are for single-model single-scale on [COCO val2017](http://cocodataset.org) dataset.<br>The key configs are `--img_size 640 --conf_thres 0.001 --iou_thres 0.65`

</details>
