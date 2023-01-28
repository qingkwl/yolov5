# 目录

- [目录](#目录)
- [YOLOv5说明](#YOLOv5说明)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本和示例代码](#脚本和示例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#训练)
        - [分布式训练](#分布式训练)
    - [推理过程](#推理过程)
        - [推理](#推理)
- [模型说明](#模型说明)
- [性能](#性能)  
    - [评估性能](#评估性能)
    - [推理性能](#推理性能)


# [YOLOv5说明](#目录)


YOLOv5 于 2020 年 4 月发布，并在 COCO 数据集目标检测任务中取得了 SOTA 成绩。它是对 YOLOv3 的一个重要改进，
新提出的 **Backbone** 结构以及对于 **Neck** 的改进使得 YOLOv5 在 mAP(mean Average Precision) 上提升了 10%，
在 FPS(Frame Per Second) 上提升了 12%。

[code](https://github.com/ultralytics/yolov5)


# [模型架构](#目录)

YOLOv5 模型以添加了 SPP 模块的 CSP 模块与 Focus 模块作为 Backbone，以 PANet 中的 Path-aggregation 模块作为 Neck，
并保留了 YOLOv3 的 Head 模块。[CSP](https://arxiv.org/abs/1911.11929) 作为 Backbone 能够有效增强 CNN 的学习能力。
添加到 CSP 模块中的 [Spatial pyramid pooling](https://arxiv.org/abs/1406.4729) 模块能够增加感受野，分离最显著的上下文特征。
YOLOv4 使用 PANet 替换了 YOLOv3 中的 Feature Pyramid Networks(FPN) 来检测目标，PANet 能够聚合不同层级检测器的参数。

CSPDarknet53 包含了 5 个 CSP 模块，CSP 模块中使用了 kernel size 为 3x3, stride 为 2x2 的卷积层；
而在 PANet 和 SPP 中使用了 1x1、5x5、9x9、13x13 的 max pooling 网络层。


# [数据集](#目录)

数据集: [COCO2017](<https://cocodataset.org/#download>)

您可以使用 **COCO2017** 或者其他具有 COCO 格式的数据集来运行我们提供的脚本。但是我们建议您使用 COCO 数据集来尝试我们的模型。

# [快速入门](#目录)

按照官方网站的指导安装 MindSpore 之后，您可以使用以下命令来训练或者推理：

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

您可以输入 `--help` 或者 `-H` 来查看更多 Shell 脚本的使用方法。

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

脚本运行相关的配置文件存放在 `config` 文件夹中。`config/data` 路径下的 `coco.yaml` 保存了数据集相关的配置，
`hyp.scratch-low.yaml` 保存了模型超参数的配置。`yolov5s.yaml` 存放了模型结构的配置。

# [脚本说明](#目录)

## [脚本和示例代码](#目录)

```text
yolov5
├── README.md                                      // descriptions about yolov5
├── __init__.py
├── config
│   ├── args.py                                    // get config parameters from command line
│   ├── data
│   │   ├── coco.yaml                              // configs about dataset
│   │   └── hyp.scratch-low.yaml                   // configs about hyperparameters
│   └── network
│       └── yolov5s.yaml                           // configs about model architecture
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


## [脚本参数](#目录)

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


## [训练过程](#目录)

### Training

对于 Ascend 设备，可以使用以下命令进行单卡训练：

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

对于自定义数据集，您可能需要微调模型的超参数。以上 `Python` 命令会在后台运行。

### 分布式训练

Distributed training example(8p) by shell script:

```bash
# For Ascend device, distributed training example(8p) by shell script
bash run_distribute_train_ascend.sh -c ../config/network/yolov5s.yaml -d ../config/data/coco.yaml \
     -h ../config/data/hyp.scratch-low.yaml -r hccl_8p_xx.json

# For GPU device, distributed training example(8p) by shell script
bash run_distribute_train_gpu.sh ../config/network/yolov5s.yaml ../config/data/coco.yaml \
     ../config/data/hyp.scratch-low.yaml
```


## [推理过程](#目录)

### 推理

在运行以下命令之前，请检查用于推理的 Checkpoint 文件是否存在，名称是否正确。

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

以上 `Python` 命令会在后台运行。您可以通过 `log.txt` 文件查看输出信息。


# [模型说明](#目录)

## [性能](#目录)

### 评估性能

TO BE DONE.

### 推理性能

TO BE DONE.
