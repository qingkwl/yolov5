# 目录

<details>
    <summary>点此打开/折叠</summary>

- [目录](#目录)
- [YOLOv5说明](#[YOLOv5说明](#目录))
- [模型架构](#[模型架构](#目录))
- [数据集](#[数据集](#目录))
  - [数据下载](#[数据下载](#目录))
  - [数据组织结构](#[数据组织结构](#目录))
  - [数据转换](#[数据转换](#目录))
- [快速入门](#[快速开始](#目录))
- [脚本说明](#[脚本说明](#目录))
    - [脚本和示例代码](#[脚本和示例代码](#目录))
    - [脚本参数](#[脚本参数](#目录))
    - [训练过程](#[训练过程](#目录))
        - [训练](#[训练](#目录))
        - [分布式训练](#[分布式训练](#目录))
    - [验证过程](#[验证过程](#目录))
        - [验证](#[验证](#目录))
    - [推理过程](#[推理过程](#目录))
        - [环境](#[环境](#目录))
        - [推理](#[推理](#目录))
- [模型说明](#[模型说明](#目录))
- [性能](#[性能](#目录))
- [Q&A](#[Q&A](#目录))

</details>

# [YOLOv5说明](#目录)

YOLOv5 由 [Ultralytics](https://ultralytics.com/) 于 2020 年 4 月发布，并在 COCO 数据集目标检测任务中取得了 SOTA 成绩。它是对 YOLOv3 的一个重要改进，
新提出的 **Backbone** 结构以及对于 **Neck** 的改进使得 YOLOv5 在 mAP(mean Average Precision) 上提升了 10%，
在 FPS(Frame Per Second) 上提升了 12%。

官方`PyTorch`实现仓库：https://github.com/ultralytics/yolov5

# [模型架构](#目录)

YOLOv5 模型以添加了 SPP 模块的 CSP 模块与 Focus 模块作为 Backbone，以 PANet 中的 Path-aggregation 模块作为 Neck，
并保留了 YOLOv3 的 Head 模块。[CSP](https://arxiv.org/abs/1911.11929) 作为 Backbone 能够有效增强 CNN 的学习能力。
添加到 CSP 模块中的 [Spatial pyramid pooling](https://arxiv.org/abs/1406.4729) 模块能够增加感受野，分离最显著的上下文特征。
YOLOv4 使用 PANet 替换了 YOLOv3 中的 Feature Pyramid Networks(FPN) 来检测目标，PANet 能够聚合不同层级检测器的参数。

CSPDarknet53 包含了 5 个 CSP 模块，CSP 模块中使用了 kernel size 为 3x3, stride 为 2x2 的卷积层；
而在 PANet 和 SPP 中使用了 1x1、5x5、9x9、13x13 的 max pooling 网络层。

# [数据集](#目录)

`YOLOv5` 使用 `COCO` 数据集的图片，以及 `YOLO` 格式的标注文件进行训练。

## [数据下载](#目录)

数据集:

- 原始数据
    - 训练集：http://images.cocodataset.org/zips/train2017.zip
    - 验证集：http://images.cocodataset.org/zips/val2017.zip
    - 测试集：http://images.cocodataset.org/zips/test2017.zip
- YOLO 格式标注 `coco2017labels`：https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip
- YOLO 格式分割标注 `coco2017labels-segments`：https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels-segments.zip

下载原始数据集的图片，并根据模型的不同下载对应的标注文件：

| Model   | Label                   |
|---------|-------------------------|
| YOLOv5n | coco2017labels          |
| YOLOv5s | coco2017labels          |
| YOLOv5m | coco2017labels-segments |
| YOLOv5l | coco2017labels-segments |
| YOLOv5x | coco2017labels-segments |

## [数据组织结构](#目录)

数据下载完成后，需要按照如下方式组织数据。其中 `images` 文件夹存放图片，
`labels` 文件夹存放对应的标签。`train2017.txt`等文本文件中存放了对应
数据集包含的图片路径。

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

## [数据转换](#目录)

如果使用自定义的 `COCO` 格式或者 `labelme` 格式的数据集，可以使用我们提供的转换脚本转换成 `YOLO` 格式。

转换步骤：

1. 进入 `config/data_conversion` 目录。目录下文件名称对应数据集格式，如 `coco.yaml` 表示 COCO 格式数据集的配置；
2. 配置数据原格式和目标转换格式对应的配置文件，根据实际情况修改其中的路径；
3. 配置文件编辑完成后，运行 `convert_data.py` 脚本，如 `python convert_data.py coco yolo` 表示将 COCO 数据格式转换为 YOLO 格式。

# [快速开始](#目录)

<details>
<summary>安装</summary>

参照 `MindSpore` [官方网站](https://www.mindspore.cn/install)的指引安装 `mindspore` 模块。
之后使用以下命令安装其他所需模块：

```shell
pip install -r requirements.txt
```

</details>

<details>
<summary>训练</summary>

您可以使用以下命令进行单卡训练：

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

或者使用 `shell` 脚本。脚本支持单卡和多卡训练，使用指令如下：

```bash
# Run 1p by shell script, please change `device_target` in config file to run on Ascend/GPU, and change `T_max`, `max_epoch`, `warmup_epochs` refer to contents of notes
bash run_standalone_train_ascend.sh -c ../config/network/yolov5s.yaml -d ../config/data/coco.yaml \
     -h ../config/data/hyp.scratch-low.yaml

# For Ascend device, distributed training example(8p) by shell script
bash run_distribute_train_ascend.sh -c ../config/network/yolov5s.yaml -d ../config/data/coco.yaml \
     -h ../config/data/hyp.scratch-low.yaml -r hccl_8p_xx.json
```

以上脚本可以输入 `--help` 或者 `-H` 来查看更多详细使用方法。

如果使用自定义的数据集，可以使用 `compute_anchors.py` 计算新的 anchors，使用输出值更新对应模型配置文件中的 `anchors` 项。

</details>

<details>
<summary>评估</summary>

您可以使用如下命令评估训练完成的模型：

```bash
# Run evaluation on Ascend/GPU by python command
python val.py \
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

`rect` 开关在某些时候能够增加推理的精度，官方仓库公布的[推理结果](https://github.com/ultralytics/yolov5#pretrained-checkpoints)启用了该配置，
当您需要对比结果时请注意该差异。

您也可以使用 `shell` 脚本进行评估：

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

</details>

# [脚本说明](#目录)

## [脚本和示例代码](#目录)

<details>
    <summary>点此展开/折叠</summary>

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
├── compute_anchors.py                             // compute anchors for specified data
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

## [脚本参数](#目录)

```text
Major parameters in train.py are:

optional arguments:
  --ms_strategy           Training strategy. Default: "StaticShape"
  --distributed_train     Distributed training or not. Default: False
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
  --distributed_eval      Distributed evaluation or not. Default: False
```

## [训练过程](#目录)

### [训练](#目录)

对于 Ascend 设备，可以使用 `shell` 脚本。脚本支持单卡和多卡训练，使用指令如下：

```bash
# Run 1p by shell script, please change `device_target` in config file to run on Ascend/GPU, and change `T_max`, `max_epoch`, `warmup_epochs` refer to contents of notes
bash run_standalone_train_ascend.sh -c ../config/network/yolov5s.yaml -d ../config/data/coco.yaml \
     -h ../config/data/hyp.scratch-low.yaml

# For Ascend device, distributed training example(8p) by shell script
bash run_distribute_train_ascend.sh -c ../config/network/yolov5s.yaml -d ../config/data/coco.yaml \
     -h ../config/data/hyp.scratch-low.yaml -r hccl_8p_xx.json
```


也可以使用以下命令进行单卡训练：

```shell
# Run training example(1p) on Ascend/GPU by python command
python train.py \
    --ms_strategy="StaticShape" \
    --optimizer="momentum" \
    --cfg="../config/network/yolov5s.yaml" \
    --data="../config/data/coco.yaml" \
    --hyp="../config/data/hyp.scratch-low.yaml" \
    --device_target=Ascend \
    --epochs=300 \
    --batch_size=32  > log.txt 2>&1 &
```

**推荐使用脚本运行方式。**


对于自定义数据集，您可能需要微调模型的超参数。以上 `Python` 命令会在后台运行。

### [分布式训练](#目录)

分布式训练脚本示例：

```bash
# For Ascend device, distributed training example(8p) by shell script
bash run_distribute_train_ascend.sh -c ../config/network/yolov5s.yaml -d ../config/data/coco.yaml \
     -h ../config/data/hyp.scratch-low.yaml -r hccl_8p_xx.json

# For GPU device, distributed training example(8p) by shell script
bash run_distribute_train_gpu.sh ../config/network/yolov5s.yaml ../config/data/coco.yaml \
     ../config/data/hyp.scratch-low.yaml
```

也可以通过 OpenMPI 进行分布式训练。需要按照[官方教程](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.0.0-alpha/parallel/train_gpu.html#%E9%85%8D%E7%BD%AE%E5%88%86%E5%B8%83%E5%BC%8F%E7%8E%AF%E5%A2%83)
配置好 OpenMPI 环境，然后执行以下命令：

```bash
bash mpirun_train.sh -c ../config/network/yolov5s.yaml -d ../config/data/coco.yaml \
     -h ../config/data/hyp.scratch-low.yaml
```

## [验证过程](#目录)

### [验证](#目录)

在运行以下命令之前，请检查用于推理的 Checkpoint 文件是否存在，名称是否正确。

```shell
# Run evaluation by python command
python val.py \
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

以上 `Python` 命令会在后台运行。您可以通过 `log.txt` 文件查看输出信息。

也可以使用 OpenMPI 运行分布式推理。需要按照[官方教程](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.0.0-alpha/parallel/train_gpu.html#%E9%85%8D%E7%BD%AE%E5%88%86%E5%B8%83%E5%BC%8F%E7%8E%AF%E5%A2%83)
配置好 OpenMPI 环境，然后执行以下命令：

```bash
bash mpirun_test.sh --w path/to/weights.ckpt -c ../config/network/yolov5s.yaml -d ../config/data/coco.yaml \
     -h ../config/data/hyp.scratch-low.yaml
```

## [推理过程](#目录)

### [环境](#目录)

在昇腾社区下载[MindX SDK 社区版](https://www.hiascend.com/zh/software/mindx-sdk/community)网站中下载对应架构类型版本的`Ascend-mindxsdk-mxmanufacture`软件包，
推荐下载`run`类型的软件包，当前适配的版本为 `3.0`。

下载完成后，先确定配置好已经配置昇腾框架相关的环境变量，然后使用命令：

```shell
bash Ascend-mindxsdk-mxmanufacture_xxx.run --install
```

安装 `MindX` 软件包。安装完成后可以使用 `python -c "import mindx"` 测试是否成功安装。

如果碰到 `libgobject.so.2` 相关的错误，需要设置 `libffi.so.7` 环境变量：

- 使用 `find / -nane "libffi.so.7"` 查找该链接库位置；
- 使用 `export LD_PRELOAD=/path/to/libffi.so.7` 配置环境变量。

### [推理](#目录)

训练获得的模型 `ckpt` 可用 `atc` 工具转换为 `om` 格式的模型，
在推理服务器上执行推理。步骤如下：

1. 导出 `AIR` 格式的模型：
  `python export.py --weights /path/to/model.ckpt --file_format AIR`；
2. 使用 `atc` 工具将 `AIR` 格式模型转换为 `om` 格式：
  `/usr/local/Ascend/latest/atc/bin/atc --model=yolov5s.air --framework=1 --output=./yolov5s --input_format=NCHW --input_shape="Inputs:1,3,640,640" --soc_version=Ascend310`,
  其中 `--soc_version` 可通过 `npu-smi info` 指令查看，支持 `Ascend310`，`Ascend310P3` 等；
3. 通过 `infer.py` 脚本执行推理：`python infer.py --batch_size 1 --om yolov5s.om`

需要注意的是，由于当前`om`格式暂不支持动态`shape`推理，因此无法启用`rect`配置，故推理精度相较`ckpt`格式启用了`rect`配置时偏低。

# [模型说明](#目录)

## [性能](#目录)

| Model   | size<br><sup>(pixels) | mAP<sup>val<br>50-95<br>rect=True | mAP<sup>val<br>50<br>rect=True | mAP<sup>val<br>50-95<br>rect=False | mAP<sup>val<br>50<br>rect=False | Epoch Time(s) | Throughput<br>(images/s) |
|---------|-----------------------|-----------------------------------|--------------------------------|------------------------------------|---------------------------------|---------------|--------------------------|
| YOLOv5n | 640                   | 0.279                             | 0.459                          | 0.277                              | 0.455                           | 66            | 224.00                   |
| YOLOv5s | 640                   | 0.375                             | 0.572                          | 0.373                              | 0.57                            | 79            | 187.14                   |
| YOLOv5m | 640                   | 0.453                             | 0.637                          | 0.451                              | 0.637                           | 133           | 111.16                   |
| YOLOv5l | 640                   | 0.489                             | 0.675                          | 0.486                              | 0.671                           | 163           | 90.70                    |
| YOLOv5x | 640                   | 0.505                             | 0.686                          | 0.506                              | 0.687                           | 221           | 66.90                    |

<details>
<summary>注释</summary>

- 所有模型都使用默认配置，训练 300 epochs。YOLOv5n和YOLOv5s模型使用 hyp.scratch-low.yaml 配置，其他模型都使用 hyp.scratch-high.yaml 配置。
- 下面为不同模型训练时使用的配置：

```bash
--data coco.yaml --epochs 300 --weights '' --cfg yolov5n.yaml  --batch-size 16
                                                 yolov5s.yaml               32
                                                 yolov5m.yaml               24
                                                 yolov5l.yaml               24
                                                 yolov5x.yaml               24
```

- `Epoch Time` 为 Ascend 910A 机器 8 卡测试结果，每张卡的 batch_size 为 32。
- `Throughput` 为 Ascend 910A 的单卡吞吐率。
- **mAP<sup>val</sup>** 在单模型单尺度上计算，数据集使用 [COCO val2017](http://cocodataset.org) 。<br>关键参数为 `--img_size 640 --conf_thres 0.001 --iou_thres 0.65`。
- 当数据处理为性能瓶颈时，可以设置 `--cache_images` 为 `ram` 或者 `disk`，提升数据处理性能。注意 `ram` 可能会导致 out of memory。
- `yolov5n` 需要开启 `--sync_bn`。

</details>


## [Q&A](#目录)

1. cannot allocate memory in static TLS block
```txt
ImportError: /xxx/scikit_image.libs/libgomp-xxx.so: cannot allocate memory in static TLS block
It seems that scikit-image has not been built correctly.
```

这个错误本身与我们的代码无关，而是和依赖 `scikit-image` 库的相关第三方库有关。一般的处理办法改变导入第三方库的顺序，可以在 `train.py` 文件的
最开始部分添加 `import sklearn` 或者 `import skimage` 语句。

如果仍旧不能解决，可以尝试网上搜索其他办法。

<br>
<br>

2. 训练过程中打印的 `loss` 突然变得很大（一般是 `lobj loss` 突然变得很大），或者出现 `nan`。

这个问题一般是模型在训练中，因为溢出导致 `loss` 出现 `nan`，而后在更新模型参数时，使得权重数值变得很大，导致最后计算的 `loss` 也
变大。常出现在单类别数据集，且数据量较小的情况下。

遇到这个问题，可以将 `hyp-scratch.xx.yaml` 文件中 `enable_clip_grad` 项设置为 `True`，以裁剪过大梯度。另外，在最新的代码中，
我们也添加了溢出检测，当出现溢出时，跳过该 step 对模型权重的更新，基本能够避免这一问题的出现。


<br>
<br>


3. `mAP` 结果不高。

这一问题的原因有很多。上文第二点中原因经常会导致 `mAP` 较低，甚至训练过程中的 `mAP` 出现跳动。
采用第二点中的措施之后，一般该问题能够解决。

此外，还可以调节 `config/data/hyp-scratch-xx.yaml` 中的学习率 `lr0`，或者改变脚本中的 `--batch_size` 大小。
