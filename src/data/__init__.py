# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Type
import yaml

from src.data.base import BaseArgs, COCOArgs, YOLOArgs, LabelmeArgs, exists
from src.data.coco import COCOManager
from src.data.yolo import YOLOManager
from src.data.labelme import LabelmeManager
from src.general import LOGGER

_dataset_arg_mapping: dict[str, Type[BaseArgs]] = {
    "coco": COCOArgs,
    "yolo": YOLOArgs,
    "labelme": LabelmeArgs
}

# PROJECT_ROOT/config
DATA_CONFIG_ROOT = Path(__file__).parent.parent.parent / "config" / "data_conversion"


def merge_args(config):
    dataset_name: str = config.dataset_name.lower()
    model_name: str = config.model_name.lower()
    root: str = config.root
    if model_name.startswith("yolov5") or model_name.startswith("yolov7"):
        target_format = "yolo"
    elif model_name.startswith("yolox"):
        target_format = "coco"
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    if target_format == 'yolo':
        # arg_class: Type[YOLOArgs] = _dataset_arg_mapping["yolo"]
        # TODO: Add dataset yaml path to config
        arg_path = DATA_CONFIG_ROOT / "yolo.yaml"
        with open(arg_path, "r") as file:
            args_dict = yaml.load(file, Loader=yaml.SafeLoader)
        # args: YOLOArgs = arg_class.load_args(arg_path)
        if not root:
            args_dict["root"] = root
        args = YOLOArgs(**args_dict)
        config.train = str(args.train_anno)
        config.val = str(args.val_anno)
        config.eval = str(args.test_anno)
        config.nc = int(args.nc)
        if isinstance(args.names, (str, Path)):
            config.names = YOLOManager.read_txt(args.names)
    else:
        raise NotImplementedError(f"Unsupported dataset {dataset_name}.")
    return config


def convert(src_format: str, dst_format: str, src_root: str, dst_root: str, src_cfg: str, dst_cfg: str,
            split: bool = False):
    """
        Convert data format from 'src_format' to 'dst_format'
    """
    # TODO: refactor the following code
    def load_args(path: str):
        with open(path, "r") as file:
            args = yaml.load(file, Loader=yaml.SafeLoader)
        return args

    def get_args(src_cfg_: str, dst_cfg_: str):
        src_cfg_ = DATA_CONFIG_ROOT / f"{src_format}.yaml" if not src_cfg_ else src_cfg_
        dst_cfg_ = DATA_CONFIG_ROOT / f"{dst_format}.yaml" if not dst_cfg_ else dst_cfg_
        src_arg_ = load_args(src_cfg_)
        dst_arg_ = load_args(dst_cfg_)
        if src_root:
            src_arg["root"] = src_root
        if dst_root:
            dst_arg_["root"] = dst_root
        return src_arg_, dst_arg_

    coco_args: COCOArgs
    yolo_args: YOLOArgs
    labelme_args: LabelmeArgs
    _logger = LOGGER
    src_format, dst_format = src_format.lower(), dst_format.lower()
    if dst_format == 'yolo':
        if src_format == 'coco':
            # Convert dataset from 'coco' to 'yolo'
            src_arg, dst_arg = get_args(src_cfg, dst_cfg)
            coco_args = COCOArgs(**src_arg)
            coco_manager = COCOManager(coco_args)
            yolo_args = YOLOArgs(**dst_arg)
            if exists(yolo_args.root):
                _logger.info(f"Skip conversion because {yolo_args.root} exists.")
                return
            if split:
                coco_manager.split()
            coco_manager.convert("yolo", yolo_args)
        elif src_format == 'labelme':
            # Convert dataset from 'labelme' to 'coco'
            src_arg, dst_arg = get_args(src_cfg, dst_cfg)
            labelme_args = LabelmeArgs(**src_arg)
            labelme_manager = LabelmeManager(labelme_args)
            coco_args = COCOArgs.load_args(DATA_CONFIG_ROOT / "coco.yaml")
            # TODO: intermediate configs may need to modify by src_arg
            yolo_args = YOLOArgs(**dst_arg)
            if exists(yolo_args.root):
                _logger.info(f"Skip conversion because {yolo_args.root} exists.")
                return
            labelme_manager.convert("coco", coco_args)
            # Convert dataset from 'coco' to 'yolo'
            coco_manager = COCOManager(coco_args)
            if split:
                coco_manager.split()
                coco_args.data_anno = ""
                coco_args.data_dir = ""  # Reset data_anno and data_dir
            coco_manager = COCOManager(coco_args)
            coco_manager.convert("yolo", yolo_args)
    elif dst_format == 'coco':
        if src_format == 'yolo':
            src_arg, dst_arg = get_args(src_cfg, dst_cfg)
            # Convert from 'yolo' to 'coco'
            yolo_args = YOLOArgs(**src_arg)
            yolo_manager = YOLOManager(yolo_args)
            coco_args = COCOArgs(**dst_arg)
            if exists(coco_args.root):
                _logger.info(f"Skip conversion because {coco_args.root} exists.")
                return
            yolo_manager.convert("coco", coco_args)
        elif src_format == 'labelme':
            src_arg, dst_arg = get_args(src_cfg, dst_cfg)
            labelme_args = LabelmeArgs(**src_arg)
            labelme_manager = LabelmeManager(labelme_args)
            coco_args = COCOArgs(**dst_arg)
            if exists(coco_args.root):
                _logger.info(f"Skip conversion because {coco_args.root} exists.")
                return
            labelme_manager.convert("coco", coco_args)
    else:
        raise NotImplementedError(f"Not supported data format conversion from {src_format} to {dst_format}")
