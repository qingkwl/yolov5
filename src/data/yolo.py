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

"""
YOLO Dataset files structure

YOLO
├── images
|   ├── train
│   |   ├── img0001.jpg
|   |   ├── img0002.jpg
|   |   └── ...
|   ├── val
│   |   ├── img0001.jpg
|   |   ├── img0002.jpg
|   |   └── ...
|   └── test
│       ├── img0001.jpg
|       ├── img0002.jpg
|       └── ...
├── labels
|   ├── train
│   |   ├── img0001.txt
|   |   ├── img0002.txt
|   |   └── ...
|   ├── val
│   |   ├── img0001.txt
|   |   ├── img0002.txt
|   |   └── ...
|   └── test
│       ├── img0001.txt
|       ├── img0002.txt
|       └── ...
├── train.txt
├── val.txt
├── test.txt
└── categories.txt
"""

from __future__ import annotations

import json
import os
import shutil
import time
from typing import Any, Optional, Union
from pathlib import Path

import cv2
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.data.base import PATH, BaseArgs, BaseManager, empty, exists, valid_path, COCOArgs, YOLOArgs
from src.general import empty, WRITE_FLAGS, READ_FLAGS, FILE_MODE


class YOLOManager(BaseManager):
    def __init__(self, args: YOLOArgs, class_map: Optional[Union[list[int], dict[int, int]]] = None) -> None:
        super(YOLOManager, self).__init__()
        self.cur_year = time.strftime('%Y', time.localtime(time.time()))
        self.ann_id = 1
        self.args = args
        # TODO: Customize class map
        self.class_names = self._get_class_names()
        self.class_map = self._get_class_map() if class_map is None else class_map
        self.category_dict = self._get_category()
        self._check_args()
        self.logger.info(f"YOLO Dataset Args:\n {self.args}")

    @staticmethod
    def read_txt(txt_path: PATH) -> list[str]:
        with open(txt_path, 'r', encoding='utf-8') as f:
            data = list(map(lambda x: x.rstrip('\n'), f))
        return data

    def reset(self) -> None:
        # Reset some class members for afterwards conversion
        self.ann_id = 1

    def convert(self, target_format: str, data_config: BaseArgs, copy_images: bool = True) -> None:
        target_format = target_format.lower()
        self._validate_dataset()
        if target_format == "coco":
            self._to_coco(data_config, copy_images)
        else:
            raise ValueError(f"The target format [{target_format}] is not supported.")

    def _get_class_names(self) -> dict[int, str]:
        if isinstance(self.args.names, (str, Path)):
            categories = self.read_txt(self.args.names)
        elif isinstance(self.args.names, list):
            categories = self.args.names
        else:
            raise TypeError(f"Unsupported type {type(self.args.names)} of 'self.args.names', "
                            f"which requires 'str', 'Path' or 'list'")
        self.args.nc = len(categories)
        return {idx: name for idx, name in enumerate(categories)}

    def _get_class_map(self) -> dict[int, int]:
        return {idx: idx for idx in range(len(self.class_names))}

    def _check_args(self) -> None:
        if empty(self.args.root) or not exists(self.args.root):
            raise FileNotFoundError(f"The root directory [{self.args.root}] not found.")

    @staticmethod
    def _count_lines(filename: Path) -> int:
        f = open(filename, 'rb')
        lines = 0
        buf_size = 1024 * 1024
        read_f = f.raw.read

        buf = read_f(buf_size)
        while buf:
            lines += buf.count(b'\n')
            buf = read_f(buf_size)
        f.close()
        return lines

    @staticmethod
    def _get_box_info(vertex_info: list[str], height: int, width: int):
        cx, cy, w, h = [float(i) for i in vertex_info]

        cx = cx * width
        cy = cy * height
        box_w = w * width
        box_h = h * height

        # left top
        x0 = max(cx - box_w / 2, 0)
        y0 = max(cy - box_h / 2, 0)

        # right bottom
        x1 = min(x0 + box_w, width)
        y1 = min(y0 + box_h, height)

        segmentation = [[x0, y0, x1, y0, x1, y1, x0, y1]]
        segmentation = [list(map(lambda x: round(x, 2), seg)) for seg in segmentation]
        bbox = [x0, y0, box_w, box_h]
        bbox = list(map(lambda x: round(x, 2), bbox))
        area = box_w * box_h
        return segmentation, bbox, area

    def _convert_to_coco(self, anno_file: PATH, target_dir: Optional[PATH]) -> dict[str, Any]:
        _images, _annotations = [], []
        anno_file = Path(anno_file)
        if target_dir is not None:
            target_dir = Path(target_dir)
        root = Path(self.args.root)
        img_path_list = self.read_txt(anno_file)
        with logging_redirect_tqdm(loggers=[self.logger]):
            for _img_path in tqdm(img_path_list):
                img_path = root / _img_path
                if not exists(img_path):
                    raise FileNotFoundError(f"Image [{img_path}] not found.")
                img_info = self._get_img_info(img_path, target_dir)
                _images.append(img_info)

                label_path = Path(self.args.root) / 'labels' / anno_file.stem / f'{img_path.stem}.txt'
                if not exists(label_path):
                    self.logger.warning(f"Label [{label_path}] not found.")
                    continue
                _ann = self._get_annotations(label_path, img_info)
                if not empty(_ann):
                    _annotations.extend(_ann)
        json_data = {
            'images': _images,
            'annotations': _annotations,
            'categories': self.category_dict,
        }
        return json_data

    def _get_annotations(self, label_path: Path, img_info: dict) -> list[dict[str, Any]]:
        _ann = []
        img_id, height, width = img_info['id'], img_info['height'], img_info['width']
        with os.fdopen(os.open(label_path, READ_FLAGS, FILE_MODE), 'r', encoding='utf-8') as f:
            label_list = list(map(lambda x: x.rstrip('\n'), f))
        for i, line in enumerate(label_list):
            label_info = line.split(' ')
            if len(label_info) < 5:
                self.logger.warning(f'The {i + 1} line of the [{label_path}] has been corrupted.')
                continue
            category_id, vertex_info = label_info[0], label_info[1:]
            segmentation, bbox, area = self._get_box_info(vertex_info, height, width)
            _ann.append({
                'segmentation': segmentation,
                'area': area,
                'iscrowd': 0,  # YOLO dataset 'iscrowd' equals 0 for all annotations
                'image_id': img_id,
                'bbox': bbox,
                'category_id': self.class_map[int(category_id)],
                'id': self.ann_id,
            })
            self.ann_id += 1
        return _ann

    def _get_img_info(self, img_path: Path, target_dir: Optional[Path]) -> dict[str, Any]:
        img_id = int(img_path.stem)
        img = cv2.imread(str(img_path))
        height, width = img.shape[:2]
        new_img_name = f'{img_id:012d}.jpg'
        save_img_path = img_path
        if target_dir is not None:
            save_img_path = target_dir / new_img_name
            if img_path.suffix.lower() == ".jpg":
                shutil.copyfile(img_path, save_img_path)
            else:
                cv2.imwrite(str(save_img_path), img)
        img_info = {
            'date_captured': self.cur_year,
            'file_name': save_img_path.name,
            'id': img_id,
            'height': height,
            'width': width,
        }
        return img_info

    def _get_category(self) -> list[dict[str, Any]]:
        categories = []
        for key, name in self.class_names.items():
            categories.append({
                'supercategory': name,
                'id': self.class_map[key],
                'name': name,
            })
        return categories

    def _to_coco(self, data_config: BaseArgs, copy_images: bool = False) -> None:
        self.reset()
        if not isinstance(data_config, COCOArgs):
            raise TypeError(f"The type of data_config is not 'COCOArgs'. Please check it again.")
        data_config.make_dirs()

        def convert_data(anno_file: PATH, _target_dir: Optional[PATH], json_path: PATH):
            anno_data = self._convert_to_coco(anno_file, _target_dir)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(anno_data, f, ensure_ascii=False)

        if valid_path(self.args.data_anno):
            target_dir = data_config.data_dir if copy_images else None
            convert_data(self.args.data_anno, target_dir, data_config.data_anno)
        else:
            if valid_path(self.args.train_anno):
                target_dir = data_config.train_dir if copy_images else None
                convert_data(self.args.train_anno, target_dir, data_config.train_anno)
            if valid_path(self.args.val_anno):
                target_dir = data_config.val_dir if copy_images else None
                convert_data(self.args.val_anno, target_dir, data_config.val_anno)
            if valid_path(self.args.test_anno):
                target_dir = data_config.test_dir if copy_images else None
                convert_data(self.args.test_anno, target_dir, data_config.test_anno)

    def _get_img_dir(self, anno_file: Path) -> Path:
        with open(anno_file, "r") as file:
            first_line = file.readline()
        img_path = Path(self.args.root) / first_line
        parent_dir = img_path.parent
        return parent_dir

    def _check_images(self, anno_file: PATH) -> None:
        anno_file = Path(anno_file)
        count_lines = self._count_lines(anno_file)
        parent_dir = self._get_img_dir(anno_file)
        image_num = len(list(parent_dir.iterdir()))
        if count_lines != image_num:
            self.logger.warning(f"The number of lines in [{anno_file}] is {count_lines} "
                                f"while the number of images in [{parent_dir}] is {image_num}.")

    def _validate_subset(self, anno_file: PATH, strict: bool = True) -> None:
        # Validate image directory and annotation file
        passed = True
        passed = passed and self._check_file(anno_file, strict)
        # Validate image numbers
        if passed:
            self._check_images(anno_file)
        else:
            self.logger.warning(f"Skip checking images for annotation file [{anno_file}] "
                                f"because the previous check not passed.")

    def _validate_category(self) -> None:
        pass

    def _validate_dataset(self) -> None:
        # Check images
        if valid_path(self.args.data_anno):
            self.logger.info("Checking dataset...")
            self._validate_subset(self.args.data_anno, strict=False)
        else:
            self.logger.info("Checking train dataset...")
            self._validate_subset(self.args.train_anno, strict=False)
            self.logger.info("Checking val dataset...")
            self._validate_subset(self.args.val_anno, strict=False)
            self.logger.info("Checking test dataset...")
            self._validate_subset(self.args.test_anno, strict=False)
            # Check category consistency between train, val (and test if necessary)
            self.logger.info("Checking category consistency...")
            self._validate_category()
