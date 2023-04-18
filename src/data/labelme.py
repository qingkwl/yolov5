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
Labelme Dataset files structure

Labelme
├── train
│   ├── 0001.jpg
│   ├── 0001.json
|   ├── 0002.jpg
|   ├── 0002.json
|   └── ...
├── val
│   ├── 0010.jpg
│   ├── 0010.json
|   ├── 0011.jpg
|   ├── 0011.json
|   └── ...
└── test
│   ├── 0020.jpg
│   ├── 0020.json
|   ├── 0021.jpg
|   ├── 0021.json
    └── ...
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import numpy as np

from src.data.base import PATH, BaseArgs, BaseManager, empty, valid_path, COCOArgs, LabelmeArgs


class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(Encoder, self).default(obj)


class LabelmeManager(BaseManager):
    """
        Convert labelme dataset format to other format.
        Support convert to:
            COCO
        The labelme dataset should be formed as the above annotation shows.
    """
    def __init__(self, args: LabelmeArgs) -> None:
        super(LabelmeManager, self).__init__()
        self.args = args
        self._check_dirs()
        self.logger.info(f"Labelme Dataset Args:\n{self.args}")
        self.images: list[dict[str, Any]] = []
        self.categories: dict[str, dict[str, Any]] = {}
        self.annotations: list[dict[str, Any]] = []
        self.ann_id = 1
        self.img_id = 1

    def reset(self) -> None:
        self.images.clear()
        self.categories.clear()
        self.annotations.clear()
        self.ann_id = 1
        self.img_id = 1

    def _check_dirs(self) -> None:
        if empty(self.args.root):
            raise ValueError(f"The root directory is empty, which must be set.")

    def convert(self, target_format: str, data_config: BaseArgs, copy_images: bool = True):
        target_format = target_format.lower()
        self._validate_dataset()
        if target_format == 'coco':
            self._to_coco(data_config, copy_images=copy_images)
        else:
            raise ValueError(f"The target format [{target_format}] is not supported.")

    def _convert_to_coco(self, data_dir: Path, target_dir: Path, copy_images: bool = True) -> dict[str, Any]:
        json_file_list = list(data_dir.rglob("*.json"))
        with logging_redirect_tqdm(loggers=[self.logger]):
            for _, json_file in enumerate(tqdm(json_file_list)):
                img_id = self.img_id
                try:
                    img_info, img_data = self._get_img_info(img_id, json_file)
                    self._get_annotation(img_id, img_data)
                    src_img = data_dir / img_info["file_name"]
                    dst_img = target_dir / f'{img_id:012d}{src_img.suffix}'
                    if copy_images:
                        shutil.copy(src_img, dst_img)
                    img_info["file_name"] = dst_img.name    # Update to new image name
                    self.images.append(img_info)
                except Exception:
                    self.logger.exception(f"Exception when processing json file [{json_file}].")
                    continue
                self.img_id += 1
        coco = {
            'images': self.images,
            'categories': list(self.categories.values()),
            'annotations': self.annotations
        }
        return coco

    def _get_annotation(self, img_id: int, img_data: dict[str, Any]) -> None:
        for shape in img_data['shapes']:
            label = shape['label']
            if label not in self.categories:
                category = {
                    'supercategory': label,
                    'id': len(self.categories) + 1,
                    'name': label
                }
                self.categories[label] = category
            points = shape['points']
            polygon = np.array(points)
            x, y = polygon[:, 0], polygon[:, 1]
            xmin, xmax, ymin, ymax = np.min(x), np.max(x), np.min(y), np.max(y)
            bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
            bbox = list(map(float, bbox))
            _annotation = {
                'segmentation': [list(np.asarray(points).flatten())],
                'iscrowd': 0,
                'image_id': img_id,
                'bbox': bbox,
                'area': bbox[2] * bbox[3],
                'category_id': self.categories[label]['id'],
                'id': self.ann_id
            }
            self.annotations.append(_annotation)
            self.ann_id += 1

    def _get_img_info(self, img_id: int, json_file: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        with open(json_file, 'r', encoding='unicode_escape') as file:
            img_data = json.load(file)
        parent_dir = json_file.parent
        img_path = parent_dir / img_data['imagePath']
        ext_name = img_path.suffix
        lower_ext, upper_ext = ext_name.lower(), ext_name.upper()
        if img_path.with_suffix(lower_ext).exists():
            img_path = img_path.with_suffix(lower_ext)
        elif img_path.with_suffix(upper_ext).exists():
            img_path = img_path.with_suffix(upper_ext)
        else:
            self.logger.warning(f"Image path [{img_path}] does not exist.")
        img_info = {
            'height': img_data['imageHeight'],
            'width': img_data['imageWidth'],
            'id': img_id,
            'file_name': img_path.name
        }
        return img_info, img_data

    def is_img_suffix(self, suffix: str) -> bool:
        return suffix.lower() in ('.jpg', '.png', '.bmp')

    def _to_coco(self, data_config: BaseArgs, copy_images: bool = False) -> None:
        def _convert_data(data_dir: PATH, target_dir: PATH, target_json: PATH):
            self.images.clear()
            self.categories.clear()
            self.annotations.clear()
            data_dir = Path(data_dir)
            target_dir = Path(target_dir)
            target_json = Path(target_json)
            coco = self._convert_to_coco(data_dir, target_dir, copy_images)
            with open(target_json, 'w') as file:
                json.dump(coco, file, indent=4, cls=Encoder)

        self.reset()
        if not isinstance(data_config, COCOArgs):
            raise TypeError(f"The type of data_config is not 'COCOArgs'. Please check it again.")
        data_config.make_dirs()
        if valid_path(self.args.data_dir):
            _convert_data(self.args.data_dir, data_config.data_dir, data_config.data_anno)
        else:
            if valid_path(self.args.train_dir):
                _convert_data(self.args.train_dir, data_config.train_dir, data_config.train_anno)
            if valid_path(self.args.val_dir):
                _convert_data(self.args.val_dir, data_config.val_dir, data_config.val_anno)
            if valid_path(self.args.test_dir):
                _convert_data(self.args.test_dir, data_config.test_dir, data_config.test_anno)

    def _check_images(self, img_dir: PATH) -> None:
        img_dir = Path(img_dir)
        file_paths = list(img_dir.iterdir())
        img_paths = [path for path in file_paths if self.is_img_suffix(path.suffix)]
        ann_paths = [path for path in file_paths if path.suffix.lower() == ".json"]
        if len(img_paths) != len(ann_paths):
            self.logger.warning(f"The number of images in [{img_dir}] is {len(img_paths)} "
                                f"while the number of annotations is {len(ann_paths)}.")

    def _validate_subset(self, img_dir: PATH, strict: bool = True) -> None:
        # Validate image directory and annotation file
        passed = True
        passed = passed and self._check_dir(img_dir, strict)
        # Validate image numbers
        if passed:
            self._check_images(img_dir)
        else:
            self.logger.warning(f"Skip checking images for img_dir [{img_dir}] because the previous check not passed.")

    def _validate_category(self) -> None:
        # if empty(self.args.train_dir) or not exists(self.args.train_dir):
        #     raise FileNotFoundError(f"Training images directory {self.args.train_dir} not found.")
        # TODO: Check category ids consistency
        pass

    def _validate_dataset(self) -> None:
        # Check images
        if valid_path(self.args.data_dir):
            self.logger.info("Checking dataset...")
            self._validate_subset(self.args.data_dir, strict=False)
        else:
            self.logger.info("Checking train dataset...")
            self._validate_subset(self.args.train_dir, strict=False)
            self.logger.info("Checking val dataset...")
            self._validate_subset(self.args.val_dir, strict=False)
            self.logger.info("Checking test dataset...")
            self._validate_subset(self.args.test_dir, strict=False)
            # Check category consistency between train, val (and test if necessary)
            self.logger.info("Checking category consistency...")
            self._validate_category()
