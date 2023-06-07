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
COCO Dataset files structure

COCO
├── annotations
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── test2017
│   ├── img0001.jpg
|   ├── img0002.jpg
|   └── ...
├── train2017
│   ├── img0010.jpg
|   ├── img0011.jpg
|   └── ...
└── val2017
    ├── img0020.jpg
    ├── img0021.jpg
    └── ...
"""
from __future__ import annotations

import json
import os
import shutil
import random
from pathlib import Path

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import numpy as np
from pycocotools.coco import COCO

from src.data.base import PATH, BaseArgs, BaseManager, empty_path, exists, valid_path, COCOArgs, YOLOArgs
from src.general import WRITE_FLAGS, FILE_MODE


class COCOManager(BaseManager):
    """
    Convert COCO dataset format to other format.
    Support convert to:
        YOLO
        labelme
        VOC
    The COCO dataset should be formed as the above annotation shows.
    """
    def __init__(self, args: COCOArgs) -> None:
        super(COCOManager, self).__init__()
        self.args = args
        self._check_args()
        self.logger.info(f"COCO Dataset Args:\n {self.args}")

    @staticmethod
    def _get_box(img, ann) -> np.ndarray:
        h, w = img['height'], img['width']
        # The COCO box format is [top left x, top left y, width, height]
        box = np.array(ann['bbox'], dtype=np.float64)
        box[:2] += box[2:] / 2  # xy top-left corner to center
        box[[0, 2]] /= w  # normalize x
        box[[1, 3]] /= h  # normalize y
        return box

    @staticmethod
    def _check_continuity(arr: list) -> bool:
        return all(a + 1 == b for a, b in zip(arr, arr[1:]))

    @staticmethod
    def _min_index(arr1: np.ndarray, arr2: np.ndarray):
        """Find a pair of indexes with the shortest distance.
        Args:
            arr1 (np.ndarray): (N, 2).
            arr2 (np.ndarray): (M, 2).
        Return:
            a pair of indexes(tuple).
        """
        dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
        return np.unravel_index(np.argmin(dis, axis=None), dis.shape)

    def convert(self,
                target_format: str,
                data_config: BaseArgs,
                use_segments: bool = False,
                copy_images: bool = True) -> None:
        """
            Convert coco format to 'target' format, case-insensitive.
            Args:
                target_format (str): target dataset format
                data_config (BaseArgs): target dataset args
                use_segments (bool): whether to use segmentation information
                copy_images (bool): whether copy images to target directory
        """
        # Validate data
        target_format = target_format.lower()
        self._validate_dataset()
        if target_format == "yolo":
            self._to_yolo(data_config, use_segments, copy_images)
        elif target_format == "voc":
            self._to_voc(data_config)
        elif target_format == "labelme":
            self._to_labelme(data_config)
        else:
            raise ValueError(f"The target format [{target_format}] is not supported.")

    def split(self) -> None:
        if empty_path(self.args.data_dir) or not exists(self.args.data_dir):
            raise FileNotFoundError(f"Directory [{self.args.data_dir}] not found.")
        src_dir = Path(self.args.data_dir)
        train_coco, val_coco = self._split_annotations()
        with os.fdopen(os.open(self.args.train_anno, WRITE_FLAGS, FILE_MODE), "w") as file:
            json.dump(train_coco, file, indent=4)
        with os.fdopen(os.open(self.args.val_anno, WRITE_FLAGS, FILE_MODE), "w") as file:
            json.dump(val_coco, file, indent=4)
        self._copy_images(src_dir, "train")
        self._copy_images(src_dir, "val")
        self._validate_dataset()

    def _check_args(self) -> None:
        if empty_path(self.args.root) or not exists(self.args.root):
            raise FileNotFoundError(f"The root directory [{self.args.root}] not found.")

    def _to_yolo(self,
                 data_config: BaseArgs,
                 use_segments: bool = False,
                 copy_images: bool = False) -> None:
        """
        Convert dataset to yolo format.
        """
        if not isinstance(data_config, YOLOArgs):
            raise TypeError(f"The type of data_config is not 'YOLOArgs'. Please check it again.")
        target_dir = Path(data_config.root)
        data_config.make_dirs()

        def convert_data(src_dir: PATH, json_file: PATH) -> None:
            src_dir = Path(src_dir)
            json_file = Path(json_file)
            self.logger.info(f"Processing {json_file} ...")
            folder_name = json_file.stem.replace("instances_", "")
            label_folder = target_dir / "labels" / folder_name
            if label_folder.exists():
                self.logger.warning(f"Annotation file [{json_file}] has been processed. Skip processing.")
                return
            label_folder.mkdir(parents=True, exist_ok=True)
            image_folder = target_dir / "images" / folder_name
            if image_folder.exists():
                self.logger.warning(f"Annotation file [{json_file}] has been processed. Skip processing.")
                return
            image_folder.mkdir(parents=True, exist_ok=True)
            coco = COCO(json_file)
            _categories, _images = coco.cats, coco.imgs
            category_ids = sorted(list(_categories.keys()))
            category_map = {old_id: new_id for new_id, old_id in enumerate(category_ids)}
            img2ann = coco.imgToAnns
            # Write labels file
            with logging_redirect_tqdm(loggers=[self.logger]), \
                    os.fdopen(os.open(target_dir / f"{folder_name}.txt", WRITE_FLAGS, FILE_MODE), "w") as txt:
                for img_id, anns in tqdm(img2ann.items(), desc=f'Annotations {json_file}'):
                    img = _images[img_id]
                    file_name = img['file_name']
                    new_img_name = f'{img_id:012d}.jpg'
                    bboxes, segments = self._get_boxes(img, anns, category_map, use_segments)
                    # Write
                    with os.fdopen(os.open((label_folder / file_name).with_suffix('.txt'),
                                           WRITE_FLAGS, FILE_MODE), 'a') as file:
                        for i, box in enumerate(bboxes):
                            line = (*(segments[i] if use_segments else box),)  # cls, box or segments
                            file.write(('%g ' * len(line)).rstrip() % line + '\n')
                    dst_img_path = image_folder / new_img_name
                    if copy_images:
                        shutil.copy(src_dir / file_name, dst_img_path)
                    txt.write(f"./{dst_img_path.relative_to(target_dir)}\n")
            # Copy ground-truth annotation file
            shutil.copy(json_file, target_dir / "annotations" / json_file.name)

        data_config.make_dirs()
        if valid_path(self.args.data_anno):
            convert_data(self.args.data_dir, self.args.data_anno)
        else:
            if valid_path(self.args.train_anno):
                convert_data(self.args.train_dir, self.args.train_anno)
            if valid_path(self.args.val_anno):
                convert_data(self.args.val_dir, self.args.val_anno)
            if valid_path(self.args.test_anno):
                convert_data(self.args.test_dir, self.args.test_anno)

    def _to_voc(self, data_config: BaseArgs):
        pass

    def _to_labelme(self, data_config: BaseArgs):
        pass

    def _split_annotations(self) -> tuple[dict, dict]:
        if empty_path(self.args.data_anno) or not exists(self.args.data_anno):
            raise FileNotFoundError(f"The annotation file [{self.args.data_anno}] not found.")
        train_coco, val_coco = {}, {}
        with open(self.args.data_anno, "r") as file:
            train_val_ann = json.load(file)
        # Copy information
        for key, value in train_val_ann.items():
            if key not in ("images", "annotations"):
                val_coco[key] = value
                train_coco[key] = value
        coco = COCO(self.args.data_anno)
        images = coco.imgs
        img_ids = sorted(list(images.keys()))
        if self.args.shuffle:
            if isinstance(self.args.seed, int):
                random.seed(self.args.seed)
            random.shuffle(img_ids)
        assert len(set(img_ids)) == len(img_ids), "Image ids duplicated"
        train_img_num = int(len(img_ids) * self.args.split_ratio)
        train_img_ids, val_img_ids = img_ids[:train_img_num], img_ids[train_img_num:]
        train_images = [images[key] for key in train_img_ids]
        val_images = [images[key] for key in val_img_ids]
        train_coco["images"] = train_images
        val_coco["images"] = val_images
        train_coco["annotations"] = [ann for img_id in train_img_ids for ann in coco.imgToAnns[img_id]]
        val_coco["annotations"] = [ann for img_id in val_img_ids for ann in coco.imgToAnns[img_id]]
        return train_coco, val_coco

    def _copy_images(self, src_dir: PATH, mode: str = "train") -> None:
        """
        Args:
            src_dir (PATH): source directory including raw images.
            mode (str): specify which annotation file to use.
        """
        if isinstance(src_dir, str) and (not src_dir):
            self.logger.warning("The given 'src_dir' is empty.")
            return
        src_dir = Path(src_dir)
        if mode == "train":
            ann_file, dst_dir = self.args.train_anno, self.args.train_dir
        elif mode == "val":
            ann_file, dst_dir = self.args.val_anno, self.args.val_dir
        elif mode == "test":
            ann_file, dst_dir = self.args.test_anno, self.args.test_dir
        else:
            raise ValueError(f"Unsupported split mode {mode}.")
        ann_file, dst_dir = Path(ann_file), Path(dst_dir)
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        with open(ann_file, "r") as file:
            data = json.load(file)
        images = data["images"]
        self.logger.info(f"Copying {mode} set images...")
        with logging_redirect_tqdm(loggers=[self.logger]):
            for img in tqdm(images):
                name = img["file_name"]
                shutil.copy(src_dir / name, dst_dir)

    def _merge_multi_segment(self, segments):
        """Merge multi segments to one list.
        Find the coordinates with min distance between each segment,
        then connect these coordinates with one thin line to merge all
        segments into one.
        Args:
            segments(List(List)): original segmentations in coco's json file.
                like [segmentation1, segmentation2,...],
                each segmentation is a list of coordinates.
        """
        s = []
        segments = [np.array(i).reshape(-1, 2) for i in segments]
        idx_list: list[list] = [[] for _ in range(len(segments))]

        # record the indexes with min distance between each segment
        for i in range(1, len(segments)):
            idx1, idx2 = self._min_index(segments[i - 1], segments[i])
            idx_list[i - 1].append(idx1)
            idx_list[i].append(idx2)

        # use two round to connect all the segments
        for k in range(2):
            # forward connection
            if k == 0:
                for i, idx in enumerate(idx_list):
                    # middle segments have two indexes
                    # reverse the index of middle segments
                    if len(idx) == 2 and idx[0] > idx[1]:
                        idx = idx[::-1]
                        segments[i] = segments[i][::-1, :]

                    segments[i] = np.roll(segments[i], -idx[0], axis=0)
                    segments[i] = np.concatenate([segments[i], segments[i][:1]])
                    # deal with the first segment and the last one
                    if i in [0, len(idx_list) - 1]:
                        s.append(segments[i])
                    else:
                        idx = [0, idx[1] - idx[0]]
                        s.append(segments[i][idx[0]:idx[1] + 1])

            else:
                for i in range(len(idx_list) - 1, -1, -1):
                    if i not in [0, len(idx_list) - 1]:
                        idx = idx_list[i]
                        nidx = abs(idx[1] - idx[0])
                        s.append(segments[i][nidx:])
        return s

    def _get_boxes(self, img, anns, category_map, use_segments: bool = False):
        bboxes = []
        segments = []
        h, w = img['height'], img['width']
        for ann in anns:
            if ann['iscrowd']:
                continue
            box = self._get_box(img, ann)
            if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                continue
            # Category mapping
            cls = category_map[ann['category_id']]
            box = [cls] + box.tolist()
            if box not in bboxes:
                bboxes.append(box)
            # Segments
            if use_segments:
                if len(ann['segmentation']) > 1:
                    s = self._merge_multi_segment(ann['segmentation'])
                    s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                else:
                    s = [j for i in ann['segmentation'] for j in i]  # all segments concatenated
                    s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                s = [cls] + s
                if s not in segments:
                    segments.append(s)
        return bboxes, segments

    def _check_images(self, anno: PATH, img_dir: PATH) -> None:
        anno, img_dir = Path(anno), Path(img_dir)
        with open(anno, "r") as file:
            data = json.load(file)
        img_ids = set(img['id'] for img in data['images'])
        img_paths = list(img_dir.iterdir())
        if len(img_ids) != len(img_paths):
            self.logger.warning(f"The number of image ids in annotation [{anno}] is {len(img_ids)} "
                                f"while the number of images in [{img_dir}] is {len(img_paths)}.")
        cat_ids = set(ann['category_id'] for ann in data['annotations'])
        category_ids = set(cat['id'] for cat in data['categories'])
        assert category_ids == (cat_ids | category_ids), \
            'Category ids are not consistent in annotation file.'

    def _validate_subset(self, anno_path: PATH, img_dir: PATH, strict: bool = True) -> None:
        # Validate image directory and annotation file
        passed = True
        passed = passed and self._check_file(anno_path, strict)
        passed = passed and self._check_dir(img_dir, strict)
        # Validate image numbers
        if passed:
            self._check_images(anno_path, img_dir)
        else:
            self.logger.warning(f"Skip checking images for anno_path [{anno_path}] "
                                f"and img_dir [{img_dir}] because the previous check not passed.")

    def _validate_category(self) -> None:
        if empty_path(self.args.train_anno) or not exists(self.args.train_anno):
            raise FileNotFoundError(f"Training annotation file [{self.args.train_anno}] not found.")
        with open(self.args.train_anno, "r") as file:
            train_anno = json.load(file)
        train_cat_ids = set(cat['id'] for cat in train_anno['categories'])

        def get_valid_ids(anno: PATH):
            if empty_path(anno) or not exists(anno):
                return None
            with open(anno, "r") as ann_file:
                _anno = json.load(ann_file)
            cat_ids = set(cat['id'] for cat in _anno['categories'])
            return cat_ids

        val_cat_ids = get_valid_ids(self.args.val_anno)
        test_cat_ids = get_valid_ids(self.args.test_anno)
        if val_cat_ids is not None:
            assert train_cat_ids == val_cat_ids, \
                "The ids of categories in training and validation subset are not consistent. Please check it again."
        if test_cat_ids is not None:
            assert train_cat_ids == test_cat_ids, \
                "The ids of categories in training and test subset are not consistent. Please check it again."

    def _validate_dataset(self) -> None:
        # Check images
        if valid_path(self.args.data_dir):
            self.logger.info("Checking dataset...")
            self._validate_subset(self.args.data_anno, self.args.data_dir, strict=False)
        else:
            self.logger.info("Checking train dataset...")
            self._validate_subset(self.args.train_anno, self.args.train_dir, strict=False)
            self.logger.info("Checking val dataset...")
            self._validate_subset(self.args.val_anno, self.args.val_dir, strict=False)
            self.logger.info("Checking test dataset...")
            self._validate_subset(self.args.test_anno, self.args.test_dir, strict=False)
            # Check category consistency between train, val (and test if necessary)
            self.logger.info("Checking category consistency...")
            self._validate_category()
