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

import os
import yaml
from dataclasses import dataclass
from typing import Union
from pathlib import Path

from src.logging import get_logger


PATH = Union[str, Path]


@dataclass
class BaseArgs:
    @classmethod
    def load_args(cls, yaml_path: PATH):
        with open(yaml_path, "r") as file:
            args = yaml.load(file, Loader=yaml.SafeLoader)
        return cls(**args)


def join_path(*args: PATH) -> PATH:
    if (not args) or (not args[0]):
        # If args is empty, or args[0] is None, or args[0] is empty
        return ""
    joined_path = Path(args[0])
    for path in args[1:]:
        if isinstance(path, str) and (not path):
            # If path is empty string
            return ""
        joined_path = joined_path / path
    return joined_path


def empty(path: PATH) -> bool:
    if isinstance(path, str) and (not path):
        return True
    return False


def exists(path: PATH) -> bool:
    if isinstance(path, str):
        return os.path.exists(path)
    if isinstance(path, Path):
        return path.exists()
    return True


def valid_path(path: PATH) -> bool:
    return (not empty(path)) and exists(path)


def mkdir(dir_path: PATH, exist_ok=True) -> None:
    if isinstance(dir_path, str):
        os.makedirs(dir_path, exist_ok=exist_ok)
    if isinstance(dir_path, Path):
        dir_path.mkdir(parents=True, exist_ok=exist_ok)


class BaseManager:
    def __init__(self):
        self.logger = get_logger()

    def _check_dir(self, dir_path: PATH, strict: bool = True) -> bool:
        if empty(dir_path):
            if strict:
                raise FileNotFoundError("The given 'dir_path' is empty.")
            self.logger.warning("The given 'dir_path' is empty.")
            return False
        dir_path = Path(dir_path)
        if not dir_path.exists():
            if strict:
                raise FileNotFoundError(f"Directory {dir_path} not found.")
            self.logger.error(f"Directory {dir_path} not found.")
            return False
        if not dir_path.is_dir():
            if strict:
                raise NotADirectoryError(f"Path {dir_path} is not a directory.")
            self.logger.warning(f"Path {dir_path} is not a directory.")
            return False
        return True

    def _check_file(self, file_path: PATH, strict: bool = True) -> bool:
        if empty(file_path):
            if strict:
                raise FileNotFoundError("The given 'file_path' is empty.")
            self.logger.warning("The given 'file_path' is empty.")
            return False
        file_path = Path(file_path)
        if not file_path.exists():
            if strict:
                raise FileNotFoundError(f"File {file_path} not found.")
            self.logger.warning(f"File {file_path} not found.")
            return False
        if not file_path.is_file():
            if strict:
                raise Exception(f"Path {file_path} is not a file.")
            self.logger.warning(f"Path {file_path} is not a file.")
            return False
        return True


@dataclass
class COCOArgs(BaseArgs):
    root: PATH                       # Data root directory path. Absolute path.
    train_dir: PATH = "train"        # Directory containing training images. Relative path to 'root'.
    val_dir: PATH = "val"            # Directory containing validation images. Relative path to 'root'.
    test_dir: PATH = ""              # Directory containing test images. Relative path to 'root'.
    anno_dir: PATH = "annotations"   # Directory containing annotations. Relative path to 'root'.
    train_anno: PATH = "train.json"  # Training images annotation file. Relative path to 'anno_dir'.
    val_anno: PATH = "val.json"      # Validation images annotation file. Relative path to 'anno_dir'.
    test_anno: PATH = ""             # Test images annotation file. Relative path to 'anno_dir'.
    data_dir: PATH = ""              # Optional. Directory containing all images. Relative path to 'root'.
    data_anno: PATH = ""             # Optional. All images annotation file. Relative path to 'anno_dir'.
    split_ratio: float = 0.8         # Split ratio. Valid when 'data_anno' is set.
    shuffle: bool = False            # Shuffle. Valid when 'data_anno' is set. Shuffle when split data.
    seed: int = 0                    # Random seed used to shuffle data. Valid when 'shuffle' is True.

    def __post_init__(self):
        self.process_path()

    def process_path(self):
        self.root = join_path(self.root)
        if empty(self.root):
            raise ValueError("The 'root' must be non-empty string.")
        self.train_dir = join_path(self.root, self.train_dir)
        self.val_dir = join_path(self.root, self.val_dir)
        self.test_dir = join_path(self.root, self.test_dir)
        self.anno_dir = join_path(self.root, self.anno_dir)
        self.data_dir = join_path(self.root, self.data_dir)

        self.train_anno = join_path(self.anno_dir, self.train_anno)
        self.val_anno = join_path(self.anno_dir, self.val_anno)
        self.test_anno = join_path(self.anno_dir, self.test_anno)
        self.data_anno = join_path(self.anno_dir, self.data_anno)

    def make_dirs(self):
        for folder in (self.root, self.train_dir, self.val_dir, self.test_dir, self.anno_dir, self.data_dir):
            if not empty(folder):
                mkdir(folder)


@dataclass
class YOLOArgs(BaseArgs):
    root: PATH                           # Data root directory path. Absolute path.
    train_anno: PATH = "train.txt"       # Directory containing training images. Relative path to 'root'.
    val_anno: PATH = "val.txt"           # Directory containing validation images. Relative path to 'root'.
    test_anno: PATH = ""                 # Directory containing test images. Relative path to 'root'.
    data_anno: PATH = ""                 # Optional. All images annotation file. Relative path to 'root'.
    # List including all categories' names, or text file relative path to 'root',
    # which contains categories information. One line for one category.
    names: Union[PATH, list[str]] = ""
    nc: int = 0                          # Number of classes

    def __post_init__(self):
        self.process_path()

    def process_path(self):
        self.root = join_path(self.root)
        if empty(self.root):
            raise ValueError("The 'root' must be non-empty string.")
        self.train_anno = join_path(self.root, self.train_anno)
        self.val_anno = join_path(self.root, self.val_anno)
        self.test_anno = join_path(self.root, self.test_anno)
        self.data_anno = join_path(self.root, self.data_anno)
        if isinstance(self.names, str) or isinstance(self.names, Path):
            self.names = join_path(self.root, self.names)

    def make_dirs(self):
        if not empty(self.root):
            mkdir(self.root)
        root = Path(self.root)
        mkdir(root / "labels")
        mkdir(root / "images")
        mkdir(root / "annotations")


@dataclass
class LabelmeArgs(BaseArgs):
    root: PATH                  # Data root directory path. Absolute path.
    train_dir: PATH = "train"   # Directory containing training images. Relative path to 'root'.
    val_dir: PATH = "val"       # Directory containing validation images. Relative path to 'root'.
    test_dir: PATH = ""         # Directory containing test images. Relative path to 'root'.
    data_dir: PATH = ""         # Training and validation images annotation file. Relative path to 'anno_dir'.

    def __post_init__(self):
        self.process_path()

    def process_path(self):
        self.root = join_path(self.root)
        if empty(self.root):
            raise ValueError("The 'root' must be non-empty string.")
        self.train_dir = join_path(self.root, self.train_dir)
        self.val_dir = join_path(self.root, self.val_dir)
        self.test_dir = join_path(self.root, self.test_dir)
        self.data_dir = join_path(self.root, self.data_dir)

    def make_dirs(self):
        for folder in (self.root, self.train_dir, self.val_dir, self.test_dir, self.data_dir):
            if not empty(folder):
                mkdir(folder)
