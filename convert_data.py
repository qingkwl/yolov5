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

import argparse
import ast

from src.data import convert


def get_args():
    # src_format    source dataset format
    # dst_format    destination dataset format
    # src_root      root directory of source dataset
    # dst_root      root directory of destination dataset
    # src_cfg       config path of source dataset
    # dst_cfg       config path of destination dataset
    parser = argparse.ArgumentParser("Convert dataset format")
    parser.add_argument("--src_format", type=str, required=True, help="Source dataset format.")
    parser.add_argument("--dst_format", type=str, required=True, help="Destination dataset format.")
    parser.add_argument("--src_root", type=str, default='', help="Root of source dataset. "
                                                                 "If set, replace 'root' argument in config file.")
    parser.add_argument("--dst_root", type=str, default='', help="Root of destination dataset. "
                                                                 "If set, replace 'root' argument in config file.")
    parser.add_argument("--src_cfg", type=str, default='', help="Config file path of source dataset. "
                                                                "If not set, use default path.")
    parser.add_argument("--dst_cfg", type=str, default='', help="Config file path of destination dataset. "
                                                                "If not set, use default path.")
    parser.add_argument("--split", type=ast.literal_eval, default=False, help="Whether split dataset if necessary.")
    # command = "--src_format labelme --dst_format yolo"
    # command += " --split=True --dst_root=/mnt/d/BaiduNetdiskDownload/labelme2yolo_4"
    # command += " --dst_cfg=/mnt/d/Code/yolov5/config/data_test/yolo_test.yaml"
    # command += " --src_root --dst_root  --src_cfg  --dst_cfg "
    # if not command:
    #     args = parser.parse_args()
    # else:
    # args = parser.parse_args(command.split(" "))
    return args


if __name__ == "__main__":
    args = get_args()
    convert(args.src_format, args.dst_format, args.src_root, args.dst_root, args.src_cfg, args.dst_cfg,
            split=args.split)
