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
# =======================================================================================

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from dataclasses import dataclass, field

import yaml
import numpy as np
from pycocotools.coco import COCO

from config.args import get_args_infer
from deploy.infer_engine.mindx import MindXModel
from src.general import COCOEval as COCOeval
from src.general import LOGGER, coco80_to_coco91_class, xyxy2xywh, empty, WRITE_FLAGS, FILE_MODE
from src.metrics import non_max_suppression, scale_coords
from val import DataManager

# python infer.py --


class Detect:
    def __init__(self, nc=80, anchor=(), stride=()):
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchor)
        self.na = len(anchor[0]) // 2
        self.anchor_grid = np.array(anchor).reshape((self.nl, 1, -1, 1, 1, 2))
        self.stride = stride

    def __call__(self, x):
        z = ()
        outs = ()
        for i, out in enumerate(x):
            bs, _, ny, nx = out.shape
            out = out.reshape(bs, self.na, self.no, ny, nx).transpose(0, 1, 3, 4, 2)
            outs += (out,)

            xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
            grid = np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2))
            y = 1 / (1 + np.exp(-out))
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid) * self.stride[i]
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
            z += (y.reshape(bs, -1, self.no),)
        return np.concatenate(z, 1), outs


def get_data_path(data_dict, subset: str):
    root = data_dict.get('root', None)
    if subset not in ('train', 'val'):
        raise ValueError(f"Only support 'train' or 'val' subset, but given {subset}")
    subset = data_dict.get(subset, None)
    data_dict[subset] = os.path.join(root, subset)
    return data_dict[subset]


@dataclass
class InferStats:
    result_dicts_lst: list = field(default_factory=lambda: [])
    sample_num: int = 0
    infer_time: float = 0.
    nms_time: float = 0.


def compute_map(val_dataset, infer_stats: InferStats, anno_json_path, is_coco_dataset):
    # Compute mAP
    try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        anno = COCO(anno_json_path)  # init annotations api
        pred = anno.loadRes(infer_stats.result_dicts_lst)  # init predictions api
        coco_eval = COCOeval(anno, pred, 'bbox')
        if is_coco_dataset:
            coco_eval.params.imgIds = [int(Path(im_file).stem) for im_file in val_dataset.img_files]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        mean_ap, map50 = coco_eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        LOGGER.info(f"\nCOCO mAP:\n{coco_eval.stats_str}")
    except Exception as e:
        LOGGER.exception('pycocotools unable to run:')
        raise e
    return mean_ap, map50


def show_infer_stats(infer_stats: InferStats, opt):
    sample_num = infer_stats.sample_num
    infer_time = infer_stats.infer_time
    nms_time = infer_stats.nms_time
    t = tuple(x / sample_num * 1E3 for x in (infer_time, nms_time, infer_time + nms_time)) + \
        (opt.img_size, opt.img_size, opt.batch_size)  # tuple
    LOGGER.info('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g;' % t)


def infer(opt):
    # Create Network
    network = MindXModel(opt.om)
    with open(opt.cfg, 'r', encoding='utf-8') as f:
        network_cfg = yaml.load(f, Loader=yaml.SafeLoader)
    detector = Detect(nc=network_cfg['nc'], anchor=network_cfg['anchors'], stride=network_cfg['stride'])
    data_manager = DataManager(opt, network_cfg, hyp=None)
    with open(opt.data, 'r', encoding='utf-8') as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    data_path = get_data_path(data_dict, subset=opt.subset)
    dataset_pack = data_manager.get_dataset(epoch_size=1, mode='val')
    val_dataloader = dataset_pack.dataloader
    val_dataset = dataset_pack.dataset
    is_coco_dataset = 'coco' in data_dict['dataset_name']
    infer_stats = _infer(network, detector, val_dataloader, opt, is_coco_dataset)

    # Save predictions json
    if not os.path.exists(opt.output_dir):
        os.mkdir(opt.output_dir)
    with os.fdopen(os.open(os.path.join(opt.output_dir, 'predictions.json'), WRITE_FLAGS, FILE_MODE), 'w') as file:
        json.dump(infer_stats.result_dicts_lst, file)

    dataset_dir = os.path.dirname(data_path)
    anno_json_path = os.path.join(dataset_dir, opt.ann)
    mean_ap, map50 = compute_map(val_dataset, infer_stats, anno_json_path, is_coco_dataset)
    show_infer_stats(infer_stats, opt)

    return mean_ap, map50


def _infer(network, detector, val_dataloader, opt, is_coco_dataset):
    coco91class = coco80_to_coco91_class()
    infer_stats = InferStats()
    step_num = val_dataloader.get_dataset_size()
    loader = val_dataloader.create_dict_iterator(output_numpy=True, num_epochs=opt.epoch)
    for i, meta in enumerate(loader):
        img, paths, shapes = meta["img"], meta["img_files"], meta["shapes"]
        img = img / 255.0

        # Run infer
        infer_start = time.time()
        out = network.infer(img)  # inference and training outputs
        out, _ = detector(out)
        infer_stats.infer_time += time.time() - infer_start

        # Run NMS
        nms_start = time.time()
        out = non_max_suppression(out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, multi_label=True)
        infer_stats.nms_time += time.time() - nms_start

        # Statistics pred
        for si, pred in enumerate(out):
            shape = shapes[si][0]
            path = Path(str(paths[si]))
            infer_stats.sample_num += 1
            if empty(pred):
                continue

            # Predictions
            predn = np.copy(pred)
            scale_coords(img[si].shape[1:], predn[:, :4], shape, ratio_pad=shapes[si][1:])  # native-space pred

            image_id = int(path.stem) if path.stem.isnumeric() else path.stem
            box = xyxy2xywh(predn[:, :4])  # xywh
            box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
            for p, b in zip(pred.tolist(), box.tolist()):
                infer_stats.result_dicts_lst.append(
                    {'image_id': image_id,
                     'category_id': coco91class[int(p[5])] if is_coco_dataset else int(p[5]),
                     'bbox': [round(x, 3) for x in b],
                     'score': round(p[4], 5)}
                )
        LOGGER.info(f"Sample {step_num}/{i + 1}, time cost: {(time.time() - infer_start) * 1000:.2f} ms.")
    return infer_stats


def main():
    parser = get_args_infer()
    opt = parser.parse_args()

    # Only support infer on single device
    LOGGER.info("Only support inference on single device, set rank = 0, rank_size = 1")
    rank, rank_size = 0, 1
    opt.rank, opt.rank_size = rank, rank_size

    infer(opt)


if __name__ == '__main__':
    main()
