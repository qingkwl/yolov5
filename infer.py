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

import json
import os
import time
from pathlib import Path

import numpy as np
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from config.args import get_args_infer
from deploy.infer_engine.mindx import MindXModel
from src.dataset import create_dataloader
from src.general import LOGGER, coco80_to_coco91_class, xyxy2xywh
from src.metrics import non_max_suppression, scale_coords

# python infer.py --


def Detect(nc=80, anchor=(), stride=()):
    no = nc + 5
    nl = len(anchor)
    na = len(anchor[0]) // 2
    anchor_grid = np.array(anchor).reshape(nl, 1, -1, 1, 1, 2)

    def forward(x):
        z = ()
        outs = ()
        for i in range(len(x)):
            out = x[i]
            bs, _, ny, nx = out.shape
            out = out.reshape(bs, na, no, ny, nx).transpose(0, 1, 3, 4, 2)
            outs += (out,)

            xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
            grid = np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2))
            y = 1 / (1 + np.exp(-out))
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid) * stride[i]
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]
            z += (y.reshape(bs, -1, no),)
        return np.concatenate(z, 1), outs

    return forward


def infer(opt):
    # Create Network
    network = MindXModel(opt.om)
    with open(opt.cfg) as f:
        network_cfg = yaml.load(f, Loader=yaml.SafeLoader)
    detect = Detect(nc=80, anchor=network_cfg['anchors'], stride=network_cfg['stride'])

    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    rank_size = 1
    rank = 0
    val_dataloader, val_dataset, _ = create_dataloader(data_dict['val'], opt.img_size, opt.batch_size,
                                                       stride=32, opt=opt,
                                                       epoch_size=1, pad=0.5, rect=opt.rect,
                                                       rank=rank, rank_size=rank_size,
                                                       num_parallel_workers=4 if rank_size > 1 else 8,
                                                       shuffle=False,
                                                       drop_remainder=False,
                                                       prefix='val: ')

    loader = val_dataloader.create_dict_iterator(output_numpy=True, num_epochs=1)
    dataset_dir = data_dict['val'][:-len(data_dict['val'].split('/')[-1])]
    anno_json_path = os.path.join(dataset_dir, 'annotations/instances_val2017.json')
    coco91class = coco80_to_coco91_class()
    is_coco_dataset = ('coco' in data_dict['dataset_name'])

    step_num = val_dataloader.get_dataset_size()
    sample_num = 0
    infer_times = 0.
    nms_times = 0.
    result_dicts = []
    for i, meta in enumerate(loader):
        img, targets, paths, shapes = meta["img"], meta["label_out"], meta["img_files"], meta["shapes"]
        img = img / 255.0
        nb, _, height, width = img.shape

        # Run infer
        _t = time.time()
        out = network.infer(img)  # inference and training outputs
        out, _ = detect(out)
        infer_times += time.time() - _t

        # Run NMS
        t = time.time()
        out = non_max_suppression(out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, multi_label=True)
        nms_times += time.time() - t

        # Statistics pred
        for si, pred in enumerate(out):
            shape = shapes[si][0]
            path = Path(str(paths[si]))
            sample_num += 1
            if len(pred) == 0:
                continue

            # Predictions
            predn = np.copy(pred)
            scale_coords(img[si].shape[1:], predn[:, :4], shape, ratio_pad=shapes[si][1:])  # native-space pred

            image_id = int(path.stem) if path.stem.isnumeric() else path.stem
            box = xyxy2xywh(predn[:, :4])  # xywh
            box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
            for p, b in zip(pred.tolist(), box.tolist()):
                result_dicts.append({'image_id': image_id,
                                     'category_id': coco91class[int(p[5])] if is_coco_dataset else int(p[5]),
                                     'bbox': [round(x, 3) for x in b],
                                     'score': round(p[4], 5)})
        LOGGER.info(f"Sample {step_num}/{i + 1}, time cost: {(time.time() - _t) * 1000:.2f} ms.")

    # Save predictions json
    if not os.path.exists(opt.output_dir):
        os.mkdir(opt.output_dir)
    with open(os.path.join(opt.output_dir, 'predictions.json'), 'w') as file:
        json.dump(result_dicts, file)

    # Compute mAP
    try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        anno = COCO(anno_json_path)  # init annotations api
        pred = anno.loadRes(result_dicts)  # init predictions api
        eval = COCOeval(anno, pred, 'bbox')
        if is_coco_dataset:
            # eval.params.imgIds = [int(Path(im_file).stem) for im_file in val_dataset.img_files]
            eval.params.imgIds = [int(Path(im_file).stem) for im_file in val_dataset.im_files]
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
        map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
    except Exception as e:
        LOGGER.exception('pycocotools unable to run:')
        raise e

    t = tuple(x / sample_num * 1E3 for x in (infer_times, nms_times, infer_times + nms_times)) + \
        (height, width, opt.batch_size)  # tuple
    LOGGER.info(f'Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g;' % t)

    return map, map50


def main():
    parser = get_args_infer()
    opt = parser.parse_args()
    infer(opt)


if __name__ == '__main__':
    main()
