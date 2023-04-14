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

import glob
import math
import os
import re
import threading
import time
from pathlib import Path

import mindspore as ms
import mindspore.nn as nn
import numpy as np
import pkg_resources as pkg
from mindspore import ops

from src.logging import get_logger, set_logger
from src.utils import emojis

LOGGER = get_logger()
set_logger(LOGGER)

try:
    from third_party.fast_coco.fast_coco_eval_api import \
        Fast_COCOeval as COCOeval
    LOGGER.info("Use third party coco eval api to speed up mAP calculation.")
except ImportError:
    LOGGER.exception("Third party coco eval api import failed, use default api.")
    from pycocotools.cocoeval import COCOeval


def check_version(current='0.0.0', minimum='0.0.0', name='version ', pinned=False, hard=False, verbose=False):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    s = f'WARNING ⚠️ {name}{minimum} is required by YOLOv5, but {name}{current} is currently installed'  # string
    if hard:
        assert result, emojis(s)  # assert min requirements met
    if verbose and not result:
        print(s, flush=True)
    return result


def check_file(file):
    # Search for file if not found
    if Path(file).is_file() or file == '':
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), f'File Not Found: {file}'  # assert file was found
        assert len(files) == 1, f"Multiple files match '{file}', specify exact path: {files}"  # assert unique
        return files[0]  # return file


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2
    return boxes


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape (n,2)
    y = np.copy(x)
    y[..., 0] = w * x[..., 0] + padw  # top left x
    y[..., 1] = h * x[..., 1] + padh  # top left y
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        x = clip_boxes(x, (h - eps, w - eps))  # warning: inplace clip
    y = np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 ([N, 4])
        box2 ([M, 4])
    Returns:
        iou ([N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (np.minimum(box1[:, None, 2:], box2[:, 2:]) - np.maximum(box1[:, None, :2], box2[:, :2])).clip(0,
                                                                                                           None).prod(2)
    return inter / (area1[:, None] + area2 - inter) # iou = inter / (area1 + area2 - inter)


def segment2box(segment, width=640, height=640):
    # Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y, = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # xyxy


def segments2boxes(segments):
    # Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh


def bbox_ioa(box1, box2):
    # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

    # Intersection over box2 area
    return inter_area / box2_area


def resample_segments(segments, n=1000):
    # Up-sample an (n,2) segment
    for i, s in enumerate(segments):
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # segment xy
    return segments


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size


def labels_to_class_weights(labels, nc=80):
    # Get class weights (inverse frequency) from training labels
    if labels[0] is None:  # no labels loaded
        return np.array(1)

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int32)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurrences per class

    # Prepend gridpoint count (for uCE training)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return weights


_true = ms.Tensor(True, ms.bool_)


def all_finite_cpu(inputs):
    return _true


class AllReduce(nn.Cell):
    def __init__(self):
        super(AllReduce, self).__init__()
        self.all_reduce = ops.AllReduce(op=ops.ReduceOp.SUM)

    def construct(self, x):
        return self.all_reduce(x)


class Synchronize:
    def __init__(self, rank_size):
        self.all_reduce = AllReduce()
        self.rank_size = rank_size

    def __call__(self):
        sync = ms.Tensor(np.array([1]).astype(np.int32))
        sync = self.all_reduce(sync)  # For synchronization
        sync = sync.asnumpy()[0]
        if sync != self.rank_size:
            raise ValueError(
                f"Sync value {sync} is not equal to number of device {self.rank_size}. "
                f"There might be wrong with devices."
            )


def methods(instance):
    # Get class/instance methods
    return [f for f in dir(instance) if callable(getattr(instance, f)) and not f.startswith('__')]


class Callbacks:
    """"
    Handles all registered callbacks for YOLOv5 Hooks
    """

    def __init__(self):
        # Define the available callbacks
        self._callbacks = {
            'on_pretrain_routine_start': [],
            'on_pretrain_routine_end': [],
            'on_train_start': [],
            'on_train_epoch_start': [],
            'on_train_batch_start': [],
            'optimizer_step': [],
            'on_before_zero_grad': [],
            'on_train_batch_end': [],
            'on_train_epoch_end': [],
            'on_val_start': [],
            'on_val_batch_start': [],
            'on_val_image_end': [],
            'on_val_batch_end': [],
            'on_val_end': [],
            'on_fit_epoch_end': [],  # fit = train + val
            'on_model_save': [],
            'on_train_end': [],
            'on_params_update': [],
            'teardown': [],}
        self.stop_training = False  # set True to interrupt training

    def register_action(self, hook, name='', callback=None):
        """
        Register a new action to a callback hook

        Args:
            hook: The callback hook name to register the action to
            name: The name of the action for later reference
            callback: The callback to fire
        """
        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"
        assert callable(callback), f"callback '{callback}' is not callable"
        self._callbacks[hook].append({'name': name, 'callback': callback})

    def get_registered_actions(self, hook=None):
        """"
        Returns all the registered actions by callback hook

        Args:
            hook: The name of the hook to check, defaults to all
        """
        return self._callbacks[hook] if hook else self._callbacks

    def run(self, hook, *args, thread=False, **kwargs):
        """
        Loop through the registered actions and fire all callbacks on main thread

        Args:
            hook: The name of the hook to check, defaults to all
            args: Arguments to receive from YOLOv5
            thread: (boolean) Run callbacks in daemon thread
            kwargs: Keyword Arguments to receive from YOLOv5
        """

        assert hook in self._callbacks, f"hook '{hook}' not found in callbacks {self._callbacks}"
        for logger in self._callbacks[hook]:
            if thread:
                threading.Thread(target=logger['callback'], args=args, kwargs=kwargs, daemon=True).start()
            else:
                logger['callback'](*args, **kwargs)


class COCOEval(COCOeval):
    # https://github.com/facebookresearch/maskrcnn-benchmark/issues/524
    """
        This is a modified version of original coco eval api which provide map result of each class.
    """

    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        super().__init__(cocoGt, cocoDt, iouType)
        self.stats_str = ''
        self.category_stats = []
        self.category_stats_strs = []

    def summarize(self, categoryIds=None):
        """
            Compute and display summary metrics for evaluation results.
            Note this function can *only* be applied on the default parameter setting
        """
        # categoryIds = -1, get all categories' individual result
        p = self.params
        if categoryIds == -1:
            categoryIds = [i for i, i_catId in enumerate(p.catIds)]
        elif categoryIds is not None:   # list of int
            categoryIds = [i for i, i_catId in enumerate(p.catIds) if i in categoryIds]

        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100, categoryId=None):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
                if categoryId is not None:
                    category_index = [i for i, i_catId in enumerate(p.catIds) if i_catId == categoryId]
                    s = s[:, :, category_index, :]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
                if categoryId is not None:
                    category_index = [i for i, i_catId in enumerate(p.catIds) if i_catId == categoryId]
                    s = s[:, category_index, :]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            return mean_s, iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)

        def _summarizeDets(categoryId=None):
            stats = np.zeros((12,))
            stats_str = [''] * 12
            stats[0], stats_str[0] = _summarize(1, categoryId=categoryId)
            stats[1], stats_str[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2], categoryId=categoryId)
            stats[2], stats_str[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2], categoryId=categoryId)
            stats[3], stats_str[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2], categoryId=categoryId)
            stats[4], stats_str[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2], categoryId=categoryId)
            stats[5], stats_str[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2], categoryId=categoryId)
            stats[6], stats_str[6] = _summarize(0, maxDets=self.params.maxDets[0], categoryId=categoryId)
            stats[7], stats_str[7] = _summarize(0, maxDets=self.params.maxDets[1], categoryId=categoryId)
            stats[8], stats_str[8] = _summarize(0, maxDets=self.params.maxDets[2], categoryId=categoryId)
            stats[9], stats_str[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2], categoryId=categoryId)
            stats[10], stats_str[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2], categoryId=categoryId)
            stats[11], stats_str[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2], categoryId=categoryId)
            return stats, '\n'.join(stats_str)

        def _summarizeKps(categoryId=None):
            stats = np.zeros((10,))
            stats_str = [''] * 10
            stats[0], stats_str[0] = _summarize(1, maxDets=20, categoryId=categoryId)
            stats[1], stats_str[1] = _summarize(1, maxDets=20, iouThr=.5, categoryId=categoryId)
            stats[2], stats_str[2] = _summarize(1, maxDets=20, iouThr=.75, categoryId=categoryId)
            stats[3], stats_str[3] = _summarize(1, maxDets=20, areaRng='medium', categoryId=categoryId)
            stats[4], stats_str[4] = _summarize(1, maxDets=20, areaRng='large', categoryId=categoryId)
            stats[5], stats_str[5] = _summarize(0, maxDets=20, categoryId=categoryId)
            stats[6], stats_str[6] = _summarize(0, maxDets=20, iouThr=.5, categoryId=categoryId)
            stats[7], stats_str[7] = _summarize(0, maxDets=20, iouThr=.75, categoryId=categoryId)
            stats[8], stats_str[8] = _summarize(0, maxDets=20, areaRng='medium', categoryId=categoryId)
            stats[9], stats_str[9] = _summarize(0, maxDets=20, areaRng='large', categoryId=categoryId)
            return stats, '\n'.join(stats_str)

        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps

        self.stats, self.stats_str = summarize()
        if categoryIds is not None:
            self.category_stats = []
            self.category_stats_strs = []
            for category_id in categoryIds:
                category_stat, category_stats_str = summarize(categoryId=category_id)
                self.category_stats.append(category_stat)
                self.category_stats_strs.append(category_stats_str)


class SynchronizeManager:
    def __init__(self, rank, rank_size, distributed, project_dir):
        self.rank = rank
        self.rank_size = rank_size
        self.distributed = distributed  # whether distributed or not
        self.sync = Synchronize(rank_size) if (distributed and rank_size > 1) else None
        self.sync_file = os.path.join(project_dir, 'sync_file.temp')

    def __enter__(self):
        if self.distributed:
            if self.rank == 0:
                LOGGER.info(f"Create sync file {self.sync_file}")
                os.mknod(self.sync_file)
            if self.sync is not None:
                self.sync()
        return self.sync_file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.rank == 0:
            if os.path.exists(self.sync_file):
                LOGGER.info(f"Delete sync file {self.sync_file}")
                os.remove(self.sync_file)
        else:
            LOGGER.info(f"Waiting for rank [0] device...")
            while os.path.exists(self.sync_file):
                time.sleep(1)
            LOGGER.info(f"Rank [{self.rank}] continue executing.")
