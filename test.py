import io
import glob
import os
import time
import json
import yaml
import codecs
from typing import Optional, List
from pathlib import Path
from contextlib import redirect_stdout

import numpy as np
from pycocotools.coco import COCO

import mindspore as ms
from mindspore.context import ParallelMode
from mindspore import context, Tensor, ops
from mindspore.communication.management import init, get_rank, get_group_size

from src.general import LOGGER
from src.network.yolo import Model
from config.args import get_args_test
from src.general import coco80_to_coco91_class, check_file, check_img_size, xyxy2xywh, xywh2xyxy, \
    colorstr, box_iou, Synchronize, increment_path, Callbacks
from src.general import COCOEval as COCOeval
from src.dataset import create_dataloader
from src.metrics import ConfusionMatrix, non_max_suppression, scale_coords, ap_per_class
from src.plots import plot_study_txt, plot_images, output_to_target
from third_party.yolo2coco.yolo2coco import YOLO2COCO


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)})


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = np.array(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(np.array(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(np.bool_)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = np.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = np.concatenate((np.stack(x, 1), iou[x[0], x[1]][:, None]), 1)  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return np.array(correct).astype(np.bool_)


def load_checkpoint_to_yolo(model, ckpt_path):
    param_dict = ms.load_checkpoint(ckpt_path)
    new_params = {}
    for k, v in param_dict.items():
        if k.startswith("model.") or k.startswith("updates"):
            new_params[k] = v
        if k.startswith("ema.ema."):
            k = k[len("ema.ema."):]
            new_params[k] = v
    ms.load_param_into_net(model, new_params)
    LOGGER.info(f"load ckpt from \"{ckpt_path}\" success.")


def compute_coco_stats(cfg, metric_stats):
    # Compute metrics
    # p, r, f1, mp, mr, map50, map = 0., 0., 0., 0., 0., 0., 0.
    # ap, ap_class = metric_stats.avg_precis, metric_stats.avg_precis_class
    # stats = metric_stats.pred_stats
    # stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    metric_stats.pred_stats = [np.concatenate(x, 0) for x in zip(*metric_stats.pred_stats)]  # to numpy
    stats = metric_stats.pred_stats
    seen = metric_stats.seen
    names = cfg.names
    nc = cfg.nc
    verbose = cfg.verbose
    training = cfg.training
    plots = cfg.plots
    save_dir = cfg.save_dir
    if len(stats) and stats[0].any():
        # tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        result = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        metric_stats.set_ap_per_class(result)
        # ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        metric_stats.ap50 = metric_stats.ap[:, 0]
        metric_stats.ap = metric_stats.ap.mean(1)
        # mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        # metric_stats.set_mean_stats(metric_stats.precision, metric_stats.recall, ap50, ap)
        metric_stats.set_mean_stats()
    nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class

    # Print results
    title = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
    pf = '{:20s}' + '{:12d}' * 2 + '{:12.3g}' * 4  # print format
    LOGGER.info(title)
    # print(pf.format('all', seen, nt.sum(), mp, mr, map50, map))
    LOGGER.info(pf.format('all', seen, nt.sum(), *metric_stats.get_mean_stats()))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(metric_stats.ap_class):
            # Class     Images  Instances          P          R      mAP50   mAP50-95:
            # print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))
            LOGGER.info(pf.format(names[c], seen, nt[c], *metric_stats.get_ap_per_class(i)))


def coco_eval(anno_json, pred_json, dataset, is_coco):
    anno = COCO(anno_json)  # init annotations api
    pred = anno.loadRes(pred_json)  # init predictions api
    eval_result = COCOeval(anno, pred, 'bbox')
    if is_coco:
        eval_result.params.imgIds = [int(Path(x).stem) for x in dataset.im_files]  # image IDs to evaluate
    eval_result.evaluate()
    eval_result.accumulate()
    eval_result.summarize(categoryIds=-1)
    coco_result = COCOResult(eval_result)
    return coco_result


def merge_json(project_dir, prefix):
    merged_json = os.path.join(project_dir, f"{prefix}_predictions_merged.json")
    merged_result = []
    for json_file in Path(project_dir).rglob("*.json"):
        LOGGER.info(f"Merge json file {json_file.resolve()}")
        with open(json_file, "r") as file_handler:
            merged_result.extend(json.load(file_handler))
    with open(merged_json, "w") as file_handler:
        json.dump(merged_result, file_handler)
    LOGGER.info(f"Write merged results to file {merged_json} successfully.")
    return merged_json, merged_result


def view_result(anno_json, result_json, val_path, score_threshold=None, recommend_threshold=False):
    from src.coco_visual import CocoVisualUtil
    dataset_coco = COCO(anno_json)
    coco_visual = CocoVisualUtil()
    eval_types = ["bbox"]
    config = {"dataset": "coco"}
    data_dir = Path(val_path).parent
    img_path_name = os.path.splitext(os.path.basename(val_path))[0]
    im_path_dir = os.path.join(data_dir, "images", img_path_name)
    config = Dict(config)
    with open(result_json, 'r') as f:
        result = json.load(f)
    result_files = coco_visual.results2json(dataset_coco, result, "./results.pkl")
    coco_visual.coco_eval(config, result_files, eval_types, dataset_coco, im_path_dir=im_path_dir,
                          score_threshold=score_threshold, recommend_threshold=recommend_threshold)


def compute_metrics(img, targets, out, paths, shapes, cfg, metric_stats):
    iouv = metric_stats.iouv
    niou = metric_stats.niou
    single_cls = cfg.single_cls
    for si, pred in enumerate(out):
        labels = targets[targets[:, 0] == si, 1:]
        nl, npr, shape = labels.shape[0], pred.shape[0], shapes[si][0]  # number of labels, predictions
        if type(paths[si]) is np.ndarray or type(paths[si]) is np.bytes_:
            path = Path(str(codecs.decode(paths[si].tostring()).strip(b'\x00'.decode())))
        else:
            path = Path(paths[si])
        correct = np.zeros((npr, niou)).astype(np.bool_)  # init
        metric_stats.seen += 1

        if npr == 0:
            if nl:
                metric_stats.pred_stats.append((correct, *np.zeros((2, 0)).astype(np.bool_), labels[:, 0]))
                if cfg.plots:
                    metric_stats.confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
            continue

        # Predictions
        if single_cls:
            pred[:, 5] = 0
        predn = np.copy(pred)
        predn[:, :4] = scale_coords(img[si].shape[1:], predn[:, :4], shape, shapes[si][1:])  # native-space pred

        # Evaluate
        if nl:
            tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
            tbox = scale_coords(img[si].shape[1:], tbox, shape, shapes[si][1:])  # native-space labels
            labelsn = np.concatenate((labels[:, 0:1], tbox), 1)  # native-space labels
            correct = process_batch(predn, labelsn, iouv)
            if cfg.plots:
                metric_stats.confusion_matrix.process_batch(predn, labelsn)
        # (correct, conf, pcls, tcls)
        metric_stats.pred_stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))

        # Save/log
        # if save_txt:
        if cfg.save_txt:
            save_one_txt(predn, cfg.save_conf, shape, file=os.path.join(cfg.save_dir, 'labels', f'{path.stem}.txt'))
        # if save_json:
        if cfg.save_json:
            save_one_json(predn, metric_stats.pred_json, path, cfg.class_map)  # append to COCO-JSON dictionary


class COCOResult:
    def __init__(self, eval_result=None):
        if eval_result is not None:
            self.stats = eval_result.stats # np.ndarray
            self.stats_str = eval_result.stats_str  # str
            self.category_stats = eval_result.category_stats    # List[np.ndarray]
            self.category_stats_strs = eval_result.category_stats_strs    # List[str]
        else:
            self.stats = None
            self.stats_str = ''
            self.category_stats = []
            self.category_stats_strs = []

    def get_map(self):
        if self.stats is None:
            return -1
        return self.stats[0]

    def get_map50(self):
        if self.stats is None:
            return -1
        return self.stats[1]


class MetricStatistics:
    def __init__(self):
        self.mp = 0.        # mean precision
        self.mr = 0.        # mean recall
        self.map50 = 0.     # mAP@50
        self.map = 0.       # mAP@50:95
        self.loss_box = 0.
        self.loss_obj = 0.
        self.loss_cls = 0.

        self.pred_json = []
        self.pred_stats = []
        self.tp = 0.    # true positive
        self.fp = 0.    # false positive
        self.precision = 0.
        self.recall = 0.
        self.f1 = 0.
        self.ap = []        # average precision(AP)
        self.ap50 = []      # average precision@50(AP@50)
        self.ap_class = []  # average precision(AP) of each class

        self.seen = 0
        self.confusion_matrix = None

        self.iouv = np.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = np.prod(self.iouv.shape)

    def __iter__(self):
        for name, val in vars(self).items():
            yield val

    def set_loss(self, loss):
        self.loss_box, self.loss_obj, self.loss_cls = loss.tolist()

    def get_loss_tuple(self):
        return self.loss_box, self.loss_obj, self.loss_cls

    def set_mean_stats(self):
        self.mp = self.precision.mean()
        self.mr = self.recall.mean()
        self.map50 = self.ap50.mean()
        self.map = self.ap.mean()

    def get_mean_stats(self):
        return self.mp, self.mr, self.map50, self.map

    def set_ap_per_class(self, result):
        # result: return value of ap_per_class() function
        self.tp, self.fp = result[:2]
        self.precision, self.recall, self.f1 = result[2:5]
        self.ap = result[5]
        self.ap_class = result[6]

    def get_ap_per_class(self, idx):
        return self.precision[idx], self.recall[idx], self.ap50[idx], self.ap[idx]


class TimeStatistics:
    def __init__(self):
        self.total_infer_duration = 0.
        self.total_nms_duration = 0.
        self.total_metric_duration = 0.

    def total_duration(self):
        return self.total_infer_duration + self.total_nms_duration + self.total_metric_duration

    def get_tuple(self):
        return self.total_infer_duration, self.total_nms_duration, self.total_metric_duration, self.total_duration()


class Config:
    def __init__(self):
        self.nc = 0
        self.names = {}
        self.verbose = False
        self.training = False
        self.save_conf = False
        self.save_json = False
        self.save_txt = False
        self.plots = False
        self.save_dir = ''
        self.class_map = []
        self.single_cls = False


def test(data,
         weights=None,
         batch_size=32,
         imgsz=640,
         conf_thres=0.001,
         iou_thres=0.6,  # for NMS
         save_json=False,
         single_cls=False,
         augment=False,
         verbose=False,
         model=None,
         dataloader=None,
         dataset=None,
         save_dir=Path(''),  # for saving images
         save_txt=False,  # for auto-labelling
         save_hybrid=False,  # for hybrid auto-labelling
         save_conf=False,  # save auto-label confidences
         plots=False,
         compute_loss=None,
         half_precision=False,
         trace=False,
         rect=False,
         is_coco=False,
         v5_metric=False,
         is_distributed=False,
         rank=0,
         rank_size=1,
         opt=None,
         cur_epoch=0,
         callbacks=Callbacks()):
    # Configure
    if isinstance(data, str):
        is_coco = data.endswith('coco.yaml')
        with open(data) as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
    cfg = Config()
    cfg.nc = 1 if single_cls else int(data['nc'])  # number of classes
    cfg.verbose = verbose
    cfg.training = model is not None
    cfg.save_conf = save_conf
    cfg.save_json = save_json
    cfg.save_txt = save_txt
    cfg.plots = plots
    cfg.single_cls = single_cls
    synchronize = Synchronize(rank_size) if is_distributed else None
    project_dir = os.path.join(opt.project, f"epoch_{cur_epoch}")
    save_dir = os.path.join(project_dir, f"save_dir_{rank}")
    save_dir = increment_path(save_dir, exist_ok=opt.exist_ok)
    os.makedirs(os.path.join(save_dir, f"labels_{rank}"), exist_ok=opt.exist_ok)
    cfg.save_dir = save_dir
    # Initialize/load model and set device
    if model is None:  # called by train.py
        # Load model
        # Hyperparameters
        with open(opt.hyp) as f:
            hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
        model = Model(opt.cfg, ch=3, nc=cfg.nc, anchors=hyp.get('anchors'), sync_bn=False, hyp=hyp)  # create
        ckpt_path = weights
        load_checkpoint_to_yolo(model, ckpt_path)
        gs = max(int(ops.cast(model.stride, ms.float16).max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check img_size

    gs = max(int(ops.cast(model.stride, ms.float16).max()), 32)  # grid size (max stride)
    imgsz = imgsz[0] if isinstance(imgsz, list) else imgsz
    imgsz = check_img_size(imgsz, s=gs)  # check img_size

    # Half
    if half_precision:
        model.to_float(ms.float16)

    model.set_train(False)

    # Dataloader
    if dataloader is None or dataset is None:
        LOGGER.info(f"Enable rect: {rect}")
        task = opt.task if opt.task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader, dataset, per_epoch_size = create_dataloader(data[task], imgsz, batch_size, gs, opt,
                                                                epoch_size=1, pad=0.5, rect=rect,
                                                                rank=rank,
                                                                rank_size=rank_size,
                                                                num_parallel_workers=4 if rank_size > 1 else 8,
                                                                shuffle=False,
                                                                drop_remainder=False,
                                                                prefix=colorstr(f'{task}: '))
        assert per_epoch_size == dataloader.get_dataset_size()
        data_loader = dataloader.create_dict_iterator(output_numpy=True, num_epochs=1)
        LOGGER.info(f"Test create dataset success, epoch size {per_epoch_size}.")
    else:
        assert dataset is not None
        assert dataloader is not None
        per_epoch_size = dataloader.get_dataset_size()
        data_loader = dataloader.create_dict_iterator(output_numpy=True, num_epochs=1)

    if v5_metric:
        LOGGER.info("Testing with YOLOv5 AP metric...")

    metric_stats = MetricStatistics()
    time_stats = TimeStatistics()
    metric_stats.confusion_matrix = ConfusionMatrix(nc=cfg.nc)
    cfg.names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}

    start_idx = 1
    cfg.class_map = coco80_to_coco91_class() if is_coco else list(range(start_idx, 1000 + start_idx))

    loss = np.zeros(3)
    # jdict, stats, ap, ap_class = [], [], [], []
    callbacks.run('on_val_start')
    step_start_time = time.time()
    for batch_i, meta in enumerate(data_loader):
        callbacks.run('on_val_batch_start')
        img, targets, paths, shapes = meta["img"], meta["label_out"], meta["img_files"], meta["shapes"]
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        img_tensor = Tensor.from_numpy(img)
        targets_tensor = Tensor.from_numpy(targets)
        if half_precision:
            img_tensor = ops.cast(img_tensor, ms.float16)
            targets_tensor = ops.cast(targets_tensor, ms.float16)

        targets = targets.reshape((-1, 6))
        targets = targets[targets[:, 1] >= 0]
        nb, _, height, width = img.shape  # batch size, channels, height, width
        data_duration = time.time() - step_start_time
        # Run model
        infer_start_time = time.time()
        # inference and training outputs
        if compute_loss or not augment:
            pred_out, train_out = model(img_tensor)
        else:
            pred_out, train_out = (model(img_tensor, augment=augment), None)
        infer_duration = time.time() - infer_start_time
        time_stats.total_infer_duration += infer_duration

        # Compute loss
        if compute_loss:
            loss += compute_loss(train_out, targets_tensor)[1][:3].asnumpy()  # box, obj, cls

        # NMS
        targets[:, 2:] *= np.array([width, height, width, height], targets.dtype)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        nms_start_time = time.time()
        out = non_max_suppression(pred_out.asnumpy(),
                                  conf_thres,
                                  iou_thres,
                                  labels=lb,
                                  multi_label=True,
                                  agnostic=single_cls)
        nms_duration = time.time() - nms_start_time
        time_stats.total_nms_duration += nms_duration

        # Metrics
        metric_start_time = time.time()
        compute_metrics(img, targets, out, paths, shapes, cfg, metric_stats)
        metric_duration = time.time() - metric_start_time
        time_stats.total_metric_duration += metric_duration
        # Plot images
        if cfg.plots and batch_i < 3:
            labels_path = os.path.join(cfg.save_dir, f'test_batch{batch_i}_labels.jpg')  # labels
            plot_images(img, targets, paths, labels_path, cfg.names)
            pred_path = os.path.join(cfg.save_dir, f'test_batch{batch_i}_pred.jpg')  # predictions
            plot_images(img, output_to_target(out), paths, pred_path, cfg.names)

        LOGGER.info(f"Step {batch_i + 1}/{per_epoch_size} "
                    f"Time total {(time.time() - step_start_time):.2f}s  "
                    f"Data {data_duration * 1e3:.2f}ms  "
                    f"Infer {infer_duration * 1e3:.2f}ms  "
                    f"NMS {nms_duration * 1e3:.2f}ms  "
                    f"Metric {metric_duration * 1e3:.2f}ms")
        step_start_time = time.time()

    compute_coco_stats(cfg, metric_stats)

    # Print speeds
    total_time_fmt_str = 'Total time: {:.1f}/{:.1f}/{:.1f}/{:.1f} s ' \
                         'inference/NMS/Metric/total {:g}x{:g} image at batch-size {:g}'
    speed_fmt_str = 'Speed: {:.1f}/{:.1f}/{:.1f}/{:.1f} ms ' \
                    'inference/NMS/Metric/total per {:g}x{:g} image at batch-size {:g}'
    total_time = (*time_stats.get_tuple(), imgsz, imgsz, batch_size)  # tuple
    speed = tuple(x / metric_stats.seen * 1E3 for x in total_time[:4]) + (imgsz, imgsz, batch_size)  # tuple

    LOGGER.info(speed_fmt_str.format(*speed))
    LOGGER.info(total_time_fmt_str.format(*total_time))

    # Plots
    if cfg.plots:
        metric_stats.confusion_matrix.plot(save_dir=cfg.save_dir, names=list(cfg.names.values()))

    coco_result = COCOResult()
    # Save JSON
    if save_json and len(metric_stats.pred_json):
        w = Path(weights).stem if weights is not None else ''  # weights
        data_dir = Path(data["val"]).parent
        anno_json = os.path.join(data_dir, "annotations/instances_val2017.json")
        if opt.transfer_format and not os.path.exists(
                anno_json):  # data format transfer if annotations does not exists
            LOGGER.info("Transfer annotations from yolo to coco format.")
            transformer = YOLO2COCO(data_dir, output_dir=data_dir,
                                    class_names=data["names"], class_map=cfg.class_map,
                                    mode='val', annotation_only=True)
            transformer()
        pred_json_path = os.path.join(cfg.save_dir, f"{w}_predictions_{rank}.json")  # predictions json
        LOGGER.info(f'Evaluating pycocotools mAP... saving {pred_json_path}...')
        with open(pred_json_path, 'w') as f:
            json.dump(metric_stats.pred_json, f)
        sync_tmp_file = os.path.join(project_dir, 'sync_file.tmp')
        if is_distributed:
            if rank == 0:
                LOGGER.info(f"Create sync temp file at path {sync_tmp_file}")
                os.mknod(sync_tmp_file)
            synchronize()
            # Merge multiple results files
            if rank == 0:
                LOGGER.info("Merge detection results...")
                pred_json_path, merged_results = merge_json(project_dir, prefix=w)
        try:
            if rank == 0 and (opt.result_view or opt.recommend_threshold):
                LOGGER.info("Start visualization result.")
                view_result(anno_json, pred_json_path, data["val"], score_threshold=None,
                            recommend_threshold=opt.recommend_threshold)
                LOGGER.info("Visualization result completed.")
        except Exception as e:
            LOGGER.exception("Failed when visualize evaluation result.")

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            if rank == 0:
                LOGGER.info("Start evaluating mAP...")
                coco_result = coco_eval(anno_json,
                                        merged_results if is_distributed else metric_stats.pred_json,
                                        dataset, is_coco)
                LOGGER.info("Finish evaluating mAP.")
                LOGGER.info(f"\nCOCO mAP:\n{coco_result.stats_str}")
                if os.path.exists(sync_tmp_file):
                    LOGGER.info(f"Delete sync temp file at path {sync_tmp_file}")
                    os.remove(sync_tmp_file)
            else:
                LOGGER.info(f"Waiting for rank [0] device...")
                while os.path.exists(sync_tmp_file):
                    time.sleep(1)
                LOGGER.info(f"Rank [{rank}] continue executing.")
        except Exception as e:
            LOGGER.exception("Exception when running pycocotools")

    # Return results
    if not cfg.training:
        s = f"\n{len(glob.glob(os.path.join(cfg.save_dir, 'labels/*.txt')))} labels saved to " \
            f"{os.path.join(cfg.save_dir, 'labels')}" if cfg.save_txt else ''
        LOGGER.info(f"Results saved to {cfg.save_dir}, {s}")
        with open("class_map.txt", "w") as file:
            file.write(f"COCO map:\n{coco_result.stats_str}\n")
            if coco_result.category_stats_strs:
                for idx, category_str in enumerate(coco_result.category_stats_strs):
                    file.write(f"\nclass {data['names'][idx]}:\n{category_str}\n")
    maps = np.zeros(cfg.nc) + coco_result.get_map()
    for i, c in enumerate(metric_stats.ap_class):
        maps[c] = metric_stats.ap[i]

    model.set_train()
    metric_stats.set_loss(loss / per_epoch_size)
    return metric_stats, maps, speed, coco_result


def main():
    parser = get_args_test()
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
    print(opt)

    ms_mode = ms.GRAPH_MODE if opt.ms_mode == "graph" else ms.PYNATIVE_MODE
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target=opt.device_target)
    # ms.set_context(pynative_synchronize=True)
    context.set_context(mode=ms_mode, device_target=opt.device_target)
    if opt.device_target == "Ascend":
        device_id = int(os.getenv('DEVICE_ID', 0))
        context.set_context(device_id=device_id)
    rank, rank_size, parallel_mode = 0, 1, ParallelMode.STAND_ALONE
    # Distribute Test
    if opt.is_distributed:
        init()
        rank, rank_size, parallel_mode = get_rank(), get_group_size(), ParallelMode.DATA_PARALLEL
    context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=rank_size)
    opt.total_batch_size = opt.batch_size
    opt.rank_size = rank_size
    opt.rank = rank
    if rank_size > 1:
        assert opt.batch_size % opt.rank_size == 0, '--batch-size must be multiple of device count'
        opt.batch_size = opt.total_batch_size // opt.rank_size
    if opt.task in ('train', 'val', 'test'):  # run normally
        print("opt:", opt)
        test(opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment,
             opt.verbose,
             save_txt=opt.save_txt | opt.save_hybrid,
             save_hybrid=opt.save_hybrid,
             save_conf=opt.save_conf,
             trace=not opt.no_trace,
             rect=opt.rect,
             plots=not opt.noplots,
             half_precision=False,
             v5_metric=opt.v5_metric,
             is_distributed=opt.is_distributed,
             rank=opt.rank,
             rank_size=opt.rank_size,
             opt=opt)

    elif opt.task == 'speed':  # speed benchmarks
        test(opt.data, opt.weights, opt.batch_size, opt.img_size, 0.25, 0.45,
             save_json=False, plots=False, half_precision=False, v5_metric=opt.v5_metric)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python test.py --task study --data coco.yaml --iou 0.65 --weights yolov5.ckpt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        f = f'study_{Path(opt.data).stem}_{Path(opt.weights).stem}.txt'  # filename to save to
        y = []  # y axis
        for i in x:  # img-size
            print(f'\nRunning {f} point {i}...')
            metric_stats, _, speed, _ = test(opt.data, opt.weights, opt.batch_size, i,
                                             opt.conf_thres, opt.iou_thres, opt.save_json,
                                             plots=False, half_precision=False, v5_metric=opt.v5_metric)
            y.append(tuple(metric_stats)[:7] + speed)  # results and times
        np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot


if __name__ == '__main__':
    main()
