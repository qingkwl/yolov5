import copy
import glob
import os
import time
from pathlib import Path
import yaml
from argparse import Namespace
from threading import Thread
from typing import Optional
import json
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ms_dataset
from mindspore import save_checkpoint
from mindspore import Tensor
from mindspore.train.callback import Callback
from mindspore._checkparam import Validator

from src.metrics import ConfusionMatrix, non_max_suppression, scale_coords, ap_per_class
from src.general import coco80_to_coco91_class, check_img_size, xyxy2xywh, xywh2xyxy, \
    colorstr, box_iou, Synchronize
from src.network.yolo import Model
from src.dataset import create_dataloader, LoadImagesAndLabels
from src.plots import plot_images, output_to_target
from test import test
try:
    from third_party.fast_coco.fast_coco_eval_api import Fast_COCOeval as COCOeval
    print("[INFO] Use third party coco eval api to speed up mAP calculation.")
except ImportError:
    from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO


class BaseEvaluator:
    def __init__(self, model: nn.Cell, dataloader: ms_dataset.Dataset,
                 workdir: str, rank: int, rank_size: int, is_distributed: bool):
        self.model = model
        self.dataloader = dataloader
        self.workdir = workdir  # Directory path to save prediction results
        self.rank = rank
        self.rank_size = rank_size
        self.is_distributed = is_distributed
        self.synchronize = None if not is_distributed else Synchronize(rank_size)
        self.results = []

    def preprocess(self):
        pass

    def inference(self):
        pass

    def merge_json(self):
        pass

    def postprocess(self):
        pass

    def evaluate(self):
        pass

    def reset(self):
        self.results.clear()


class COCOEvaluator:
    def __init__(self, gt_anno_path: str):
        self.gt_anno_path = gt_anno_path
        self.ground_truth = COCO(gt_anno_path)
        self.evaluator = None

    def evaluate(self, dt_anno_path):
        predictions = self.ground_truth.loadRes(dt_anno_path)
        self.evaluator = COCOeval(self.ground_truth, predictions, 'bbox')
        self.evaluator.evaluate()
        self.evaluator.accumulate()
        self.evaluator.summarize()
        return self.evaluator.stats


class YOLOv5COCOEvaluator(COCOEvaluator):
    def __init__(self, gt_anno_path: str):
        super(YOLOv5COCOEvaluator, self).__init__(gt_anno_path)

    def evaluate(self, dt_anno_path: str, dataset: Optional[LoadImagesAndLabels] = None):
        predictions = self.ground_truth.loadRes(dt_anno_path)
        self.evaluator = COCOeval(self.ground_truth, predictions, 'bbox')
        if dataset is not None:
            self.evaluator.params.imgIds = [int(Path(x).stem) for x in dataset.img_files]   # image IDs to evaluate
        self.evaluator.evaluate()
        self.evaluator.accumulate()
        self.evaluator.summarize()
        return self.evaluator.stats


class RunContext:
    def __init__(self):
        self.precision = 0.
        self.recall = 0.
        self.f1 = 0.
        self.mean_precision = 0.
        self.mean_recall = 0.
        self.mean_ap50 = 0.
        self.mean_ap = 0.
        self.seen = 0
        self.model_duration = 0.
        self.nms_duration = 0.
        self.loss = np.zeros(3)
        self.jdict = []
        self.stats = []
        self.ap = []
        self.ap_class = []


class YOLOv5Evaluator:
    def __init__(self, opt: Namespace):
        self.opt = opt
        self.data = None
        self.weights = None
        self.batch_size = 32
        self.img_size = 640
        self.conf_threshold = 0.001
        self.iou_threshold = 0.6  # for NMS
        self.save_json = False
        self.single_cls = False
        self.augment = False
        self.verbose = False
        self.model = None
        self.dataloader = None
        self.dataset = None
        self.save_dir = Path('')  # for saving images
        self.save_txt = False  # for auto-labelling
        self.save_hybrid = False  # for hybrid auto-labelling
        self.save_conf = False  # save auto-label confidences
        self.plots = False
        self.compute_loss = None
        self.half_precision = False
        self.trace = False
        self.is_coco = False
        self.v5_metric = False
        self.is_distributed = False
        self.rank = 0
        self.rank_size = 1
        self.num_class = 1
        self.iouv = np.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = np.prod(self.iouv.shape)
        self.synchronize = None
        self.is_training = False
        self.per_epoch_size = 32
        self.best_mAP = 0.
        self.coco_evaluator = None
        self.coco91class = coco80_to_coco91_class()
        self.set_parameters()

    def set_parameters(self):
        if self.opt.task in ('train', 'val', 'test'):
            self.data = self.opt.data
            self.weights = self.opt.weights
            self.batch_size = self.opt.batch_size
            self.img_size = self.opt.img_size
            self.conf_threshold = self.opt.conf_thres
            self.iou_threshold = self.opt.iou_thres,
            self.save_json = self.opt.save_json
            self.single_cls = self.opt.single_cls
            self.augment = self.opt.augment
            self.verbose = self.opt.verbose
            self.save_txt = self.opt.save_txt | self.opt.save_hybrid
            self.save_hybrid = self.opt.save_hybrid
            self.save_conf = self.opt.save_conf
            self.trace = not self.opt.no_trace
            self.v5_metric = self.opt.v5_metric
            self.is_distributed = self.opt.is_distributed
            self.rank = self.opt.rank
            self.rank_size = self.opt.rank_size
        elif self.opt.task == 'speed':
            self.data = self.opt.data
            self.weights = self.opt.weights
            self.batch_size = self.opt.batch_size
            self.img_size = self.opt.img_size
            self.conf_threshold = 0.25
            self.iou_threshold = 0.45
            self.v5_metric = self.opt.v5_metric

    def preprocess(self):
        # Configure
        if isinstance(self.data, str):
            self.is_coco = self.data.endswith('coco.yaml')
            with open(self.data) as f:
                self.data = yaml.load(f, Loader=yaml.SafeLoader)
        self.num_class = 1 if self.single_cls else int(self.data['nc'])
        self.synchronize = Synchronize(self.rank_size) if self.is_distributed else None
        # self.is_training = self.model is not None
        self.set_model()
        self.set_dataloader()
        # Half
        if self.half_precision:
            ms.amp.auto_mixed_precision(self.model, amp_level="O2")
        if self.v5_metric:
            print("Testing with YOLOv5 AP metric...")

    def set_model(self):
        if self.model is not None:
            return
        self.save_dir = os.path.join(self.opt.project, f"save_dir_{self.rank}")
        os.makedirs(os.path.join(self.save_dir, f"labels_{self.rank}"), exist_ok=False)
        # Load model
        # Hyperparameters
        with open(self.opt.hyp) as f:
            hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
        self.model = Model(self.opt.cfg, ch=3, nc=self.num_class, anchors=hyp.get('anchors'), sync_bn=False)
        ckpt_path = self.weights
        param_dict = ms.load_checkpoint(ckpt_path)
        ms.load_param_into_net(self.model, param_dict)
        print(f"load ckpt from \"{ckpt_path}\" success.")
        grid_size = max(int(ops.cast(self.model.stride, ms.float16).max()), 32)  # grid size (max stride)
        self.img_size = check_img_size(self.img_size, s=grid_size)  # check img_size

    def set_dataloader(self):
        grid_size = max(int(ops.cast(self.model.stride, ms.float16).max()), 32)
        # Dataloader
        if self.dataloader is None or self.dataset is None:
            task = self.opt.task if self.opt.task in (
            'train', 'val', 'test') else 'val'  # path to train/val/test images
            self.dataloader, self.dataset, self.per_epoch_size = create_dataloader(
                self.data[task], self.img_size,
                self.batch_size, grid_size, self.opt,
                epoch_size=1, pad=0.5, rect=False,
                rank=self.rank,
                rank_size=self.rank_size,
                num_parallel_workers=4 if self.rank_size > 1 else 8,
                shuffle=False,
                drop_remainder=False,
                prefix=colorstr(f'{task}: ')
            )
            assert self.per_epoch_size == self.dataloader.get_dataset_size()
            self.dataloader = self.dataloader.create_dict_iterator(output_numpy=True, num_epochs=1)
            print(f"Test create dataset success, epoch size {self.per_epoch_size}.")
        else:
            assert self.dataset is not None
            assert self.dataloader is not None
            self.per_epoch_size = self.dataloader.get_dataset_size()
            self.dataloader = self.dataloader.create_dict_iterator(output_numpy=True, num_epochs=1)

    def inference(self):
        self.model.set_train(False)
        # seen = 0
        confusion_matrix = ConfusionMatrix(nc=self.num_class)
        names = {k: v for k, v in enumerate(self.model.names if hasattr(self.model, 'names')
                                            else self.model.module.names)}
        # precision, recall, f1, mean_precision, mean_recall, mean_ap50, mean_ap = 0., 0., 0., 0., 0., 0., 0.
        run_context = RunContext()
        # model_duration, nms_duration = 0., 0.
        # loss = np.zeros(3)
        # jdict, stats, ap, ap_class = [], [], [], []
        start_time = time.time()
        for batch_i, meta_data in enumerate(self.dataloader):
            img, targets, paths, shapes = meta_data["img"], meta_data["label_out"], \
                                          meta_data["img_files"], meta_data["shapes"]
            dtype = ms.float16 if self.half_precision else ms.float32
            img = img.astype(np.float) / 255.0  # 0 - 255 to 0.0 - 1.0
            img_tensor = ms.Tensor(img, dtype)
            targets_tensor = ms.Tensor(targets, dtype)
            targets = targets.reshape((-1, 6))
            targets = targets[targets[:, 1] >= 0]
            batch_size, _, height, width = img.shape  # batch size, channels, height, width

            # Run model
            time_point = time.time()

            pred_out, train_out = self.model(img_tensor, augment=self.augment)  # inference and training outputs
            run_context.model_duration += time.time() - time_point

            # Compute loss # metric with common loss
            if self.compute_loss is not None:
                run_context.loss += self.compute_loss(train_out, targets_tensor)[1][:3].asnumpy()  # box, obj, cls

            # Run NMS
            targets[:, 2:] *= np.array([width, height, width, height], targets.dtype)  # to pixels
            # for auto labelling
            labels = [targets[targets[:, 0] == i, 1:] for i in range(batch_size)] if self.save_hybrid else []
            time_point = time.time()
            out = pred_out.asnumpy()
            out = non_max_suppression(out, conf_thres=self.conf_threshold, iou_thres=self.iou_threshold,
                                      labels=labels, multi_label=True)
            run_context.nms_duration += time.time() - time_point

            # Statistics per image
            self.compute_img_statistics(confusion_matrix, img, targets, paths, shapes, out, run_context)

            # Plot images
            if self.plots and batch_i < 3:
                f = os.path.join(self.save_dir, f'test_batch{batch_i}_labels.jpg')  # labels
                Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
                f = os.path.join(self.save_dir, f'test_batch{batch_i}_pred.jpg')  # predictions
                Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

            print(f"Test step {batch_i + 1}/{self.per_epoch_size}, cost time {time.time() - start_time:.2f}s", flush=True)

            start_time = time.time()

        # Compute statistics
        stats = [np.concatenate(x, 0) for x in zip(*run_context.stats)]  # to numpy
        if len(stats) and stats[0].any():
            tp, fp, run_context.precision, run_context.recall, run_context.f1, ap, ap_class = ap_per_class(
                *stats, plot=self.plots,
                save_dir=self.save_dir, names=names
            )
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            run_context.mean_precision, run_context.mean_recall, run_context.mean_ap50, run_context.mean_ap = run_context.precision.mean(), run_context.recall.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(int), minlength=self.num_class)  # number of targets per class

        # Print results
        pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
        # print(pf % ('all', seen, nt.sum(), mean_precision, mean_recall, mean_ap50, mean_ap), flush=True)
        print(pf % ('all', run_context.seen, nt.sum(), run_context.mean_precision, run_context.mean_recall, run_context.mean_ap50, run_context.mean_ap), flush=True)

        # Print results per class
        if (self.verbose or (self.num_class < 50 and not self.is_training)) and self.num_class > 1 and len(stats):
            for i, c in enumerate(ap_class):
                print(pf % (names[c], run_context.seen, nt[c], run_context.precision[i], run_context.recall[i], ap50[i], ap[i]))

        # Print speeds
        duration = tuple(x / run_context.seen * 1E3 for x in (run_context.model_duration, run_context.nms_duration, run_context.model_duration + run_context.nms_duration)) + (self.img_size, self.img_size, self.batch_size)  # tuple
        if not self.is_training:
            print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % duration)

        # Plots
        if self.plots:
            confusion_matrix.plot(save_dir=self.save_dir, names=list(names.values()))

        # Save JSON
        if self.save_json and len(run_context.jdict):
            anno_json, pred_json = self.save_json_file(run_context.jdict)
            run_context.mean_ap, run_context.mean_ap50 = self.compute_map(anno_json, pred_json) # update results (mAP@0.5:0.95, mAP@0.5)

        # Return results
        if not self.is_training:
            s = f"\n{len(glob.glob(os.path.join(self.save_dir, 'labels/*.txt')))} labels saved to " \
                f"{os.path.join(self.save_dir, 'labels')}" if self.save_txt else ''
            print(f"Results saved to {self.save_dir}, {s}", flush=True)
        maps = np.zeros(self.num_class) + run_context.mean_ap
        for i, c in enumerate(ap_class):
            maps[c] = ap[i]
        if self.is_training:
            # save best result
            if run_context. mean_ap > self.best_mAP:
                # update best checkpoint
                best_ckpt_path = os.path.join(self.save_dir, "best.ckpt")
                print("Save best checkpoint...", flush=True)
                save_checkpoint(self.model, best_ckpt_path)
                self.best_mAP = run_context.mean_ap
        self.model.set_train()
        return (run_context.mean_precision, run_context.mean_recall, run_context.mean_ap50, run_context.mean_ap, *(run_context.loss / self.per_epoch_size).tolist()), maps, time_point

    def compute_img_statistics(self, confusion_matrix, img, targets, paths, shapes, out, run_context):
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(str(paths[si]))
            # seen += 1
            run_context.seen += 1

            if len(pred) == 0:
                if nl:
                    run_context.stats.append((np.zeros((0, self.niou), dtype=np.bool),
                                              np.zeros(0, dtype=pred.dtype),
                                              np.zeros(0, dtype=pred.dtype),
                                              tcls))
                continue

            # Predictions
            predn = np.copy(pred)
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0, :], shapes[si][1:, :])  # native-space pred

            # Append to text file
            if self.save_txt:
                gn = np.array(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(np.array(xyxy).reshape((1, 4))) / gn).reshape(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                    with open(os.path.join(self.save_dir, 'labels', (path.stem + '.txt')), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # Append to pycocotools JSON dictionary
            if self.save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                box = xyxy2xywh(predn[:, :4])  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    run_context.jdict.append({'image_id': image_id,
                                              'category_id': self.coco91class[int(p[5])] if self.is_coco else int(p[5]),
                                              'bbox': [round(x, 3) for x in b],
                                              'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = np.zeros((pred.shape[0], self.niou), dtype=np.bool)
            if nl:
                detected = []  # target indices
                tcls_np = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0, :], shapes[si][1:, :])  # native-space labels
                if self.plots:
                    confusion_matrix.process_batch(predn, np.concatenate((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in np.unique(tcls_np):
                    # ti = (cls == tcls_np).nonzero(as_tuple=False).view(-1)  # prediction indices
                    # pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices
                    ti = np.nonzero(cls == tcls_np)[0].reshape(-1)  # prediction indices
                    pi = np.nonzero(cls == pred[:, 5])[0].reshape(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        all_ious = box_iou(predn[pi, :4], tbox[ti])
                        ious = all_ious.max(1)  # best ious, indices
                        i = all_ious.argmax(1)

                        # Append detections
                        detected_set = set()
                        for j in (ious > self.iouv[0]).nonzero()[0]:
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > self.iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            run_context.stats.append((correct, pred[:, 4], pred[:, 5], tcls))

    def save_json_file(self, jdict):
        w = Path(self.weights).stem if self.weights is not None else ''  # weights
        # anno_json = './coco/annotations/instances_val2017.json'  # annotations json
        anno_json = os.path.join(self.data["val"][:-12], "annotations/instances_val2017.json")
        pred_json = os.path.join(self.save_dir, f"{w}_predictions_{self.rank}.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)
        if self.synchronize is not None:
            self.synchronize()
            # Merge multiple results files
            merged_json = os.path.join(self.opt.project, f"{w}_predictions_merged.json")
            merged_result = []
            for json_file in Path(self.opt.project).rglob("*.json"):
                with open(json_file, "r") as file_handler:
                    merged_result.extend(json.load(file_handler))
            with open(merged_json, "w") as file_handler:
                json.dump(merged_result, file_handler)
            pred_json = merged_json
        return anno_json, pred_json

    def compute_map(self, anno_json, pred_json):
        print("Start computing mAP...", flush=True)
        if self.coco_evaluator is None:
            self.coco_evaluator = YOLOv5COCOEvaluator(anno_json)
        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            return self.coco_evaluator.evaluate(pred_json, dataset=self.dataset if self.is_coco else None)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')


class YOLOv5Evaluation(Callback):
    def __init__(self, opt, val_dataloader, val_dataset, train_epoch_size):
        self.opt = opt
        self.val_dataloader = val_dataloader
        self.val_dataset = val_dataset
        self.train_epoch_size = train_epoch_size
        self.hyper_map = ops.HyperMap()
        self.best_map = 0.
        self.weights_dir = os.path.join(self.opt.save_dir, "weights")

    def is_eval_epoch(self, cur_epoch, cur_step):
        return (cur_epoch >= self.opt.eval_start_epoch) and \
               (cur_step % (self.opt.eval_epoch_interval * self.train_epoch_size)) == 0

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        cur_step = cb_params.cur_step_num
        if not self.opt.run_eval or not self.is_eval_epoch(cur_epoch, cur_step):
            return

        model = cb_params.train_network.network.yolo_net
        ema = cb_params.train_network.network.ema
        infer_model = copy.deepcopy(model) if self.opt.ema else model
        print("[INFO] Evaluating...", flush=True)
        model.set_train(False)
        if self.opt.ema:
            print("[INFO] ema parameter update", flush=True)
            self.hyper_map(ops.assign, ms.ParameterTuple(list(model.get_parameters())), ema.ema_weights)
        info, maps, t = test(self.opt.data,
                             self.opt.weights,
                             self.opt.batch_size,
                             self.opt.img_size,
                             self.opt.conf_thres,
                             self.opt.iou_thres,
                             self.opt.save_json,
                             self.opt.single_cls,
                             self.opt.augment,
                             self.opt.verbose,
                             model=infer_model,
                             dataloader=self.val_dataloader,
                             dataset=self.val_dataset,
                             save_txt=self.opt.save_txt | self.opt.save_hybrid,
                             save_hybrid=self.opt.save_hybrid,
                             save_conf=self.opt.save_conf,
                             trace=not self.opt.no_trace,
                             plots=False,
                             half_precision=False,
                             v5_metric=self.opt.v5_metric,
                             is_distributed=self.opt.is_distributed,
                             rank=self.opt.rank,
                             rank_size=self.opt.rank_size,
                             opt=self.opt)
        infer_model.set_train(True)
        if (self.opt.rank % 8 == 0) and (info[3] > self.best_map):
            self.best_map = info[3]
            print(f"[INFO] Best result: Best mAP [{self.best_map}] at epoch [{cur_epoch}]", flush=True)
            # save best checkpoint
            model_name = os.path.basename(self.opt.cfg)[:-5]  # delete ".yaml"
            ckpt_path = os.path.join(self.weights_dir, f"{model_name}_best.ckpt")
            ms.save_checkpoint(model, ckpt_path)
            if ema:
                params_list = []
                for p in ema.ema_weights:
                    _param_dict = {'name': p.name[len("ema."):], 'data': Tensor(p.data.asnumpy())}
                    params_list.append(_param_dict)

                ema_ckpt_path = os.path.join(self.weights_dir, f"EMA_{model_name}_best.ckpt")
                ms.save_checkpoint(params_list, ema_ckpt_path, append_dict={"updates": ema.updates})
            if self.opt.enable_modelarts:
                from src.modelarts import sync_data
                sync_data(ckpt_path, self.opt.train_url + "/weights/" + ckpt_path.split("/")[-1])
                if ema:
                    sync_data(ema_ckpt_path, self.opt.train_url + "/weights/" + ema_ckpt_path.split("/")[-1])


class TimeMonitor(Callback):
    def __init__(self, data_size=None):
        super(TimeMonitor, self).__init__()
        self.data_size = data_size
        self.epoch_time = time.time()

    def epoch_begin(self, run_context):
        """
        Record time at the beginning of epoch.

        Args:
            run_context (RunContext): Context of the process running.  For more details,
                    please refer to :class:`mindspore.RunContext`.
        """
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        """
        Print process cost time at the end of epoch.

        Args:
           run_context (RunContext): Context of the process running.  For more details,
                   please refer to :class:`mindspore.RunContext`.
        """
        epoch_seconds = (time.time() - self.epoch_time) * 1000
        step_size = self.data_size
        cb_params = run_context.original_args()
        mode = cb_params.get("mode", "")
        if hasattr(cb_params, "batch_num"):
            batch_num = cb_params.batch_num
            if isinstance(batch_num, int) and batch_num > 0:
                step_size = cb_params.batch_num
        Validator.check_positive_int(step_size)

        step_seconds = epoch_seconds / step_size
        print("{} epoch time: {:5.3f} ms, per step time: {:5.3f} ms".format
              (mode.title(), epoch_seconds, step_seconds), flush=True)


class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


class LossMonitor(Callback):
    def __init__(self, monitor_args, per_print_times=1):
        super(LossMonitor, self).__init__()
        Validator.check_non_negative_int(per_print_times)
        self._per_print_times = per_print_times
        self._last_print_time = 0
        self.monitor_args = Dict(monitor_args)

    def step_end(self, run_context):
        """
        Print training loss at the end of step.

        Args:
            run_context (RunContext): Include some information of the model.  For more details,
                    please refer to :class:`mindspore.RunContext`.
        """
        cb_params = run_context.original_args()

        cur_epoch_num = cb_params.get("cur_epoch_num", 1)
        loss, loss_item = cb_params.net_outputs

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        # In disaster recovery scenario, the cb_params.cur_step_num may be rollback to previous step
        # and be less than self._last_print_time, so self._last_print_time need to be updated.
        if self._per_print_times != 0 and (cb_params.cur_step_num <= self._last_print_time):
            while cb_params.cur_step_num <= self._last_print_time:
                self._last_print_time -= \
                    max(self._per_print_times, cb_params.batch_num if cb_params.dataset_sink_mode else 1)

        if self._per_print_times != 0 and (cb_params.cur_step_num - self._last_print_time) >= self._per_print_times:
            self._last_print_time = cb_params.cur_step_num
            acc_step = cur_step_in_epoch * cur_epoch_num - 1
            if self.monitor_args is None:
                print(f"epoch {cur_epoch_num}, Step {cur_step_in_epoch}, loss: {loss.asnumpy():.4f}, "
                      f"lbox: {loss_item[0].asnumpy():.4f}, lobj: {loss_item[1].asnumpy():.4f}, "
                      f"lcls: {loss_item[2].asnumpy():.4f}, ", flush=True)
            else:
                print(f"epoch {self.monitor_args.total_epoch}/{cur_epoch_num}, "
                      f"Step {self.monitor_args.per_epoch_size}/{cur_step_in_epoch}, "
                      f"loss: {loss.asnumpy():.4f}, lbox: {loss_item[0].asnumpy():.4f}, "
                      f"lobj: {loss_item[1].asnumpy():.4f}, lcls: {loss_item[2].asnumpy():.4f}, "
                      f"cur_lr: [{self.monitor_args.lr_pg0[acc_step]:.8f}, {self.monitor_args.lr_pg1[acc_step]:.8f}, "
                      f"{self.monitor_args.lr_pg2[acc_step]:.8f}], ", flush=True)

    def on_train_epoch_end(self, run_context):
        """
        When LossMoniter used in `model.fit`, print eval metrics at the end of epoch if current epoch
        should do evaluation.

        Args:
            run_context (RunContext): Include some information of the model. For more details,
                    please refer to :class:`mindspore.RunContext`.
        """
        cb_params = run_context.original_args()
        metrics = cb_params.get("metrics")
        if metrics:
            print("Eval result: epoch %d, metrics: %s" % (cb_params.cur_epoch_num, metrics))


class YoloCheckpoint(Callback):
    def __init__(self, opt, wdir):
        self.opt = opt
        self.wdir = wdir
        self.model_name = os.path.basename(opt.cfg)[:-5]  # delete ".yaml"

    def epoch_end(self, run_context):
        """
        Called after each epoch finished.
        A backwards compatibility alias for `on_train_epoch_end` and `on_eval_epoch_end`.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        cb_params = run_context.original_args()
        model = cb_params.train_network.network.yolo_net
        ema = cb_params.train_network.network.ema
        cur_epoch_num = cb_params.get("cur_epoch_num", 1)
        ckpt_path = os.path.join(self.wdir, f"{self.model_name}_{cur_epoch_num}.ckpt")
        ms.save_checkpoint(model, ckpt_path)
        if ema:
            params_list = []
            for p in ema.ema_weights:
                _param_dict = {'name': p.name[len("ema."):], 'data': Tensor(p.data.asnumpy())}
                params_list.append(_param_dict)

            ema_ckpt_path = os.path.join(self.wdir, f"EMA_{self.model_name}_{cur_epoch_num}.ckpt")
            ms.save_checkpoint(params_list, ema_ckpt_path, append_dict={"updates": ema.updates})

