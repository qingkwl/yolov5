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

import copy
import os
import math
import random
import time
from collections import deque
from contextlib import nullcontext
from pathlib import Path

import yaml
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor, context
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore.context import ParallelMode
from mindspore.ops import functional as F
from mindspore.profiler.profiling import Profiler

from config.args import get_args_train
from val import EvalManager, MetricStatistics, COCOResult
from src.autoanchor import check_anchors, check_anchor_order
from src.boost import build_train_network
from src.dataset import create_dataloader
from src.general import (check_file, check_img_size, colorstr, increment_path,
                         labels_to_class_weights, LOGGER, process_dataset_cfg, empty,
                         WRITE_FLAGS, FILE_MODE, SynchronizeManager)
from src.network.common import EMA
from src.network.loss import ComputeLoss
from src.network.yolo import Model
from src.optimizer import YoloMomentum, get_group_param, get_lr


def set_seed(seed=2):
    np.random.seed(seed)
    random.seed(seed)
    ms.set_seed(seed)


# ----------------------------------------------------------------------------------------------------
def detect_overflow(epoch, cur_step, grads):
    for i, grad in enumerate(grads):
        tmp = grad.asnumpy()
        if np.isinf(tmp).any() or np.isnan(tmp).any():
            print(f"grad_{i}", flush=True)
            print(f"Epoch: {epoch}, Step: {cur_step} grad_{i} overflow, this step drop. ", flush=True)
            return True
    return False


def load_checkpoint_to_yolo(model, ckpt_path, resume):
    trainable_params = {p.name: p.asnumpy() for p in model.get_parameters()}
    param_dict = ms.load_checkpoint(ckpt_path)
    new_params = {}
    ema_prefix = "ema.ema."
    for k, v in param_dict.items():
        if not k.startswith("model.") and not k.startswith("updates") and not k.startswith(ema_prefix):
            continue

        k = k[len(ema_prefix):] if ema_prefix in k else k
        if k in trainable_params:
            if v.shape != trainable_params[k].shape:
                print(f"[WARNING] Filter checkpoint parameter: {k}", flush=True)
                continue
            new_params[k] = v
        else:
            print(f"[WARNING] Checkpoint parameter: {k} not in model", flush=True)

    ms.load_param_into_net(model, new_params)
    print(f"load ckpt from \"{ckpt_path}\" success.", flush=True)
    resume_epoch = 0
    if resume and 'epoch' in param_dict:
        resume_epoch = int(param_dict['epoch'])
        print(f"[INFO] Resume training from epoch {resume_epoch}", flush=True)
    return resume_epoch


@ops.constexpr
def _get_new_size(img_shape, gs, imgsz):
    sz = random.randrange(int(imgsz * 0.5), int(imgsz * 1.5 + gs)) // gs * gs  # size
    sf = sz / max(img_shape[2:])  # scale factor
    new_size = img_shape
    if sf != 1:
        # new size (stretched to gs-multiple)
        # Use tuple because nn.interpolate only supports tuple `sizes` parameter must be tuple
        new_size = tuple(math.ceil(x * sf / gs) * gs for x in img_shape[2:])
    return new_size


def create_train_network(model, compute_loss, ema, optimizer, loss_scaler=None,
                         sens=1.0, enable_clip_grad=True, opt=None, gs=None, imgsz=None):
    class NetworkWithLoss(nn.Cell):
        def __init__(self, model, compute_loss, opt):
            super(NetworkWithLoss, self).__init__()
            self.model = model
            self.compute_loss = compute_loss
            self.rank_size = opt.rank_size
            self.lbox_loss = Parameter(Tensor(0.0, ms.float32), requires_grad=False, name="lbox_loss")
            self.lobj_loss = Parameter(Tensor(0.0, ms.float32), requires_grad=False, name="lobj_loss")
            self.lcls_loss = Parameter(Tensor(0.0, ms.float32), requires_grad=False, name="lcls_loss")
            self.multi_scale = opt.multi_scale if hasattr(opt, 'multi_scale') else False
            self.gs = gs
            self.imgsz = imgsz

        def construct(self, x, label, sizes=None):
            x /= 255.0
            if self.multi_scale and self.training:
                x = ops.interpolate(x, sizes=_get_new_size(x.shape, self.gs, self.imgsz),
                                    coordinate_transformation_mode="asymmetric", mode="bilinear")
            pred = self.model(x)
            loss, loss_items = self.compute_loss(pred, label)
            loss_items = ops.stop_gradient(loss_items)
            loss *= self.rank_size
            loss = F.depend(loss, ops.assign(self.lbox_loss, loss_items[0]))
            loss = F.depend(loss, ops.assign(self.lobj_loss, loss_items[1]))
            loss = F.depend(loss, ops.assign(self.lcls_loss, loss_items[2]))
            return loss

    LOGGER.info(f"rank_size: {opt.rank_size}")
    net_with_loss = NetworkWithLoss(model, compute_loss, opt)
    train_step = build_train_network(network=net_with_loss, ema=ema, optimizer=optimizer,
                                     level='O0', boost_level='O1', amp_loss_scaler=loss_scaler,
                                     sens=sens, enable_clip_grad=enable_clip_grad)
    return train_step


def val(opt, model, ema, infer_model, val_dataloader, val_dataset, cur_epoch):
    LOGGER.info("Evaluating...")
    param_dict = {}
    if opt.ema:
        LOGGER.info("ema parameter update")
        for p in ema.ema_weights:
            name = p.name[len("ema."):]
            param_dict[name] = p.data
    else:
        for p in model.get_parameters():
            name = p.name
            param_dict[name] = p.data

    ms.load_param_into_net(infer_model, param_dict)
    del param_dict
    infer_model.set_train(False)
    test_manager = EvalManager(opt)
    val_result = test_manager.eval(infer_model, val_dataset, val_dataloader, cur_epoch)
    infer_model.set_train(True)
    # Return corresponding result according to config
    if opt.metric == 'yolo':
        return val_result.metric_stats
    else:
        return val_result.coco_result



def save_ema(ema, ema_ckpt_path, append_dict=None):
    params_list = []
    for p in ema.ema_weights:
        _param_dict = {'name': p.name[len("ema."):], 'data': Tensor(p.data.asnumpy())}
        params_list.append(_param_dict)
    ms.save_checkpoint(params_list, ema_ckpt_path, append_dict=append_dict)


class CheckpointQueue:
    def __init__(self, max_ckpt_num):
        self.max_ckpt_num = max_ckpt_num
        self.ckpt_queue = deque()

    def append(self, ckpt_path):
        self.ckpt_queue.append(ckpt_path)
        if len(self.ckpt_queue) > self.max_ckpt_num:
            ckpt_to_delete = self.ckpt_queue.popleft()
            os.remove(ckpt_to_delete)


class TrainManager:
    def __init__(self, hyp, opt):
        self.hyp = hyp
        self.opt = opt

        self.best_map = 0.
        self.data_cfg = None
        self.weight_dir = os.path.join(self.opt.save_dir, "weights")

    @staticmethod
    def _write_map_result(coco_result, map_str_path, names):
        with os.fdopen(os.open(map_str_path, WRITE_FLAGS, FILE_MODE), 'w') as file:
            if isinstance(coco_result, COCOResult):
                file.write(f"COCO API:\n{coco_result.stats_str}\n")
                if coco_result.category_stats_strs is not None:
                    for idx, category_str in enumerate(coco_result.category_stats_strs):
                        file.write(f"\nclass {names[idx]}:\n{category_str}\n")
            elif isinstance(coco_result, MetricStatistics):
                title = ('{:22s}' + '{:11s}' * 6).format('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
                pf = '{:<20s}' + '{:<12d}' * 2 + '{:<12.3g}' * 4  # print format
                file.write(f"{title}\n")
                file.write(pf.format('all', coco_result.seen, coco_result.nt.sum(), *coco_result.get_mean_stats()))
                file.write("\n")
                for i, c in enumerate(coco_result.ap_cls):
                    # Class     Images  Instances          P          R      mAP50   mAP50-95:
                    file.write(pf.format(names[c], coco_result.seen, coco_result.nt[c],
                                         *coco_result.get_ap_per_class(i)) + '\n')
            else:
                raise TypeError(f"Not supported coco_result type: {type(coco_result)}")


    def train(self):
        set_seed()
        opt = self.opt
        hyp = self.hyp
        self._modelarts_sync(opt.data_url, opt.data_dir)

        self.data_cfg = self.get_data_cfg()
        num_cls = self.data_cfg['nc']

        # Directories
        os.makedirs(self.weight_dir, exist_ok=True)
        # Save run settings
        self.dump_cfg()
        self._modelarts_sync(opt.save_dir, opt.train_url)

        # Model
        sync_bn = opt.sync_bn and context.get_context("device_target") == "Ascend" and opt.rank_size > 1
        # Create Model
        model = Model(opt.cfg, ch=3, nc=num_cls, anchors=hyp.get('anchors'), sync_bn=sync_bn, opt=opt, hyp=hyp)
        model.to_float(ms.float16)
        ema = EMA(model) if opt.ema else None

        pretrained = opt.weights.endswith('.ckpt')
        resume_epoch = 0
        if pretrained:
            resume_epoch = load_checkpoint_to_yolo(model, opt.weights, opt.resume)
            ema.clone_from_model()
            LOGGER.warning("ema_weight not exist, default pretrain weight is currently used.")

        # Freeze
        model = self.freeze_layer(model)

        # Image sizes
        gs = max(int(model.stride.asnumpy().max()), 32)  # grid size (max stride)
        nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
        imgsz, _ = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples
        train_epoch_size = 1 if opt.optimizer == "thor" else opt.epochs - resume_epoch
        dataloader, dataset, per_epoch_size = self.get_dataset(model, train_epoch_size, mode="train")
        infer_model, val_dataloader, val_dataset = None, None, None
        if opt.save_checkpoint or opt.run_eval:
            infer_model = copy.deepcopy(model) if opt.ema else model
            val_dataloader, val_dataset, _ = self.get_dataset(model, epoch_size=1, mode="val")

        if not opt.resume and not opt.noautoanchor:
            anchors_path = Path(opt.save_dir) / 'anchors.npy'
            with SynchronizeManager(opt.rank % 8, min(8, opt.rank_size), opt.distributed_train, opt.save_dir):
                anchors = check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
                if anchors is not None:
                    # Save to save_dir
                    np.save(str(anchors_path.resolve()), anchors)
            if anchors_path.exists():
                # Each rank load anchors
                anchors = np.load(str(anchors_path.resolve()))
                model.anchors[:] = ms.Tensor(anchors.reshape(model.anchors.shape), dtype=ms.float32)
                check_anchor_order(model)
                info = f'Done ✅ (optional: update model *.yaml to use these anchors in the future)'
            else:
                info = f'Done ⚠️ (original anchors better than new anchors, proceeding with original anchors)'
            LOGGER.info(info)

        mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
        assert mlc < num_cls, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g'\
                              % (mlc, num_cls, opt.data, num_cls - 1)

        # Optimizer
        nbs = 64  # nominal batch size
        accumulate = max(round(nbs / opt.total_batch_size), 1)  # accumulate loss before optimizing
        hyp['weight_decay'] *= opt.total_batch_size * accumulate / nbs  # scale weight_decay
        LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")
        optimizer = self.get_optimizer(model, per_epoch_size, resume_epoch)

        # Model parameters
        model = self._configure_model_params(dataset, imgsz, model, nl)

        # Build train process function
        # amp
        ms.amp.auto_mixed_precision(model, amp_level=opt.ms_amp_level)
        compute_loss = ComputeLoss(model)  # init loss class
        ms.amp.auto_mixed_precision(compute_loss, amp_level=opt.ms_amp_level)
        loss_scaler = self.get_loss_scaler()
        train_step = self.get_train_step(compute_loss, ema, model, optimizer,
                                         gs=gs, imgsz=imgsz, loss_scaler=loss_scaler)
        model.set_train(True)
        optimizer.set_train(True)
        run_profiler_epoch = 2
        ema_ckpt_queue = CheckpointQueue(opt.max_ckpt_num)
        ckpt_queue = CheckpointQueue(opt.max_ckpt_num)

        data_size = dataloader.get_dataset_size()
        jit = opt.ms_mode.lower() == "graph"
        sink_process = ms.data_sink(train_step, dataloader, steps=data_size * opt.epochs, sink_size=data_size, jit=jit)

        summary_dir = os.path.join(opt.save_dir, opt.summary_dir, f"rank_{opt.rank}")
        steps_per_epoch = data_size
        with ms.SummaryRecord(summary_dir) if opt.summary else nullcontext() as summary_record:
            for cur_epoch in range(resume_epoch, opt.epochs):
                cur_epoch = cur_epoch + 1
                start_train_time = time.time()
                loss = sink_process()
                end_train_time = time.time()
                step_time = end_train_time - start_train_time
                LOGGER.info(f"Epoch {opt.epochs - resume_epoch}/{cur_epoch}, step {data_size}, "
                            f"epoch time {step_time * 1000:.2f} ms, "
                            f"step time {step_time * 1000 / data_size:.2f} ms, "
                            f"loss: {loss.asnumpy() / opt.batch_size:.4f}, "
                            f"lbox loss: {train_step.network.lbox_loss.asnumpy():.4f}, "
                            f"lobj loss: {train_step.network.lobj_loss.asnumpy():.4f}, "
                            f"lcls loss: {train_step.network.lcls_loss.asnumpy():.4f}.")
                self.summarize_loss(cur_epoch, loss, steps_per_epoch, summary_record, train_step)
                if opt.profiler and (cur_epoch == run_profiler_epoch):
                    break
                self.save_ckpt(ckpt_queue, cur_epoch, ema, ema_ckpt_queue, model)
                self.run_eval(cur_epoch, ema, infer_model, model, steps_per_epoch, summary_record, val_dataloader,
                              val_dataset)
        return 0

    def dump_cfg(self):
        if self.opt.rank % 8 != 0:
            return
        save_dir = self.opt.save_dir
        # with open(os.path.join(save_dir, "hyp.yaml"), 'w') as f:
        with os.fdopen(os.open(os.path.join(save_dir, "hyp.yaml"), WRITE_FLAGS, FILE_MODE), 'w') as f:
            yaml.dump(self.hyp, f, sort_keys=False)
        # with open(os.path.join(save_dir, "opt.yaml"), 'w') as f:
        with os.fdopen(os.open(os.path.join(save_dir, "opt.yaml"), WRITE_FLAGS, FILE_MODE), 'w') as f:
            yaml.dump(vars(self.opt), f, sort_keys=False)

    def run_eval(self, cur_epoch, ema, infer_model, model, steps_per_epoch, summary_record, val_dataloader,
                 val_dataset):
        opt = self.opt

        # Evaluation
        def is_eval_epoch():
            if cur_epoch == opt.eval_start_epoch:
                return True
            if (cur_epoch > opt.eval_start_epoch) and (cur_epoch % opt.eval_epoch_interval) == 0:
                return True
            return False

        if opt.run_eval and is_eval_epoch():
            coco_result = val(opt, model, ema, infer_model, val_dataloader, val_dataset, cur_epoch=cur_epoch)
            if opt.summary:
                summary_record.add_value('scalar', 'map', ms.Tensor(coco_result.get_map()))
                summary_record.record(cur_epoch * steps_per_epoch)
            self.save_eval_results(coco_result, cur_epoch, ema, model)

    def save_ckpt(self, ckpt_queue, cur_epoch, ema, ema_ckpt_queue, model):
        opt = self.opt

        def is_save_epoch():
            return (cur_epoch >= opt.start_save_epoch) and (cur_epoch % opt.save_interval == 0)

        if opt.save_checkpoint and (opt.rank % 8 == 0) and is_save_epoch():
            # Save Checkpoint
            model_name = os.path.basename(opt.cfg)[:-5]  # delete ".yaml"
            ckpt_path = os.path.join(self.weight_dir, f"{model_name}_{cur_epoch}.ckpt")
            ms.save_checkpoint(model, ckpt_path, append_dict={"epoch": cur_epoch})
            ckpt_queue.append(ckpt_path)
            self._modelarts_sync(ckpt_path, opt.train_url + "/weights/" + ckpt_path.split("/")[-1])
            if ema:
                ema_ckpt_path = os.path.join(self.weight_dir, f"EMA_{model_name}_{cur_epoch}.ckpt")
                append_dict = {"updates": ema.updates, "epoch": cur_epoch}
                save_ema(ema, ema_ckpt_path, append_dict)
                ema_ckpt_queue.append(ema_ckpt_path)
                LOGGER.info(f"Save ckpt path: {ema_ckpt_path}")
                self._modelarts_sync(ema_ckpt_path, opt.train_url + "/weights/" + ema_ckpt_path.split("/")[-1])

    def get_dataset(self, model, epoch_size, mode="train"):
        opt = self.opt
        gs = max(int(model.stride.asnumpy().max()), 32)  # grid size (max stride)
        imgsz, _ = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

        if mode == "train":
            train_path = self.data_cfg["train"]
            dataloader, dataset, per_epoch_size = create_dataloader(train_path, imgsz, opt.batch_size, gs, opt,
                                                                    epoch_size=epoch_size,
                                                                    hyp=self.hyp, augment=True, cache=opt.cache_images,
                                                                    rect=opt.rect, rank_size=opt.rank_size,
                                                                    rank=opt.rank, num_parallel_workers=12,
                                                                    image_weights=opt.image_weights, quad=opt.quad,
                                                                    prefix=colorstr('train: '), model_train=True)
            return dataloader, dataset, per_epoch_size
        # val
        rect = False
        test_path = self.data_cfg["val"]
        num_parallel_workers = 4 if opt.rank_size > 1 else 8
        val_dataloader, val_dataset, val_per_epoch_size = create_dataloader(test_path, imgsz, opt.batch_size, gs,
                                                                            opt, epoch_size=epoch_size, pad=0.5,
                                                                            rect=rect,
                                                                            rank=(opt.rank % 8) if opt.distributed_eval else 0,
                                                                            rank_size=min(8, opt.rank_size) if opt.distributed_eval else 1,
                                                                            num_parallel_workers=num_parallel_workers,
                                                                            shuffle=False,
                                                                            drop_remainder=False,
                                                                            prefix=colorstr('val: '))
        return val_dataloader, val_dataset, val_per_epoch_size

    def summarize_loss(self, cur_epoch, loss, steps_per_epoch, summary_record, train_step):
        if not self.opt.summary or (cur_epoch % self.opt.summary_interval == 0):
            return
        summary_record.add_value('scalar', 'loss', loss / self.opt.batch_size)
        summary_record.add_value('scalar', 'lbox', train_step.network.lbox_loss)
        summary_record.add_value('scalar', 'lobj', train_step.network.lobj_loss)
        summary_record.add_value('scalar', 'lcls', train_step.network.lcls_loss)
        summary_record.record(cur_epoch * steps_per_epoch)

    def get_data_cfg(self):
        opt = self.opt
        with open(opt.data) as f:
            data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
        if opt.enable_modelarts:
            data_dict['root'] = opt.data_dir
        data_dict = process_dataset_cfg(data_dict)
        nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
        names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
        assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check
        data_dict['names'] = names
        data_dict['nc'] = nc
        return data_dict

    def save_eval_results(self, coco_result, cur_epoch, ema, model):
        opt = self.opt
        cls_names = self.data_cfg['names']
        if opt.rank % 8 != 0:
            return
        model_name = Path(opt.cfg).stem  # delete ".yaml" suffix
        map_str_path = os.path.join(self.weight_dir, f"{model_name}_{cur_epoch}_map.txt")
        self._write_map_result(coco_result, map_str_path, cls_names)
        cur_map = coco_result.get_map()
        if cur_map > self.best_map:
            self.best_map = cur_map
            LOGGER.info(f"Best result: Best mAP [{self.best_map}] at epoch [{cur_epoch}]")
            # save the best checkpoint
            ckpt_path = os.path.join(self.weight_dir, f"{model_name}_best.ckpt")
            ms.save_checkpoint(model, ckpt_path, append_dict={"epoch": cur_epoch})
            self._modelarts_sync(ckpt_path, opt.train_url + "/weights/" + ckpt_path.split("/")[-1])
            if ema:
                ema_ckpt_path = os.path.join(self.weight_dir, f"EMA_{model_name}_best.ckpt")
                append_dict = {"updates": ema.updates, "epoch": cur_epoch}
                save_ema(ema, ema_ckpt_path, append_dict)
                self._modelarts_sync(ema_ckpt_path,
                                     opt.train_url + "/weights/" + ema_ckpt_path.split("/")[-1])

    def get_train_step(self, compute_loss, ema, model, optimizer, gs, imgsz, loss_scaler):
        if self.opt.ms_strategy == "StaticShape":
            train_step = create_train_network(model, compute_loss, ema, optimizer,
                                              loss_scaler=loss_scaler, sens=self.opt.ms_grad_sens, opt=self.opt,
                                              enable_clip_grad=self.hyp["enable_clip_grad"],
                                              gs=gs, imgsz=imgsz)
        else:
            raise NotImplementedError
        return train_step

    def get_loss_scaler(self):
        opt = self.opt
        if opt.ms_loss_scaler == "dynamic":
            from mindspore.amp import DynamicLossScaler
            loss_scaler = DynamicLossScaler(2 ** 12, 2, 1000)
        elif opt.ms_loss_scaler == "static":
            from mindspore.amp import StaticLossScaler
            loss_scaler = StaticLossScaler(opt.ms_loss_scaler_value)
        else:
            loss_scaler = None
        return loss_scaler

    def get_optimizer(self, model, per_epoch_size, resume_epoch):
        opt = self.opt
        hyp = self.hyp
        pg0, pg1, pg2 = get_group_param(model)
        lr_pg0, lr_pg1, lr_pg2, momentum_pg, _ = get_lr(opt, hyp, per_epoch_size, resume_epoch)
        group_params = [
            {'params': pg0, 'lr': lr_pg0, 'weight_decay': hyp['weight_decay']},
            {'params': pg1, 'lr': lr_pg1, 'weight_decay': 0.0},
            {'params': pg2, 'lr': lr_pg2, 'weight_decay': 0.0}]
        LOGGER.info(f"optimizer loss scale is {opt.ms_optim_loss_scale}")
        if opt.optimizer == "sgd":
            optimizer = nn.SGD(group_params, learning_rate=hyp['lr0'], momentum=hyp['momentum'], nesterov=True,
                               loss_scale=opt.ms_optim_loss_scale)
        elif opt.optimizer == "momentum":
            optimizer = YoloMomentum(group_params, learning_rate=hyp['lr0'], momentum=momentum_pg, use_nesterov=True,
                                     loss_scale=opt.ms_optim_loss_scale)
        elif opt.optimizer == "adam":
            optimizer = nn.Adam(group_params, learning_rate=hyp['lr0'], beta1=hyp['momentum'], beta2=0.999,
                                loss_scale=opt.ms_optim_loss_scale)
        else:
            raise NotImplementedError
        return optimizer

    def freeze_layer(self, model):
        freeze = self.opt.freeze
        # parameter names to freeze (full or partial)
        freeze = freeze if len(freeze) > 1 else range(freeze[0])
        freeze = [f'model.{x}.' for x in freeze]
        for n, p in model.parameters_and_names():
            if any(x in n for x in freeze):
                LOGGER.info(f'freezing {n}')
                p.requires_grad = False
        return model

    def _modelarts_sync(self, src_dir, dst_dir):
        if not self.opt.enable_modelarts:
            return
        from src.modelarts import sync_data
        os.makedirs(dst_dir, exist_ok=True)
        sync_data(src_dir, dst_dir)

    def _configure_model_params(self, dataset, imgsz, model, nl):
        hyp = self.hyp
        opt = self.opt
        num_cls = self.data_cfg['nc']
        cls_names = self.data_cfg['names']
        hyp['box'] *= 3. / nl  # scale to layers
        hyp['cls'] *= num_cls / 80. * 3. / nl  # scale to classes and layers
        hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
        hyp['label_smoothing'] = opt.label_smoothing
        model.nc = num_cls  # attach number of classes to model
        model.hyp = hyp  # attach hyperparameters to model
        model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
        model.class_weights = Tensor(labels_to_class_weights(dataset.labels, num_cls) * num_cls)  # attach class weights
        model.names = cls_names
        return model


def main():
    parser = get_args_train()
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
    assert not empty(opt.cfg) or not empty(opt.weights), 'either --cfg or --weights must be specified'
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    opt.name = 'evolve' if opt.evolve else opt.name
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

    ms_mode = context.GRAPH_MODE if opt.ms_mode == "graph" else context.PYNATIVE_MODE
    context.set_context(mode=ms_mode, device_target=opt.device_target, save_graphs=False)
    if opt.device_target == "Ascend":
        device_id = int(os.getenv('DEVICE_ID', "0"))
        context.set_context(device_id=device_id)
    # Distribute Train
    rank, rank_size, parallel_mode = 0, 1, ParallelMode.STAND_ALONE
    if opt.distributed_train:
        init()
        rank, rank_size, parallel_mode = get_rank(), get_group_size(), ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=rank_size,
                                          all_reduce_fusion_config=[10, 70, 130, 190, 250, 310])

    opt.rank, opt.rank_size = rank, rank_size
    opt.total_batch_size = opt.batch_size * opt.rank_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
    # Train
    profiler = None
    if opt.profiler:
        profiler = Profiler()

    if not opt.evolve:
        LOGGER.info(f"OPT: {opt}")
        train_manager = TrainManager(hyp, opt)
        train_manager.train()
    else:
        raise NotImplementedError("Not support evolve train")

    if opt.profiler:
        profiler.analyse()


if __name__ == '__main__':
    main()
