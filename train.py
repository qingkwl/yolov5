import copy
from collections import deque

import yaml
import math
import os
import random
import time
import numpy as np
import albumentations as A
from pathlib import Path
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.context import ParallelMode
from mindspore import context, Tensor
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.ops import functional as F
from mindspore.profiler.profiling import Profiler

from config.args import get_args_train
from src.network.loss import ComputeLoss
from src.network.yolo import Model
from src.network.common import EMA
from src.optimizer import get_group_param, get_lr
from src.dataset import create_dataloader
from src.general import increment_path, colorstr, labels_to_class_weights, check_file, check_img_size
from test import test


def set_seed(seed=2):
    np.random.seed(seed)
    random.seed(seed)
    ms.set_seed(seed)


# clip grad define ----------------------------------------------------------------------------------
clip_grad = ops.MultitypeFuncGraph("clip_grad")
hyper_map = ops.HyperMap()
GRADIENT_CLIP_TYPE = 1  # 0, ClipByValue; 1, ClipByNorm;
GRADIENT_CLIP_VALUE = 10.0


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
        Clip gradients.

        Inputs:
            clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
            clip_value (float): Specifies how much to clip.
            grad (tuple[Tensor]): Gradients.

        Outputs:
            tuple[Tensor]: clipped gradients.
        """
    if clip_type not in (0, 1):
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = ops.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                     F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad


# ----------------------------------------------------------------------------------------------------
def detect_overflow(epoch, cur_step, grads):
    for i in range(len(grads)):
        tmp = grads[i].asnumpy()
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


def create_train_network(model, optimizer, loss_scaler, grad_reducer=None, rank_size=1,
                         amp_level="O0", sens=1.0):
    # from mindspore.amp import all_finite # Bugs before MindSpore 1.9.0
    if loss_scaler is None:
        from mindspore.amp import StaticLossScaler
        loss_scaler = StaticLossScaler(sens)
    # Def train func
    compute_loss = ComputeLoss(model)  # init loss class
    ms.amp.auto_mixed_precision(compute_loss, amp_level=amp_level)

    if grad_reducer is None:
        grad_reducer = ops.functional.identity

    def forward_func(x, label, sizes=None):
        x /= 255.0
        if sizes is not None:
            x = ops.interpolate(x, sizes=sizes, coordinate_transformation_mode="asymmetric", mode="bilinear")
        pred = model(x)
        loss, loss_items = compute_loss(pred, label)
        loss *= rank_size
        return loss, ops.stop_gradient(loss_items)

    grad_fn = ops.GradOperation(get_by_list=True, sens_param=True)(forward_func, optimizer.parameters)
    sens_value = sens

    @ms.ms_function
    def train_step(x, label, sizes=None, optimizer_update=True):
        loss, loss_items = forward_func(x, label, sizes)
        sens1, sens2 = ops.fill(loss.dtype, loss.shape, sens_value), \
                       ops.fill(loss_items.dtype, loss_items.shape, sens_value)
        grads = grad_fn(x, label, sizes, (sens1, sens2))
        grads = grad_reducer(grads)
        grads = loss_scaler.unscale(grads)
        # grads_finite = all_finite(grads)

        if optimizer_update:
            loss = ops.depend(loss, optimizer(grads))

        return loss, loss_items, grads

    return train_step


def val(opt, model, ema,
             infer_model, val_dataloader, val_dataset, cur_epoch):
    print("[INFO] Evaluating...", flush=True)
    param_dict = {}
    if opt.ema:
        print("[INFO] ema parameter update", flush=True)
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
    info, maps, t, map_table_str, map_str = test(opt.data,
                                                 opt.weights,
                                                 opt.batch_size,
                                                 opt.img_size,
                                                 opt.conf_thres,
                                                 opt.iou_thres,
                                                 opt.save_json,
                                                 opt.single_cls,
                                                 opt.augment,
                                                 opt.verbose,
                                                 model=infer_model,
                                                 dataloader=val_dataloader,
                                                 dataset=val_dataset,
                                                 save_txt=opt.save_txt | opt.save_hybrid,
                                                 save_hybrid=opt.save_hybrid,
                                                 save_conf=opt.save_conf,
                                                 trace=not opt.no_trace,
                                                 plots=False,
                                                 half_precision=False,
                                                 v5_metric=opt.v5_metric,
                                                 is_distributed=opt.is_distributed,
                                                 rank=opt.rank,
                                                 rank_size=opt.rank_size,
                                                 opt=opt,
                                                 cur_epoch=cur_epoch)
    infer_model.set_train(True)
    return info, maps, t, map_table_str, map_str


def save_best_ckpt(opt, model, ema, best_map, wdir, cur_epoch):
    print(f"[INFO] Best result: Best mAP [{best_map}] at epoch [{cur_epoch}]", flush=True)
    # save best checkpoint
    model_name = os.path.basename(opt.cfg)[:-5]  # delete ".yaml"
    ckpt_path = os.path.join(wdir, f"{model_name}_best.ckpt")
    ms.save_checkpoint(model, ckpt_path, append_dict={"epoch": cur_epoch})
    if ema:
        params_list = []
        for p in ema.ema_weights:
            _param_dict = {'name': p.name[len("ema."):], 'data': Tensor(p.data.asnumpy())}
            params_list.append(_param_dict)

        ema_ckpt_path = os.path.join(wdir, f"EMA_{model_name}_best.ckpt")
        ms.save_checkpoint(params_list, ema_ckpt_path,
                           append_dict={"updates": ema.updates, "epoch": cur_epoch})
    if opt.enable_modelarts:
        from src.modelarts import sync_data
        sync_data(ckpt_path, opt.train_url + "/weights/" + ckpt_path.split("/")[-1])
        if ema:
            sync_data(ema_ckpt_path, opt.train_url + "/weights/" + ema_ckpt_path.split("/")[-1])


class CheckpointQueue:
    def __init__(self, max_ckpt_num):
        self.max_ckpt_num = max_ckpt_num
        self.ckpt_queue = deque()

    def append(self, ckpt_path):
        self.ckpt_queue.append(ckpt_path)
        if len(self.ckpt_queue) > self.max_ckpt_num:
            ckpt_to_delete = self.ckpt_queue.popleft()
            os.remove(ckpt_to_delete)


def train(hyp, opt):
    set_seed()
    if opt.enable_modelarts:
        from src.modelarts import sync_data
        os.makedirs(opt.data_dir, exist_ok=True)
        sync_data(opt.data_url, opt.data_dir)

    save_dir, epochs, batch_size, total_batch_size, weights, rank, freeze = \
        opt.save_dir, opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.rank, opt.freeze

    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    if opt.enable_modelarts:
        data_dict['train'] = os.path.join(opt.data_dir, data_dict['train'])
        data_dict['val'] = os.path.join(opt.data_dir, data_dict['val'])
        data_dict['test'] = os.path.join(opt.data_dir, data_dict['test'])

    # Directories
    wdir = os.path.join(save_dir, "weights")
    os.makedirs(wdir, exist_ok=True)
    # Save run settings
    with open(os.path.join(save_dir, "hyp.yaml"), 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(os.path.join(save_dir, "opt.yaml"), 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)
    if opt.enable_modelarts:
        sync_data(save_dir, opt.train_url)

    # Model
    sync_bn = opt.sync_bn and context.get_context("device_target") == "Ascend" and rank_size > 1
    model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors'), sync_bn=sync_bn, opt=opt)  # create
    ema = EMA(model) if opt.ema else None

    pretrained = weights.endswith('.ckpt')
    resume_epoch = 0
    if pretrained:
        resume_epoch = load_checkpoint_to_yolo(model, weights, opt.resume)
        ema.clone_from_model()
        print("ema_weight not exist, default pretrain weight is currently used.")

    # Freeze
    freeze = [f'model.{x}.' for x in
              (freeze if len(freeze) > 1 else range(freeze[0]))]  # parameter names to freeze (full or partial)
    for n, p in model.parameters_and_names():
        if any(x in n for x in freeze):
            print('freezing %s' % n)
            p.requires_grad = False

    # Image sizes
    gs = max(int(model.stride.asnumpy().max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples
    train_path = data_dict['train']
    test_path = data_dict['val']
    train_epoch_size = 1 if opt.optimizer == "thor" else opt.epochs - resume_epoch
    dataloader, dataset, per_epoch_size = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                                            epoch_size=train_epoch_size,
                                                            hyp=hyp, augment=True, cache=opt.cache_images,
                                                            rect=opt.rect, rank_size=opt.rank_size, rank=opt.rank,
                                                            num_parallel_workers=12 if rank_size > 1 else 8,
                                                            image_weights=opt.image_weights, quad=opt.quad,
                                                            prefix=colorstr('train: '))
    if opt.run_eval:
        infer_model = copy.deepcopy(model) if opt.ema else model
        rect = False
        val_batch_size = 32
        val_dataloader, val_dataset, _ = create_dataloader(test_path, imgsz, val_batch_size, gs, opt,
                                                           epoch_size=1, pad=0.5, rect=rect,
                                                           rank=rank, rank_size=rank_size,
                                                           num_parallel_workers=4 if rank_size > 1 else 8,
                                                           shuffle=False,
                                                           drop_remainder=False,
                                                           prefix=colorstr(f'val: '))
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    # accumulate = 1  # accumulate loss before optimizing

    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    print(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = get_group_param(model)
    lr_pg0, lr_pg1, lr_pg2, momentum_pg, warmup_steps = get_lr(opt, hyp, per_epoch_size, resume_epoch)
    # if opt.optimizer in ("sgd", "momentum"):
    #     assert len(momentum_pg) == warmup_steps
    group_params = [
        {'params': pg0, 'lr': lr_pg0, 'weight_decay': hyp['weight_decay']},
        {'params': pg1, 'lr': lr_pg1, 'weight_decay': 0.0},
        {'params': pg2, 'lr': lr_pg2, 'weight_decay': 0.0}]
    print(f"optimizer loss scale is {opt.ms_optim_loss_scale}")
    if opt.optimizer == "sgd":
        optimizer = nn.SGD(group_params, learning_rate=hyp['lr0'], momentum=hyp['momentum'], nesterov=True,
                           loss_scale=opt.ms_optim_loss_scale)
    elif opt.optimizer == "momentum":
        optimizer = nn.Momentum(group_params, learning_rate=hyp['lr0'], momentum=hyp['momentum'], use_nesterov=True,
                                loss_scale=opt.ms_optim_loss_scale)
    elif opt.optimizer == "adam":
        optimizer = nn.Adam(group_params, learning_rate=hyp['lr0'], beta1=hyp['momentum'], beta2=0.999,
                            loss_scale=opt.ms_optim_loss_scale)
    else:
        raise NotImplementedError

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = Tensor(labels_to_class_weights(dataset.labels, nc) * nc)  # attach class weights
    model.names = names

    # Build train process function
    # amp
    ms.amp.auto_mixed_precision(model, amp_level=opt.ms_amp_level)
    if opt.is_distributed:
        mean = context.get_auto_parallel_context("gradients_mean")
        degree = context.get_auto_parallel_context("device_num")
        grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)
    else:
        grad_reducer = ops.functional.identity

    if opt.ms_loss_scaler == "dynamic":
        from mindspore.amp import DynamicLossScaler
        loss_scaler = DynamicLossScaler(2 ** 12, 2, 1000)
    elif opt.ms_loss_scaler == "static":
        from mindspore.amp import StaticLossScaler
        loss_scaler = StaticLossScaler(opt.ms_loss_scaler_value)
    else:
        loss_scaler = None

    if opt.ms_strategy == "StaticShape":
        train_step = create_train_network(model, optimizer, loss_scaler, grad_reducer,
                                          amp_level=opt.ms_amp_level, rank_size=opt.rank_size,
                                          sens=opt.ms_grad_sens)
    else:
        raise NotImplementedError

    data_loader = dataloader.create_dict_iterator(output_numpy=True)

    run_profiler_num = 0
    run_profiler_step = 200
    accumulate_grads = None
    accumulate_cur_step = 0

    model.set_train(True)
    optimizer.set_train(True)
    best_map = 0.
    s_time_data = time.time()
    s_time_step = time.time()
    s_time_epoch = time.time()
    ema_ckpt_queue = CheckpointQueue(opt.max_ckpt_num)
    ckpt_queue = CheckpointQueue(opt.max_ckpt_num)
    for i, data in enumerate(data_loader):
        cur_epoch = resume_epoch + (i // per_epoch_size) + 1
        cur_step = (i % per_epoch_size) + 1
        run_profiler_num += 1
        if opt.profiler and run_profiler_num >= run_profiler_step:
            break

        if i < warmup_steps:
            xi = [0, warmup_steps]  # x interp
            accumulate = max(1, np.interp(i, xi, [1, nbs / total_batch_size]).round())
            if opt.optimizer in ("sgd", "momentum"):
                optimizer.momentum = Tensor(momentum_pg[i], ms.float32)
                # print("optimizer.momentum: ", optimizer.momentum.asnumpy())

        imgs, labels, paths = data["img"], data["label_out"], data["img_files"]

        if opt.ms_amp_level == "O3":
            imgs, labels = imgs.astype(np.float16), labels.astype(np.float16)
        imgs, labels = Tensor.from_numpy(imgs), Tensor.from_numpy(labels)

        # Multi-scale
        ns = None
        if opt.multi_scale:
            sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                # imgs = ops.interpolate(imgs, sizes=ns, coordinate_transformation_mode="asymmetric", mode="bilinear")

        # Accumulate Grad
        update_params = False if opt.accumulate and accumulate > 1 else True
        s_train_time = time.time()
        loss, loss_item, grads = train_step(imgs, labels, ns, update_params)
        d_train_time = time.time()
        # grads_finite = detect_overflow(cur_epoch, cur_step, grads)
        s_ema_time = time.time()
        grads_finite = False
        if update_params and ema:
            ema.update()
        d_ema_time = time.time()
        if grads_finite:
            if loss_scaler:
                loss_scaler.adjust(grads_finite)
                print("overflow, loss scale adjust to ", loss_scaler.scale_value.asnumpy(), flush=True)
            if opt.accumulate and accumulate > 1:
                print(f"Epoch: {cur_epoch}, Step: {cur_step}, this step grad overflow, drop. ", flush=True)
                continue

        # Skip if disable accumulate grads
        if not update_params:
            # accumulate gradient
            accumulate_cur_step += 1
            if accumulate_grads:
                assert len(accumulate_grads) == len(grads)
                for gi in range(len(grads)):
                    accumulate_grads[gi] += grads[gi]
            else:
                accumulate_grads = list(grads)

            if accumulate_cur_step % accumulate == 0:
                optimizer(tuple(accumulate_grads))
                if ema:
                    ema.update()
                print(f"Epoch: {cur_epoch}, Step: {cur_step}, optimizer an accumulate step success.", flush=True)
                # reset accumulate
                accumulate_grads = None
                accumulate_cur_step = 0

        _p_train_size = ns if ns else imgs.shape[2:]
        print(f"Epoch {epochs}/{cur_epoch}, Step {per_epoch_size}/{cur_step}, size {_p_train_size}, "
              f"data time:{(s_train_time - s_time_data) * 1000:.2f} ms, "
              f"ema time:{(d_ema_time - s_ema_time) * 1000:.2f} ms, "
              f"fp/bp time cost: {(d_train_time - s_train_time) * 1000:.2f} ms, ",
              f"step time: {(time.time() - s_time_step) * 1000:.2f} ms", flush=True)
        print(f"Epoch {epochs}/{cur_epoch}, Step {per_epoch_size}/{cur_step}, size {_p_train_size}, "
              f"loss: {loss.asnumpy():.4f}, lbox: {loss_item[0].asnumpy():.4f}, "
              f"lobj: {loss_item[1].asnumpy():.4f}, lcls: {loss_item[2].asnumpy():.4f}, "
              f"cur_lr: [{lr_pg0[i]:.8f}, {lr_pg1[i]:.8f}, {lr_pg2[i]:.8f}], ", flush=True)
        s_time_step = time.time()

        if (i + 1) % per_epoch_size == 0:
            print(f"Epoch {epochs}/{cur_epoch}, epoch time: {(time.time() - s_time_epoch) / 60:.2f} min.", flush=True)
            s_time_epoch = time.time()
        info, maps, t, map_table_str, map_str = None, None, None, None, None

        def is_save_epoch():
            return (cur_epoch >= opt.start_save_epoch) and ((i + 1) % (opt.save_interval * per_epoch_size) == 0)

        if opt.save_checkpoint and is_save_epoch():
            info, maps, t, map_table_str, map_str = val(opt, model, ema, infer_model,
                                                        val_dataloader, val_dataset, cur_epoch=cur_epoch)
            if rank % 8 == 0:
                # Save Checkpoint
                model_name = os.path.basename(opt.cfg)[:-5]  # delete ".yaml"
                ckpt_path = os.path.join(wdir, f"{model_name}_{cur_epoch}.ckpt")
                print("save ckpt path:", ckpt_path, flush=True)
                ms.save_checkpoint(model, ckpt_path, append_dict={"epoch": cur_epoch})
                ckpt_queue.append(ckpt_path)
                if ema:
                    params_list = []
                    for p in ema.ema_weights:
                        _param_dict = {'name': p.name[len("ema."):], 'data': Tensor(p.data.asnumpy())}
                        params_list.append(_param_dict)

                    ema_ckpt_path = os.path.join(wdir, f"EMA_{model_name}_{cur_epoch}.ckpt")
                    ms.save_checkpoint(params_list, ema_ckpt_path,
                                       append_dict={"updates": ema.updates, "epoch": cur_epoch})
                    ema_ckpt_queue.append(ema_ckpt_path)
                if opt.enable_modelarts:
                    sync_data(ckpt_path, opt.train_url + "/weights/" + ckpt_path.split("/")[-1])
                    if ema:
                        sync_data(ema_ckpt_path, opt.train_url + "/weights/" + ema_ckpt_path.split("/")[-1])
                map_str_path = os.path.join(wdir, f"{model_name}_{cur_epoch}_map.txt")
                with open(map_str_path, 'w') as file:
                    if map_table_str is not None:
                        file.write(f"COCO API:\n{map_table_str}\n")
                    if map_str is not None:
                        file.write(f"COCO API:\n{map_str}\n")

        # Evaluation
        def is_eval_epoch():
            return (cur_epoch >= opt.eval_start_epoch) and ((i + 1) % (opt.eval_epoch_interval * per_epoch_size)) == 0

        def need_eval():
            return info is None or maps is None or t is None or map_table_str is None

        if opt.run_eval and is_eval_epoch():
            if need_eval():
                info, maps, t, map_table_str, map_str = val(opt, model, ema, infer_model, val_dataloader,
                                                            val_dataset, cur_epoch=cur_epoch)
            else:
                print(f"[INFO] Evaluation has run at this epoch. Skip evaluation.", flush=True)
            # When need not evaluation again, use previous evaluation results
            mean_ap = info[3]
            if (rank % 8 == 0) and (mean_ap > best_map):
                best_map = mean_ap
                save_best_ckpt(opt, model, ema, best_map, wdir, cur_epoch=cur_epoch)
        s_time_data = time.time()
    return 0


if __name__ == '__main__':
    opt = get_args_train()
    opt.save_json |= opt.data.endswith('coco.yaml')
    # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
    opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    opt.name = 'evolve' if opt.evolve else opt.name
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

    ms_mode = context.GRAPH_MODE if opt.ms_mode == "graph" else context.PYNATIVE_MODE
    context.set_context(mode=ms_mode, device_target=opt.device_target, save_graphs=False)
    if opt.device_target == "Ascend":
        device_id = int(os.getenv('DEVICE_ID', 0))
        context.set_context(device_id=device_id)
    # Distribute Train
    rank, rank_size, parallel_mode = 0, 1, ParallelMode.STAND_ALONE
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

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
    # Train
    profiler = None
    if opt.profiler:
        profiler = Profiler()

    if not opt.evolve:
        print(f"[INFO] OPT: {opt}")
        train(hyp, opt)
    else:
        raise NotImplementedError("Not support evolve train;")

    if opt.profiler:
        profiler.analyse()
