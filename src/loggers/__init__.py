import os
from mindspore import SummaryRecord

LOGGERS = ('csv', 'ms')  # ('csv', 'tb', 'wandb', 'clearml', 'comet')  # *.csv, TensorBoard, Weights & Biases, ClearML
RANK = int(os.getenv('RANK_ID', "-1"))

WANDB = None
CLEARML = None
COMET_ML = None


class SummaryLoggers:
    # YOLOv5 Loggers class
    def __init__(self, save_dir=None, weights=None, opt=None, hyp=None, logger=None, include=LOGGERS):
        self.save_dir = save_dir
        self.weights = weights
        self.opt = opt
        self.hyp = hyp
        self.plots = not opt.noplots  # plot results
        self.logger = logger  # for printing results to console
        self.include = include
        # TODO: update keys name
        self.keys = [
            'train/box_loss',
            'train/obj_loss',
            'train/cls_loss',  # train loss
            'metrics/precision',
            'metrics/recall',
            'metrics/mAP_0.5',
            'metrics/mAP_0.5:0.95',  # metrics
            'val/box_loss',
            'val/obj_loss',
            'val/cls_loss',  # val loss
            'x/lr0',
            'x/lr1',
            'x/lr2']  # params
        self.best_keys = ['best/epoch', 'best/precision', 'best/recall', 'best/mAP_0.5', 'best/mAP_0.5:0.95']
        for k in LOGGERS:
            setattr(self, k, None)  # init empty logger dictionary
        self.csv = True  # always log to csv
        self.ms = None

    def __enter__(self):
        if 'tb' in self.include and self.opt.summary:
            self.logger.info(f"SummaryRecord: Start with 'mindinsight --summary-base-dir {self.save_dir.parent}'")
            self.ms = SummaryRecord(self.save_dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.ms is not None:
            self.ms.close()

    def on_train_start(self):
        pass

    def on_pretrain_routine_start(self):
        pass

    def on_pretrain_routine_end(self, labels, names):
        pass

    def on_train_batch_end(self, model, ni, imgs, targets, paths, vals):
        pass

    def on_train_epoch_end(self, epoch):
        pass

    def on_val_start(self):
        pass

    def on_val_image_end(self, pred, predn, path, names, im):
        pass

    def on_val_batch_end(self, batch_i, im, targets, paths, shapes, out):
        pass

    def on_val_end(self, nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix):
        pass

    def on_fit_epoch_end(self, vals, epoch, best_fitness, fi):
        pass

    def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
        pass

    def on_train_end(self, last, best, epoch, results):
        pass

    def on_params_update(self, params: dict):
        pass
