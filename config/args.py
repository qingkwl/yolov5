import argparse
import ast
from argparse import Namespace


class TestNamespace(Namespace):
    ms_mode: str
    distributed: bool
    device_target: str
    weights: str
    rect: bool
    data: str
    cfg: str
    hyp: str
    batch_size: int
    img_size: int
    conf_thres: float
    iou_thres: float
    task: str
    single_cls: bool
    augment: bool
    verbose: bool
    save_txt: bool
    save_hybrid: bool
    save_conf: bool
    save_json: bool
    project: str
    exist_ok: bool
    trace: bool
    plots: bool
    v5_metric: bool
    transfer_format: bool
    result_view: bool
    recommend_threshold: bool

    rank: int
    rank_size: int
    parallel_mode: str


def get_args_train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ms_strategy', type=str, default='StaticShape',
                        help='train strategy, StaticCell/StaticShape/MultiShape/DynamicShape')
    parser.add_argument('--ms_mode', type=str, default='graph', help='train mode, graph/pynative')
    parser.add_argument('--ms_amp_level', type=str, default='O0', help='amp level, O0/O1/O2')
    parser.add_argument('--ms_loss_scaler', type=str, default='none', help='train loss scaler, static/dynamic/none')
    parser.add_argument('--ms_loss_scaler_value', type=float, default=1.0, help='static loss scale value')
    parser.add_argument('--ms_optim_loss_scale', type=float, default=1.0, help='optimizer loss scale')
    parser.add_argument('--ms_grad_sens', type=float, default=1024, help='gard sens')
    parser.add_argument('--accumulate', type=ast.literal_eval, default=False, help='accumulate gradient')
    parser.add_argument('--overflow_still_update', type=ast.literal_eval, default=False, help='overflow still update')
    parser.add_argument('--clip_grad', type=ast.literal_eval, default=False, help='clip grad')
    parser.add_argument('--profiler', type=ast.literal_eval, default=False, help='enable profiler')
    parser.add_argument('--ema', type=ast.literal_eval, default=True, help='ema')
    parser.add_argument('--is_distributed', type=ast.literal_eval, default=False, help='Distribute train or not')
    parser.add_argument('--device_target', type=str, default='Ascend', help='device target, Ascend/GPU/CPU')
    parser.add_argument('--recompute', type=ast.literal_eval, default=False, help='Recompute')
    parser.add_argument('--recompute_layers', type=int, default=0)
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--ema_weight', type=str, default='', help='initial ema weights path')
    parser.add_argument('--cfg', type=str, default='config/network/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='config/data/coco.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='config/data/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32, help='total batch size for all device')
    parser.add_argument('--img_size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')

    parser.add_argument('--save_checkpoint', type=ast.literal_eval, default=True, help='save checkpoint')
    parser.add_argument('--start_save_epoch', type=int, default=100, help='epoch to start save checkpoint')
    parser.add_argument('--save_interval', type=int, default=5, help='the interval epoch of save checkpoint')
    parser.add_argument('--max_ckpt_num', type=int, default=40,
                        help='the maximum number of save checkpoint, delete previous checkpoints if '
                             'the number of saved checkpoints are larger than this value')

    parser.add_argument('--rect', type=ast.literal_eval, default=False, help='rectangular training')
    parser.add_argument('--resume', type=ast.literal_eval, default=False, help='resume specified checkpoint training')
    parser.add_argument('--nosave', type=ast.literal_eval, default=False, help='only save final checkpoint')
    parser.add_argument('--notest', type=ast.literal_eval, default=False, help='only test final epoch')
    parser.add_argument('--noautoanchor', type=ast.literal_eval, default=False, help='disable autoanchor check')
    parser.add_argument('--plots', type=ast.literal_eval, default=False, help='enable plot')
    parser.add_argument('--evolve', type=ast.literal_eval, default=False, help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache_images', type=ast.literal_eval, default=False, help='cache images for faster training')
    parser.add_argument('--image_weights', type=ast.literal_eval, default=False,
                        help='use weighted image selection for training')
    parser.add_argument('--multi_scale', type=ast.literal_eval, default=False, help='vary img-size +/- 50%%')
    parser.add_argument('--single_cls', type=ast.literal_eval, default=False,
                        help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, default='sgd', help='select optimizer')
    parser.add_argument('--sync_bn', type=ast.literal_eval, default=False,
                        help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist_ok', type=ast.literal_eval, default=False,
                        help='existing project/name ok, do not increment')
    parser.add_argument('--quad', type=ast.literal_eval, default=False, help='quad dataloader')
    parser.add_argument('--linear_lr', type=ast.literal_eval, default=True, help='linear LR')
    parser.add_argument('--result_view', type=ast.literal_eval, default=False, help='view the eval result')
    parser.add_argument('--recommend_threshold', type=ast.literal_eval, default=False,
                        help='recommend threshold in eval')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', type=ast.literal_eval, default=False,
                        help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0],
                        help='Freeze layers: backbone of yolov5, first3=0 1 2')
    parser.add_argument('--v5_metric', type=ast.literal_eval, default=False,
                        help='assume maximum recall as 1.0 in AP calculation')
    parser.add_argument('--summary', type=ast.literal_eval, default=False,
                        help='Whether use SummaryRecord to log intermediate data')
    parser.add_argument('--summary_dir', type=str, default="summary",
                        help='Folder to save summary files with project/summary_dir structure')
    parser.add_argument('--summary_interval', type=int, default=1,
                        help='Epoch interval to save summary files')  # Unit: epoch

    # args for evaluation
    parser.add_argument('--run_eval', type=ast.literal_eval, default=True,
                        help='Whether do evaluation after some epoch')
    parser.add_argument('--eval_start_epoch', type=int, default=200, help='Start epoch interval to do evaluation')
    parser.add_argument('--eval_epoch_interval', type=int, default=10, help='Epoch interval to do evaluation')
    parser.add_argument('--conf_thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save_txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save_hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save_conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save_json', type=ast.literal_eval, default=True,
                        help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--trace', type=ast.literal_eval, default=False, help='trace model')
    parser.add_argument('--transfer_format', type=ast.literal_eval, default=True,
                        help='whether transform data format to coco')

    # args for ModelArts
    parser.add_argument('--enable_modelarts', type=ast.literal_eval, default=False, help='enable modelarts')
    parser.add_argument('--data_url', type=str, default='', help='ModelArts: obs path to dataset folder')
    parser.add_argument('--train_url', type=str, default='', help='ModelArts: obs path to dataset folder')
    parser.add_argument('--data_dir', type=str, default='/cache/data/', help='ModelArts: obs path to dataset folder')
    return parser


def get_args_test():
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--ms_mode', type=str, default='graph', help='train mode, graph/pynative')
    parser.add_argument('--is_distributed', type=ast.literal_eval, default=False, help='Distribute test or not')
    parser.add_argument('--device_target', type=str, default='Ascend', help='device target, Ascend/GPU/CPU')
    parser.add_argument('--weights', type=str, default='/EMA_yolov5s_300.ckpt',
                        help='model.ckpt path(s)')
    parser.add_argument('--rect', type=ast.literal_eval, default=False, help='rectangular training')
    parser.add_argument('--data', type=str, default='config/data/coco.yaml', help='*.data path')
    parser.add_argument('--cfg', type=str, default='config/network/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--hyp', type=str, default='config/data/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--batch_size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--single_cls', type=ast.literal_eval, default=False, help='treat as single-class dataset')
    parser.add_argument('--augment', type=ast.literal_eval, default=False, help='augmented inference')
    parser.add_argument('--verbose', type=ast.literal_eval, default=False, help='report mAP by class')
    parser.add_argument('--save_txt', type=ast.literal_eval, default=False, help='save results to *.txt')
    parser.add_argument('--save_hybrid', type=ast.literal_eval, default=False,
                        help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save_conf', type=ast.literal_eval, default=False,
                        help='save confidences in --save-txt labels')
    parser.add_argument('--save_json', type=ast.literal_eval, default=True,
                        help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', type=str, default='./run_test', help='save to project/name')
    parser.add_argument('--exist_ok', type=ast.literal_eval, default=True,
                        help='existing project/name ok, do not increment')
    parser.add_argument('--trace', type=ast.literal_eval, default=False, help='trace model')
    parser.add_argument('--plots', type=ast.literal_eval, default=True, help='enable plot')
    parser.add_argument('--v5_metric', type=ast.literal_eval, default=True,
                        help='assume maximum recall as 1.0 in AP calculation')
    parser.add_argument('--transfer_format', type=ast.literal_eval, default=True,
                        help='whether transform data format to coco')
    parser.add_argument('--result_view', type=ast.literal_eval, default=False, help='view the eval result')
    parser.add_argument('--recommend_threshold', type=ast.literal_eval, default=False,
                        help='recommend threshold in eval')
    return parser


def get_args_infer():
    parser = argparse.ArgumentParser(prog='infer.py')
    parser.add_argument('--ms_mode', type=str, default='graph', help='train mode, graph/pynative')
    parser.add_argument('--device_target', type=str, default='Ascend', help='device target, Ascend/GPU/CPU')
    parser.add_argument('--om', type=str, default='yolov5s.om', help='model.om path')
    parser.add_argument('--rect', type=ast.literal_eval, default=False, help='rectangular training')
    parser.add_argument('--data', type=str, default='config/data/coco.yaml', help='*.data path')
    parser.add_argument('--cfg', type=str, default='config/network/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--hyp', type=str, default='config/data/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--batch_size', type=int, default=32, help='size of each image batch')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.65, help='IOU threshold for NMS')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--single_cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--output_dir', type=str, default='./output', help='Folder path to save prediction json')
    return parser


def get_args_export():
    parser = argparse.ArgumentParser(prog='export.py')
    parser.add_argument('--ms_mode', type=str, default='graph', help='train mode, graph/pynative')
    parser.add_argument('--device_target', type=str, default='Ascend', help='device target, Ascend/GPU/CPU')
    parser.add_argument('--weights', type=str, default='./EMA_yolov5s_300.ckpt',
                        help='model.ckpt path')
    parser.add_argument('--data', type=str, default='config/data/coco.yaml', help='*.data path')
    parser.add_argument('--cfg', type=str, default='config/network/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--hyp', type=str, default='config/data/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--export_batch_size', type=int, default=1, help='size of each image batch')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--file_format', type=str, default='MINDIR', help='treat as single-class dataset')

    # preprocess
    parser.add_argument('--output_path', type=str, default='./', help='output preprocess data path')
    return parser


# if __name__ == "main":
# parser = get_args_test()
# eval_args = parser.parse_args(namespace=TestNamespace())
# print("Eval args:")
# # print(vars(eval_args))
# from pprint import pprint, pformat
# pprint(pformat(eval_args))
# print(eval_args.batch_size)
# eval_args.a = 0
# print(eval_args.a)
