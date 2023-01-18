import numpy as np

from mindspore.train.serialization import load_checkpoint, save_checkpoint
import torch
from mindspore import Tensor
from collections import defaultdict


def load_pt(src_pt_path):
    torch_dict = torch.load(src_pt_path)
    return torch_dict


def ms2pt(src_ckpt_path):
    ms_params = load_checkpoint(src_ckpt_path)
    new_params = defaultdict()
    for k, v in ms_params.items():
        k_backup = k
        elem_list = k.split('.')
        if len(elem_list) <= 1:
            print("k", k)
            continue

        if elem_list[1] in ['2', '4', '6', '8', '9', '13', '17', '20', '23']:
            k = k.replace('conv1', 'cv1')
            k = k.replace('conv2', 'cv2')
            k = k.replace('conv3', 'cv3')

        if elem_list[-2] == 'bn':
            if elem_list[-1] == 'gamma':
                k = k.replace('gamma', 'weight')
            elif elem_list[-1] == 'beta':
                k = k.replace('beta', 'bias')
            elif elem_list[-1] == 'moving_mean':
                k = k.replace('moving_mean', 'running_mean')
            elif elem_list[-1] == 'moving_variance':
                k = k.replace('moving_variance', 'running_var')
            else:
                continue
        # print(k)
        v = torch.from_numpy(v.asnumpy())
        new_params[k] = v
    return new_params


def amend_pt(src_ckpt_path, src_pt_path, dst_ckpt_path):
    new_params = ms2pt(src_ckpt_path)
    torch_dict = load_pt(src_pt_path)
    print(torch_dict.keys())
    torch_dict["model"].load_state_dict(new_params, strict=False)
    torch_dict["updates"] = None
    torch_dict["ema"] = None
    torch_dict["optimizer"] = None
    torch_dict["opt"] = None
    torch_dict["epoch"] = 0
    torch_dict["best_fitness"] = 0
    torch.save(torch_dict, dst_ckpt_path)
    #print(torch_dict)


if __name__ == "__main__":
    # 修改src_pt_path, dst_ckpt_path
    import sys
    
    src_pt_path = 'epoch0.pt'
    src_ckpt_path = sys.argv[1]
    print("[INFO] src_ckpt_path:", src_ckpt_path)
    dst_ckpt_path = './yolov5s_ms2pt.pt'
    amend_pt(src_ckpt_path, src_pt_path, dst_ckpt_path)
