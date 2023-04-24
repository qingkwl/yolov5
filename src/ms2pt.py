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

from collections import defaultdict

import torch
from mindspore.train.serialization import load_checkpoint


def load_pt(src_pt_path):
    torch_dict = torch.load(src_pt_path)
    return torch_dict


def ms2pt(src_ckpt_path):
    ms_params = load_checkpoint(src_ckpt_path)
    new_params = defaultdict()
    for k, v in ms_params.items():
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


def main():
    # 修改src_pt_path, dst_ckpt_path
    import sys
    src_pt_path = 'epoch0.pt'
    src_ckpt_path = sys.argv[1]
    print("[INFO] src_ckpt_path:", src_ckpt_path)
    dst_ckpt_path = './yolov5s_ms2pt.pt'
    amend_pt(src_ckpt_path, src_pt_path, dst_ckpt_path)


if __name__ == "__main__":
    main()
