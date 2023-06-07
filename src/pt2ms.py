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

import sys
import torch
from mindspore.train.serialization import save_checkpoint
from mindspore import Tensor


def main():
    # 修改src_pt_path, dst_ckpt_path
    src_pt_path = sys.argv[1]
    dst_ckpt_path = './yolov5_pt2ms.ckpt'
    torch_dict = torch.load(src_pt_path)
    torch_dict = torch_dict["model"].state_dict()

    new_params_list = []
    for k, _ in torch_dict.items():
        k_backup = k
        elem_list = k.split('.')
        if len(elem_list) <= 1:
            continue

        if elem_list[1] in ['2', '4', '6', '8', '9', '13', '17', '20', '23']:
            k = k.replace('cv1', 'conv1')
            k = k.replace('cv2', 'conv2')
            k = k.replace('cv3', 'conv3')

        if elem_list[-2] == 'bn':
            if elem_list[-1] == 'weight':
                k = k.replace('weight', 'gamma')
            elif elem_list[-1] == 'bias':
                k = k.replace('bias', 'beta')
            elif elem_list[-1] == 'running_mean':
                k = k.replace('running_mean', 'moving_mean')
            elif elem_list[-1] == 'running_var':
                k = k.replace('running_var', 'moving_variance')
            else:
                continue

        _param_dict = {'name': k, 'data': Tensor(torch_dict[k_backup].cpu().numpy())}
        new_params_list.append(_param_dict)

    save_checkpoint(new_params_list, dst_ckpt_path)


if __name__ == "__main__":
    main()
