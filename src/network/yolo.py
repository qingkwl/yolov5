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

import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import mindspore as ms
from mindspore import Tensor, nn, ops

from src.autoanchor import check_anchor_order
from src.network.common import Detect, parse_model


def initialize_weights(model, hyp):
    for _, m in model.cells_and_names():
        classname = m.__class__.__name__
        if classname == "Conv2d":
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif classname in ("BatchNorm2d", "SyncBatchNorm"):
            m.eps = hyp["bn_eps"]
            m.momentum = hyp["bn_momentum"]


@ops.constexpr
def _get_h_w_list(ratio, gs, hw):
    return tuple([math.ceil(x * ratio / gs) * gs for x in hw])


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    # scales img(bs,3,y,x) by ratio constrained to gs-multiple
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))  # new size
    img = ops.ResizeBilinear(size=s, align_corners=False)(img)
    if not same_shape:  # pad/crop img
        h, w = _get_h_w_list(ratio, gs, (h, w))

    img = ops.pad(img, ((0, 0), (0, 0), (0, w - s[1]), (0, h - s[0])))
    img[:, :, -(w - s[1]):, :] = 0.447
    img[:, :, :, -(h - s[0]):] = 0.447
    return img


@ops.constexpr
def _get_stride_max(stride):
    return int(stride.max())


class Model(nn.Cell):
    def __init__(self, cfg='yolor-csp-c.yaml', ch=3, nc=None, anchors=None, sync_bn=False,
                 opt=None, hyp=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        self.traced = False
        self.multi_scale = False
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            print(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            print(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save, self.layers_param = parse_model(deepcopy(self.yaml), ch=[ch], sync_bn=sync_bn)
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names

        # Recompute
        if opt is not None:
            if opt.recompute and opt.recompute_layers > 0:
                for i in range(opt.recompute_layers):
                    self.model[i].recompute()
                print(f"Turn on recompute, and the results of the first {opt.recompute_layers} layers "
                      f"will be recomputed.")

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = Tensor(np.array(self.yaml['stride']), ms.int32)
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self.stride_np = np.array(self.yaml['stride'])
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self.model, hyp)

    def construct(self, x, augment=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = (1, 0.83, 0.67)  # scales
            f = (None, 3, None)  # flips (2-ud, 3-lr)
            y = ()  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(ops.ReverseV2([fi])(x) if fi else x, si, gs=_get_stride_max(self.stride_np))
                yi = self.forward_once(xi)[0]  # forward
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y += (yi,)
            return ops.concat(y, 1)  # augmented inference, train
        return self.forward_once(x)  # single-scale inference, train

    def forward_once(self, x):
        y, _ = (), ()  # outputs
        for i in range(len(self.model)):
            m = self.model[i]
            iol, f, _, _ = self.layers_param[i]  # iol: index of layers

            if not (isinstance(f, int) and f == -1):  # if not from previous layer
                if isinstance(f, int):
                    x = y[f]
                else:
                    _x = ()
                    for j in f:
                        if j == -1:
                            _x += (x,)
                        else:
                            _x += (y[j],)
                    x = _x

            if self.traced:
                if isinstance(m, Detect):
                    break

            x = m(x)  # run

            y += (x if iol in self.save else None,)  # save output

        return x

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            s = s.asnumpy()
            b = mi.bias.view(m.na, -1).asnumpy()  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99999)) if cf is None else np.log(cf / cf.sum())  # cls
            mi.bias = ops.assign(mi.bias, Tensor(b, ms.float32).view(-1))


def main():
    from mindspore import context

    context.set_context(mode=context.GRAPH_MODE, pynative_synchronize=True)
    cfg = "./config/models/yolov5s.yaml"
    model = Model(cfg, ch=3, nc=80, anchors=None)
    for p in model.trainable_params():
        print(p.name)


if __name__ == '__main__':
    main()
