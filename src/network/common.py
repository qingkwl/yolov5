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

import copy
import math
import sys
from collections import namedtuple

import numpy as np
import mindspore as ms
import mindspore.numpy as mnp
from mindspore import Tensor, nn, ops
from mindspore.common.initializer import HeUniform

from src.general import make_divisible, empty

_SYNC_BN = False


def _calculate_fan_in_and_fan_out(shape):
    dimensions = len(shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = shape[1]
    num_output_fmaps = shape[0]
    receptive_field_size = 1
    if dimensions > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def _init_bias(conv_weight_shape):
    bias_init = None
    fan_in, _ = _calculate_fan_in_and_fan_out(conv_weight_shape)
    if fan_in != 0:
        bound = 1 / math.sqrt(fan_in)
        bias_init = Tensor(np.random.uniform(-bound, bound, conv_weight_shape[0]), dtype=ms.float32)
    return bias_init


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class ResizeNearestNeighbor(nn.Cell):
    def __init__(self, scale=2):
        super(ResizeNearestNeighbor, self).__init__()
        self.scale = scale

    # @ms.ms_function
    def construct(self, x):
        return ops.ResizeNearestNeighbor((x.shape[-2] * 2, x.shape[-1] * 2))(x)


class Bottleneck(nn.Cell):
    # Standard bottleneck
    # ch_in, ch_out, shortcut, groups, expansion
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.conv1 = Conv(c1, c_, 1, 1)
        self.conv2 = Conv(c_, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def construct(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        out = c2
        if self.add:
            out = x + out
        return out


class Concat(nn.Cell):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def construct(self, x):
        return ops.concat(x, self.d)


class Conv(nn.Cell):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s,
                              pad_mode="pad",
                              padding=autopad(k, p),
                              group=g,
                              has_bias=False,
                              weight_init=HeUniform(negative_slope=math.sqrt(5)))
        if _SYNC_BN:
            self.bn = nn.SyncBatchNorm(c2, momentum=0.1, eps=1e-5)
        else:
            self.bn = nn.BatchNorm2d(c2, momentum=0.1, eps=1e-5)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Cell) else nn.Identity()

    def construct(self, x):
        return self.act(self.bn(self.conv(x)))


class C3(nn.Cell):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.conv1 = Conv(c1, c_, 1, 1)
        self.conv2 = Conv(c1, c_, 1, 1)
        self.conv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.SequentialCell(
            [Bottleneck(c_, c_, shortcut, e=1.0) for _ in range(n)])
        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        c1 = self.conv1(x)
        c2 = self.m(c1)
        c3 = self.conv2(x)
        c4 = self.concat((c2, c3))
        c5 = self.conv3(c4)

        return c5


class SPPF(nn.Cell):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super(SPPF, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.conv1 = Conv(c1, c_, 1, 1)
        self.conv2 = Conv(c_ * 4, c2, 1, 1)
        self.concat = ops.Concat(axis=1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, pad_mode="same")

    def construct(self, x):
        x = self.conv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        y = self.conv2(self.concat((x, y1, y2, y3)))
        return y


class Contract(nn.Cell):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def construct(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Cell):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def construct(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)


class BaseCell(nn.Cell):
    def __init__(self, parameter):
        super(BaseCell, self).__init__()
        self.param = parameter


@ops.constexpr(reuse_result=True)
def get_convert_matrix():
    return Tensor(np.array([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]]),
                  dtype=ms.float32)


class Detect(nn.Cell):
    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.stride = None
        self.export = False
        self.end2end = False
        self.include_nms = False
        self.concat = False
        self.is_export = False

        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.anchors = ms.Parameter(Tensor(anchors, ms.float32).view(self.nl, -1, 2),
                                    requires_grad=False)  # shape(nl,na,2)
        self.anchor_grid = ms.Parameter(Tensor(anchors, ms.float32).view(self.nl, 1, -1, 1, 1, 2),
                                        requires_grad=False)  # shape(nl,1,na,1,1,2)

        self.m = nn.CellList([nn.Conv2d(x, self.no * self.na, 1,
                                        pad_mode="valid",
                                        has_bias=True,
                                        weight_init=HeUniform(negative_slope=math.sqrt(5)),
                                        bias_init=_init_bias((self.no * self.na, x, 1, 1))) for x in ch])  # output conv

    @staticmethod
    def convert(z):
        z = ops.concat(z, 1)
        box = z[:, :, :4]
        conf = z[:, :, 4:5]
        score = z[:, :, 5:]
        score *= conf
        convert_matrix = get_convert_matrix()
        box = ops.matmul(box, convert_matrix)
        return (box, score)

    def construct(self, x):
        z = ()  # inference output
        outs = ()
        for i in range(self.nl):
            out = self.m[i](x[i])  # conv
            if self.is_export:
                outs += (out,)
                continue
            bs, _, ny, nx = out.shape  # (bs,255,20,20)
            out = ops.Transpose()(out.view(bs, self.na, self.no, ny, nx), (0, 1, 3, 4, 2))  # (bs,3,20,20,85)
            out = out
            outs += (out,)

            if not self.training:  # inference
                grid_tensor = self._make_grid(nx, ny, out.dtype)

                y = ops.Sigmoid()(out)
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid_tensor) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z += (y.view(bs, -1, self.no),)

        return outs if self.training or self.is_export else (ops.concat(z, 1), outs)

    @staticmethod
    def _make_grid(nx=20, ny=20, dtype=ms.float32):
        xv, yv = ops.meshgrid(mnp.arange(nx), mnp.arange(ny))
        return ops.cast(ops.stack((xv, yv), 2).view((1, 1, ny, nx, 2)), dtype)


class Proto(nn.Cell):
    # YOLOv5 mask Proto module for segmentation models
    def __init__(self, c1, c_=256, c2=32):  # ch_in, number of protos, number of masks
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def construct(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class Segment(Detect):
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, anchors, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.CellList([nn.Conv2d(x, self.no * self.na, 1,
                                        pad_mode="valid",
                                        has_bias=True,
                                        weight_init=HeUniform(negative_slope=math.sqrt(5)),
                                        bias_init=_init_bias((self.no * self.na, x, 1, 1))) for x in ch])  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.construct

    def construct(self, x):
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])


def parse_model(d, ch, sync_bn=False):  # model_dict, input_channels(3)
    global _SYNC_BN
    _SYNC_BN = sync_bn
    print('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    layers_param = []
    for i, layer_cfg in enumerate(d['backbone'] + d['head']):
        c2, f, n, m, args = _parse_layer(ch, d, layer_cfg)    # ch out, from, number, module, args
        m_ = nn.SequentialCell([m(*args) for _ in range(n)]) if n > 1 else m(*args)
        t = str(m)  # module type
        num_params = sum([x.size for x in m_.get_parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, num_params  # attach index, 'from' index, type, number params
        layers_param.append((i, f, t, num_params))
        print('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, num_params, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.CellList(layers), sorted(save), layers_param


def _parse_layer(ch, d, layer_cfg):
    c2 = ch[-1]     # ch out
    f, n, m, args = layer_cfg
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    m = _get_layer_module(m) if isinstance(m, str) else m
    for j, a in enumerate(args):
        if isinstance(a, str):
            args[j] = int(a) if a.isnumeric() else d[a]
    n = max(round(n * gd), 1) if n > 1 else n  # depth gain
    if m in [nn.Conv2d, Conv, C3, SPPF, Bottleneck]:
        c1, c2 = ch[f], args[0]
        if c2 != no:  # if not output
            c2 = make_divisible(c2 * gw, 8)

        args = [c1, c2, *args[1:]]
        if m in [C3]:
            args.insert(2, n)  # number of repeats
            n = 1
    elif m is nn.BatchNorm2d:
        args = [ch[f]]
    elif m is Concat:
        c2 = sum([ch[x] for x in f])
    elif m in {Detect, Segment}:
        args.append([ch[x] for x in f])
        if isinstance(args[1], int):  # number of anchors
            args[1] = [list(range(args[1] * 2))] * len(f)
        if m is Segment:
            args[3] = make_divisible(args[3] * gw, 8)
    elif m is Contract:
        c2 = ch[f] * args[0] ** 2
    elif m is Expand:
        c2 = ch[f] // args[0] ** 2
    else:
        c2 = ch[f]
    layer_tuple = namedtuple('LayerTuple', ['channel_out', 'from', 'number', 'module', 'args'], rename=True)
    return layer_tuple(c2, f, n, m, args)


def _get_layer_module(m: str):
    """
    Args:
        m: str, class name defined in this file, or absolute path of class in other package

        Note:
            if m is the absolute path, the parent module must be imported in this file.
            e.g. nn.Conv2d, ms.nn.Conv2d
            nn or ms.nn must be imported firstly
    """
    if not isinstance(m, str):
        raise TypeError("Only support input of str.")
    path = m.split('.')
    parent = ".".join(path[:-1])
    cls = path[-1]
    if empty(parent):
        module = getattr(sys.modules[__name__], m, None)  # Get module from this file
    else:
        parent_module = getattr(sys.modules[__name__], parent, None)  # Get module from this file
        if parent_module is None:
            raise ImportError(f"No module named {parent}")
        module = getattr(parent_module, cls, None)
    if module is None:
        raise ImportError(f"No module named {cls} in {parent}")
    return module


class EMA(nn.Cell):
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        super(EMA, self).__init__()
        # Create EMA
        self.ema_model = copy.deepcopy(model)
        self.weights = ms.ParameterTuple(list(model.get_parameters()))
        self.ema_weights = self.weights.clone("ema", init='same')
        self.updates = ms.Parameter(Tensor(updates, ms.float32), requires_grad=False)  # number of EMA updates
        self.decay_value = decay
        self.assign = ops.Assign()
        self.hyper_map = ops.HyperMap()

    def decay(self, x):
        # decay exponential ramp (to help early epochs)
        return self.decay_value * (1 - ops.exp(ops.neg(x) / 2000))

    @ms.ms_function
    def update(self):
        # Update EMA parameters
        def update_param(d, ema_v, weight):
            tep_v = ema_v * d
            return self.assign(ema_v, weight * (1. - d) + tep_v)

        updates = ops.assign_add(self.updates, 1)
        d = self.decay(self.updates)
        success = self.hyper_map(ops.partial(update_param, d), self.ema_weights, self.weights)
        updates = ops.depend(updates, success)

        return updates

    @ms.ms_function
    def clone_from_model(self):
        updates = ops.assign_add(self.updates, 1)
        success = self.hyper_map(ops.assign, self.ema_weights, self.weights)
        updates = ops.depend(updates, success)
        return updates
