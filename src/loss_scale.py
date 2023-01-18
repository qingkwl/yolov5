# Copyright 2021 Huawei Technologies Co., Ltd
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
"""dynamic loss scale """
import mindspore as ms
import mindspore.context as context
from mindspore.context import ParallelMode
from mindspore import nn
from mindspore.nn import TrainOneStepCell
from mindspore.nn import Cell
from mindspore import Tensor, RowTensor
from mindspore import Parameter
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype

GRADIENT_CLIP_TYPE = 0
GRADIENT_CLIP_VALUE = 1.0
from mindspore import ops


class ClipGradients(Cell):
    """
    Clip gradients.
    Inputs:
        grads (tuple[Tensor]): Gradients.
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.

    Outputs:
        tuple[Tensor], clipped gradients.
    """

    def __init__(self):
        """__init__"""
        super(ClipGradients, self).__init__()
        self.clip_by_norm = nn.ClipByNorm()
        self.cast = P.Cast()
        self.dtype = P.DType()

    def construct(self,
                  grads,
                  clip_type,
                  clip_value):
        """construct"""
        if clip_type not in (0, 1):
            return grads

        new_grads = ()
        for grad in grads:
            dt = self.dtype(grad)
            if clip_type == 0:
                t = C.clip_by_value(grad, self.cast(F.tuple_to_array((-clip_value,)), dt),
                                    self.cast(F.tuple_to_array((clip_value,)), dt))
            else:
                t = self.clip_by_norm(grad, self.cast(F.tuple_to_array((clip_value,)), dt))
            new_grads = new_grads + (t,)
        return new_grads


class TrainOneStepGrad(nn.Cell):
    r"""
    Network training with loss scaling.

    This is a training step with loss scaling. It takes a network, an optimizer and possibly a scale update
    Cell as args. The loss scale value can be updated in both host side or device side. The
    TrainOneStepWithLossScaleCell will be compiled to be graph which takes `*inputs` as input data.
    The Tensor type of `scale_sense` is acting as loss scaling value. If you want to update it on host side,
    the value must be provided. If  the Tensor type of `scale_sense` is not given, the loss scale update logic
    must be provied by Cell type of `scale_sense`.

    Args:
        network (Cell): The training network. The network only supports single output.
        optimizer (Cell): Optimizer for updating the weights.
        scale_sense (Union[Tensor, Cell]): If this value is Cell type, the loss scaling update logic cell.If this value
                                          is Tensor type, Tensor with shape :math:`()` or :math:`(1,)`.

    Inputs:
        - **(*inputs)** (Tuple(Tensor)) - Tuple of input tensors with shape :math:`(N, \ldots)`.

    Outputs:
        Tuple of 3 Tensor, the loss, overflow flag and current loss scaling value.

        - **loss** (Tensor) -  Tensor with shape :math:`()`.
        - **overflow** (Tensor) -  Tensor with shape :math:`()`, type is bool.
        - **loss scaling value** (Tensor) -  Tensor with shape :math:`()`

    Raises:
        TypeError: If `scale_sense` is neither Cell nor Tensor.
        ValueError: If shape of `scale_sense` is neither (1,) nor ().

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    def __init__(self, yolo_net, compute_loss, ema, optimizer, opt, loss_scaler=None, sens=1.0):
        super(TrainOneStepGrad, self).__init__()
        self.ema = ema
        self.enable_ema = False if ema is None else True
        self.rank_size = opt.rank_size
        self.yolo_net = yolo_net
        self.yolo_net.set_train(True)
        self.compute_loss = compute_loss
        ms.amp.auto_mixed_precision(self.compute_loss, amp_level=opt.ms_amp_level)
        ms.amp.auto_mixed_precision(self.yolo_net, amp_level=opt.ms_amp_level)

        self.loss_scaler = loss_scaler
        if loss_scaler is None:
            from mindspore.amp import StaticLossScaler
            self.loss_scaler = StaticLossScaler(sens)
        self.optimizer = optimizer
        self.optimizer.set_train(True)
        if opt.is_distributed:
            mean = context.get_auto_parallel_context("gradients_mean")
            degree = context.get_auto_parallel_context("device_num")
            self.grad_reducer = nn.DistributedGradReducer(self.optimizer.parameters, mean, degree)
        else:
            self.grad_reducer = ops.functional.identity
        self.sens = sens
        self.grad_fn = ops.GradOperation(get_by_list=True, sens_param=True)(self.get_net_loss,
                                                                            self.optimizer.parameters)

    def get_net_loss(self, x, label):
        x = x / 255.0
        pred = self.yolo_net(x)
        loss, loss_items = self.compute_loss(pred, label)
        loss *= self.rank_size
        loss_items = ops.stop_gradient(loss_items)
        return loss, loss_items

    def construct(self, x, label):
        loss, loss_items = self.get_net_loss(x, label)
        sens1, sens2 = ops.fill(loss.dtype, loss.shape, self.sens), \
                       ops.fill(loss_items.dtype, loss_items.shape, self.sens)
        grads = self.grad_fn(x, label, (sens1, sens2))
        grads = self.grad_reducer(grads)
        grads = self.loss_scaler.unscale(grads)
        # grads_finite = all_finite(grads)
        loss = ops.depend(loss, self.optimizer(grads))
        loss = ops.depend(loss, self.ema.update())
        return loss, loss_items
