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

from mindspore import boost, nn, ops
from mindspore.boost.boost_cell_wrapper import BoostTrainOneStepCell
from mindspore.common import dtype as mstype
from mindspore.nn.wrap.cell_wrapper import (TrainOneStepCell,
                                            _pipeline_clear_grad,
                                            _TrainPipelineAccuStepCell)
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.train.amp import (_add_loss_network, _check_kwargs,
                                 _check_level, _config_level,
                                 _do_keep_batchnorm_fp32, _get_pipeline_stages,
                                 auto_mixed_precision, validator)
from mindspore.amp import all_finite
from src.general import LOGGER


GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 10.0
clip_grad = C.MultitypeFuncGraph("clip_grad")


@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor], clipped gradients.
    """
    if clip_type not in (0, 1):
        return grad
    dt = F.dtype(grad)
    if clip_type == 0:
        new_grad = C.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                   F.cast(F.tuple_to_array((clip_value,)), dt))
    else:
        new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
    return new_grad


class _BoostTrainPipelineAccuStepCell(_TrainPipelineAccuStepCell):
    def __init__(self, network, ema, optimizer, amp_loss_scaler, sens=1.0, enable_clip_grad=True):
        super(_BoostTrainPipelineAccuStepCell, self).__init__(network, optimizer, sens=sens)
        self.ema = ema
        self.use_loss_scaler = amp_loss_scaler is not None
        self.enable_clip_grad = enable_clip_grad
        self.amp_loss_scaler = amp_loss_scaler
        LOGGER.info(f"Enable loss scale: {self.use_loss_scaler}")
        LOGGER.info(f"Enable enable_clip_grad: {self.enable_clip_grad}")

    def construct(self, *inputs):
        weights = self.weights
        loss = self.network(*inputs)
        sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*inputs, sens)
        grads = self.grad_reducer(grads)
        is_finite = all_finite(grads)
        if is_finite:
            if self.use_loss_scaler:
                grads = self.amp_loss_scaler.unscale(grads)
            if self.enable_clip_grad:
                grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
            accu_grads = ops.depend(self.accu_grads, grads)
            if self.opt_shard:
                succ = self.optimizer(grads)
            else:
                succ = self.optimizer(accu_grads)
            loss = ops.depend(loss, succ)
            clear = self.hyper_map(_pipeline_clear_grad, accu_grads, grads)
            loss = ops.depend(loss, clear)
            loss = F.depend(loss, self.ema.update())
        else:
            ops.print_("[WARNING] Grad infinite. Skip updating.")
        return loss


class _TrainOneStepCell(TrainOneStepCell):
    def __init__(self, network, ema, optimizer, amp_loss_scaler, sens=1.0, enable_clip_grad=True):
        super(_TrainOneStepCell, self).__init__(network, optimizer, sens=sens)
        self.ema = ema
        self.use_loss_scaler = amp_loss_scaler is not None
        self.amp_loss_scaler = amp_loss_scaler
        self.enable_clip_grad = enable_clip_grad
        self.hyper_map = ops.HyperMap()
        LOGGER.info(f"Enable loss scale: {self.use_loss_scaler}")
        LOGGER.info(f"Enable enable_clip_grad: {self.enable_clip_grad}")

    def construct(self, *inputs):
        loss = self.network(*inputs)
        sens = F.fill(loss.dtype, loss.shape, self.sens)
        grads = self.grad(self.network, self.weights)(*inputs, sens)
        grads = self.grad_reducer(grads)
        is_finite = all_finite(grads)
        if is_finite:
            if self.use_loss_scaler:
                grads = self.amp_loss_scaler.unscale(grads)
            if self.enable_clip_grad:
                grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
            loss = F.depend(loss, self.optimizer(grads))
            loss = F.depend(loss, self.ema.update())
        else:
            ops.print_("[WARNING] Grad infinite. Skip updating.")
        return loss


class _BoostTrainOneStepCell(BoostTrainOneStepCell):
    def __init__(self, network, ema, optimizer, amp_loss_scaler, sens=1.0, enable_clip_grad=True):
        super(_BoostTrainOneStepCell, self).__init__(network, optimizer, sens)
        self.ema = ema
        self.use_loss_scaler = amp_loss_scaler is not None
        self.amp_loss_scaler = amp_loss_scaler
        self.enable_clip_grad = enable_clip_grad
        print(f"[INFO] Enable loss scale: {self.use_loss_scaler}", flush=True)
        print(f"[INFO] Enable enable_clip_grad: {self.enable_clip_grad}", flush=True)

    def construct(self, *inputs):
        is_finite = True
        if self.freeze:
            loss = self.gradient_freeze_process(*inputs)
            is_finite = ops.isfinite(loss)
            if not is_finite:
                ops.print_("[WARNING] Loss infinite.")
        else:
            loss = self.network(*inputs)
            sens = F.fill(loss.dtype, loss.shape, self.sens)
            grads = self.grad(self.network, self.weights)(*inputs, sens)
            grads = self.grad_reducer(grads)
            is_finite = all_finite(grads)
            if is_finite:
                if self.use_loss_scaler:
                    grads = self.amp_loss_scaler.unscale(grads)
                if self.use_grad_accumulation:
                    loss = self.gradient_accumulation_process(loss, grads, sens, *inputs)
                else:
                    if self.enable_dim_reduce:
                        loss = F.depend(loss, self.dim_reduce(loss, grads, sens, self.weights, self.weights_clone, *inputs))
                    elif self.enable_adasum:
                        loss = F.depend(loss, self.adasum_process(loss, grads))
                    else:
                        if self.enable_clip_grad:
                            grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
                        loss = F.depend(loss, self.optimizer(grads))
            else:
                ops.print_("[WARNING] Grad infinite. Skip updating.")
        if is_finite:
            loss = F.depend(loss, self.ema.update())
        return loss


def build_train_network(network, ema, optimizer, loss_fn=None, level='O0', boost_level='O0',
                        amp_loss_scaler=None, sens=1.0, enable_clip_grad=True, **kwargs):
    validator.check_value_type('optimizer', optimizer, (nn.Optimizer, boost.FreezeOpt,
                                                        nn.AdaSumByGradWrapCell, nn.AdaSumByDeltaWeightWrapCell))

    level, enable_boost = _check_level(level, boost_level)

    _check_kwargs(kwargs)
    config = dict(_config_level.get(level), **kwargs)

    if config["cast_model_type"] == mstype.float16:
        network.to_float(mstype.float16)

        if config["keep_batchnorm_fp32"]:
            _do_keep_batchnorm_fp32(network)
    elif not config["keep_batchnorm_fp32"] and level == "O2":
        network.to_float(mstype.float16)
    elif config["cast_model_type"] == mstype.float32 and level in ("O2", "O3"):
        pass
    else:
        auto_mixed_precision(network, level)

    if loss_fn:
        network = _add_loss_network(network, loss_fn, config["cast_model_type"])

    loss_scale = 1.0
    sens = sens if config["loss_scale_manager"] is not None else loss_scale
    if _get_pipeline_stages() > 1:
        network = _BoostTrainPipelineAccuStepCell(network, ema, optimizer, amp_loss_scaler,
                                                  sens=sens, enable_clip_grad=enable_clip_grad).set_train()
    elif enable_boost:
        network = _BoostTrainOneStepCell(network, ema, optimizer, amp_loss_scaler,
                                         sens=sens, enable_clip_grad=enable_clip_grad).set_train()
    else:
        network = _TrainOneStepCell(network, ema, optimizer, amp_loss_scaler,
                                    sens=sens, enable_clip_grad=enable_clip_grad).set_train()
    return network
