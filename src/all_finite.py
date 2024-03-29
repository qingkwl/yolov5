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

import mindspore as ms


def compare_version(v1, v2="1.9.0"):
    """
    :param v1: version, format like 1.8.1
    :param v2: version, format like 1.8.1
    :return: v1 </=/> v1, return -1/0/1
    """

    l1 = str(v1).split(".")
    l2 = str(v2).split(".")
    for i in range(min(len(l1), len(l2))):
        if int(l1[i]) == int(l2[i]):
            pass
        elif int(l1[i]) < int(l2[i]):
            return -1
        else:
            return 1
    if len(l1) == len(l2):
        return 0
    if len(l1) < len(l2):
        return -1
    return 1


def get_all_finite():
    if compare_version(ms.__version__) < 0:
        from mindspore import context, ops

        _ascend_target = context.get_context("device_target") == "Ascend"
        _gpu_target = context.get_context("device_target") == "GPU"
        npu_alloc_float_status = ops.NPUAllocFloatStatus()
        npu_clear_float_status = ops.NPUClearFloatStatus()
        if context.get_context("device_target") == "Ascend":
            _status = npu_alloc_float_status()
            _ = npu_clear_float_status(_status)
        else:
            _status = None
        _hypermap = ops.HyperMap()
        _partial = ops.Partial()

        def _is_finite(inputs):
            if _gpu_target:
                return ops.FloatStatus()(inputs)[0] == 0
            status = ops.isfinite(inputs)
            return status.all()

        def all_finite(inputs):
            if _ascend_target:
                status = ops.depend(_status, inputs)
                get_status = ops.NPUGetFloatStatus()(status)
                status = ops.depend(status, get_status)
                status_finite = status.sum() == 0
                _ = ops.NPUClearFloatStatus()(status)
                return status_finite
            outputs = _hypermap(_partial(_is_finite), inputs)
            return ops.stack(outputs).all()
    else:
        from mindspore.amp import all_finite
    return all_finite


all_finite_device = get_all_finite()
