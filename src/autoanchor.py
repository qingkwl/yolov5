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

# Auto-anchor utils

from mindspore import ops


def check_anchor_order(m):
    # Check anchor order against stride order for YOLO Detect() module m, and correct if necessary
    a = ops.ReduceProd()(m.anchor_grid, -1).view(-1) # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if ops.Sign()(da) != ops.Sign()(ds): # same order
        print('Reversing anchor order')
        m.anchors[:] = ops.ReverseV2(axis=0)(m.anchors)
        m.anchor_grid[:] = ops.ReverseV2(axis=0)(m.anchor_grid)
