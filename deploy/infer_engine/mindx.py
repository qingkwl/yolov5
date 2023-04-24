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

import numpy as np
from .model_base import ModelBase


class MindXModel(ModelBase):
    def __init__(self, model_path, device_id=0):
        super().__init__()
        self.model_path = model_path
        self.device_id = device_id

        from mindx.sdk import base, Tensor
        self.base = base
        self.tensor = Tensor

        self._init_model()

    def infer(self, x):
        inputs = self.tensor(x)
        outputs = self.model.infer(inputs)
        list([output.to_host() for output in outputs])
        outputs = [np.array(output) for output in outputs]
        return outputs

    def _init_model(self):
        self.base.mx_init()
        self.model = self.base.model(self.model_path, self.device_id)
        if not self.model:
            raise ValueError(f"The model file {self.model_path} load failed.")
