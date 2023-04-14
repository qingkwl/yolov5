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

from .model_base import ModelBase

class LiteModel(ModelBase):
    def __init__(self, model_path, device_id = 0):
        super().__init__()
        self.model_path = model_path
        self.device_id = device_id

    def _init_model(self):
        import mindspore_lite as mslite

        context = mslite.Context()
        context.target = ['ascend']
        context.ascend.device_id = self.device_id

        self.model = mslite.Model()
        self.model.build_from_from_file(self.model_path, mslite.ModelType.MINDIR, context)

    def infer(self, input):
        inputs = self.model.get_inputs()
        self.model.resize(inputs, [list(input.shape)])
        inputs[0].set_data_from_numpy(input)

        outputs = self.model.predict(inputs)
        outputs = [outputs.get_data_to_numpy() for output in outputs]
        return outputs
