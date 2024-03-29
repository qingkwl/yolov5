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

from abc import ABCMeta, abstractmethod
from typing import List

import numpy as np


class ModelBase(metaclass=ABCMeta):
    """
    base class for model load and infer
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = None

    def __del__(self):
        if hasattr(self, "model") and self.model:
            del self.model

    @abstractmethod
    def infer(self, x: np.numarray) -> List[np.numarray]:
        """
        model inference, just for single input
        Args:
            x: np img

        Returns:

        """

    @abstractmethod
    def _init_model(self):
        pass
