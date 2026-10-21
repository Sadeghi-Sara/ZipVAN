# Copyright    2022-2023  Xiaomi Corp.        (authors: Daniel Povey)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from icefall, egs/librispeech/ASR/zipformer/scaling.py: only the modules
# used here are kept, and the Swoosh activations are written with softplus.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SwooshL(torch.nn.Module):
    """swoosh_l(x) = log(1 + exp(x - 4)) - 0.08x - 0.035"""

    def forward(self, x: Tensor) -> Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            x32 = x.float()
            out = F.softplus(x32 - 4.0) - 0.08 * x32 - 0.035
            return out.to(x.dtype)


class SwooshR(torch.nn.Module):
    """swoosh_r(x) = log(1 + exp(x - 1)) - 0.08x - 0.313261687"""

    def forward(self, x: Tensor) -> Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            x32 = x.float()
            out = F.softplus(x32 - 1.0) - 0.08 * x32 - 0.313261687
            return out.to(x.dtype)
