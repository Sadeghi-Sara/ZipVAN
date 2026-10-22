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

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LimitParamValue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, min: float, max: float):
        ctx.save_for_backward(x)
        assert max >= min
        ctx.min = min
        ctx.max = max
        return x

    @staticmethod
    def backward(ctx, x_grad: Tensor):
        (x,) = ctx.saved_tensors
        # where x < ctx.min, ensure all grads are negative (this will tend to make
        # x more positive).
        x_grad = x_grad * torch.where(
            torch.logical_and(x_grad > 0, x < ctx.min), -1.0, 1.0
        )
        # where x > ctx.max, ensure all grads are positive (this will tend to make
        # x more negative).
        x_grad *= torch.where(torch.logical_and(x_grad < 0, x > ctx.max), -1.0, 1.0)
        return x_grad, None, None


def limit_param_value(
    x: Tensor, min: float, max: float, prob: float = 0.6, training: bool = True
):
    # You apply this to (typically) an nn.Parameter during training to ensure that its
    # (elements mostly) stays within a supplied range.  This is done by modifying the
    # gradients in backprop.
    # It's not necessary to do this on every batch: do it only some of the time,
    # to save a little time.
    if training and random.random() < prob:
        return LimitParamValue.apply(x, min, max)
    else:
        return x


class BiasNorm(torch.nn.Module):
    """
    This is intended to be a simpler, and hopefully cheaper, replacement for
    LayerNorm.  The observation this is based on, is that Transformer-type
    networks, especially with pre-norm, sometimes seem to set one of the
    feature dimensions to a large constant value (e.g. 50), which "defeats"
    the LayerNorm because the output magnitude is then not strongly dependent
    on the other (useful) features.  Presumably the weight and bias of the
    LayerNorm are required to allow it to do this.

    Instead, we give the BiasNorm a trainable bias that it can use when
    computing the scale for normalization.  We also give it a (scalar)
    trainable scale on the output.

    Args:
       num_channels: the number of channels, e.g. 512.
       channel_dim: the axis/dimension corresponding to the channel,
         interpreted as an offset from the input's ndim if negative.
         This is NOT the num_channels; it should typically be one of
         {-2, -1, 0, 1, 2, 3}.
       log_scale: the initial log-scale that we multiply the output by; this
         is learnable.
       log_scale_min: minimum allowed value of log_scale
       log_scale_max: maximum allowed value of log_scale
       eps: added to the mean square before the square root.
    """

    def __init__(
        self,
        num_channels: int,
        channel_dim: int = -1,  # CAUTION: see documentation.
        log_scale: float = 1.0,
        log_scale_min: float = -1.5,
        log_scale_max: float = 1.5,
        eps: float = 1.0e-5,
    ) -> None:
        super(BiasNorm, self).__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.log_scale = nn.Parameter(torch.tensor(log_scale))
        self.bias = nn.Parameter(torch.empty(num_channels).normal_(mean=0, std=1e-4))

        self.log_scale_min = log_scale_min
        self.log_scale_max = log_scale_max
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[self.channel_dim] == self.num_channels
        channel_dim = self.channel_dim
        if channel_dim < 0:
            channel_dim += x.ndim
        bias = self.bias
        for _ in range(channel_dim + 1, x.ndim):
            bias = bias.unsqueeze(-1)

        log_scale = limit_param_value(
            self.log_scale,
            min=float(self.log_scale_min),
            max=float(self.log_scale_max),
            training=self.training,
        )
        var = torch.mean((x - bias) ** 2, dim=channel_dim, keepdim=True)
        return x * (var.add(self.eps).rsqrt() * log_scale.exp())


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
