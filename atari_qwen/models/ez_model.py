"""EfficientZero V2 (Atari) networks, with an optional GTrXL token mixer (the thesis variant).

`trunk="resnet"` reproduces the EZ-V2 Atari model (Wang et al. 2024, github.com/Shengjiewang-Jason/
EfficientZeroV2, config ez/config/exp/atari_breakout.yaml) layer for layer: DownSample ResNet to a
64x6x6 state, conv dynamics with a learned action plane, value/policy heads over a 16-channel 1x1
reduction, value-prefix LSTM reward head, 601-bin categorical supports, SimSiam projection 1024-d.

`trunk="gtrxl"` keeps every head, loss and interface identical and inserts a gated-transformer token
mixer (GTrXL blocks over the 36 spatial tokens) after the representation and after each dynamics
step. The gate bias starts closed (GRU gate ~ identity), so at init it behaves like the resnet trunk
and has to earn its contribution - the comparison isolates the transformer, nothing else.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from atari_qwen.models.gtrxl_layers import GTrXLBlock


# ------------------------------------------------------------------------------------------------
# Categorical support (MuZero appendix F / EZ-V2 DiscreteSupport, Atari branch: h-transform, then a
# unit-spaced support over [-300, 300] -> 601 bins).
# ------------------------------------------------------------------------------------------------
class DiscreteSupport:
    def __init__(self, vmin: float = -300.0, vmax: float = 300.0, eps: float = 0.001):
        self.vmin, self.vmax, self.eps = vmin, vmax, eps
        self.size = int(vmax - vmin) + 1

    def h(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * (torch.sqrt(x.abs() + 1) - 1) + self.eps * x

    def h_inv(self, x: torch.Tensor) -> torch.Tensor:
        e = self.eps
        return torch.sign(x) * (((torch.sqrt(1 + 4 * e * (x.abs() + 1 + e)) - 1) / (2 * e)) ** 2 - 1)

    def scalar_to_vector(self, x: torch.Tensor) -> torch.Tensor:
        """Two-hot target over the support, any leading shape -> (..., size)."""
        x = self.h(x.float()).clamp(self.vmin, self.vmax)
        low = x.floor()
        p_high = x - low
        low_idx = (low - self.vmin).long()
        high_idx = (low_idx + 1).clamp(max=self.size - 1)
        out = torch.zeros(*x.shape, self.size, device=x.device)
        out.scatter_add_(-1, low_idx.unsqueeze(-1), (1 - p_high).unsqueeze(-1))
        out.scatter_add_(-1, high_idx.unsqueeze(-1), p_high.unsqueeze(-1))
        return out

    def vector_to_scalar(self, logits: torch.Tensor) -> torch.Tensor:
        support = torch.arange(self.vmin, self.vmax + 1, device=logits.device, dtype=torch.float32)
        x = (torch.softmax(logits.float(), dim=-1) * support).sum(-1)
        return self.h_inv(x)


# ------------------------------------------------------------------------------------------------
# EZ-V2 building blocks (ez/agents/models/layer.py, base_model.py)
# ------------------------------------------------------------------------------------------------
def conv3x3(cin: int, cout: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, cin: int, cout: int, downsample: Optional[nn.Module] = None, stride: int = 1):
        super().__init__()
        self.conv1, self.bn1 = conv3x3(cin, cout, stride), nn.BatchNorm2d(cout)
        self.conv2, self.bn2 = conv3x3(cout, cout), nn.BatchNorm2d(cout)
        self.downsample = downsample

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + idt)


def mlp(sizes_in: int, hidden: list, out: int, init_zero: bool = False) -> nn.Sequential:
    sizes = [sizes_in] + list(hidden) + [out]
    layers = []
    for i in range(len(sizes) - 1):
        if i < len(sizes) - 2:
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.BatchNorm1d(sizes[i + 1]), nn.ELU()]
        else:
            layers += [nn.Linear(sizes[i], sizes[i + 1])]
    if init_zero:
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
    return nn.Sequential(*layers)


class DownSample(nn.Module):
    """96x96 -> 6x6 (two strided convs + two avg pools, one res block per stage)."""
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.conv1, self.bn1 = nn.Conv2d(cin, cout // 2, 3, 2, 1, bias=False), nn.BatchNorm2d(cout // 2)
        self.res1 = ResidualBlock(cout // 2, cout // 2)
        conv2 = nn.Conv2d(cout // 2, cout, 3, 2, 1, bias=False)
        self.down = ResidualBlock(cout // 2, cout, downsample=conv2, stride=2)
        self.res2 = ResidualBlock(cout, cout)
        self.pool1 = nn.AvgPool2d(3, 2, 1)
        self.res3 = ResidualBlock(cout, cout)
        self.pool2 = nn.AvgPool2d(3, 2, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res2(self.down(self.res1(x)))
        x = self.res3(self.pool1(x))
        return self.pool2(x)


class TokenMixer(nn.Module):
    """GTrXL blocks over the HxW spatial tokens of a (B, C, H, W) state; returns the same shape.

    Learned 2-D position embedding; GRU-gated residuals with a closed gate at init (bg_init=2 ->
    ~88% skip), so the mixer starts near identity on top of the conv state."""
    def __init__(self, channels: int, hw: int, dim: int = 128, depth: int = 2, heads: int = 4,
                 bg_init: float = 2.0):
        super().__init__()
        self.inp = nn.Linear(channels, dim)
        self.pos = nn.Parameter(torch.zeros(1, hw, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([GTrXLBlock(dim=dim, num_heads=heads, ffn_dim=4 * dim, bg_init=bg_init)
                                     for _ in range(depth)])
        self.out = nn.Linear(dim, channels)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)            # (B, HW, C)
        h = self.inp(t) + self.pos
        for blk in self.blocks:
            h = blk(h)
        t = t + self.out(h)                           # zero-init output: exact identity at init
        return t.transpose(1, 2).reshape(B, C, H, W)


class EZV2Model(nn.Module):
    def __init__(self, action_dim: int, obs_channels: int = 12, num_channels: int = 64,
                 reduced_channels: int = 16, state_hw: int = 6, fc_layers=(32,),
                 lstm_hidden: int = 512, proj_hidden: int = 1024, proj_out: int = 1024,
                 head_hidden: int = 256, action_embed_dim: int = 16, support: Optional[DiscreteSupport] = None,
                 trunk: str = "resnet", mixer_dim: int = 128, mixer_depth: int = 2):
        super().__init__()
        assert trunk in ("resnet", "gtrxl")
        self.action_dim, self.C, self.hw, self.trunk = action_dim, num_channels, state_hw, trunk
        self.support = support or DiscreteSupport()
        S = self.support.size
        flat = reduced_channels * state_hw * state_hw

        # representation
        self.down = DownSample(obs_channels, num_channels)
        self.repr_res = ResidualBlock(num_channels, num_channels)
        # dynamics (action plane a/|A| -> 1x1 conv embedding -> LayerNorm -> ReLU, concat, conv+BN+res)
        self.act_conv = nn.Conv2d(1, action_embed_dim, 1)
        self.act_ln = nn.LayerNorm([action_embed_dim, state_hw, state_hw])
        self.dyn_conv = conv3x3(num_channels + action_embed_dim, num_channels)
        self.dyn_bn = nn.BatchNorm2d(num_channels)
        self.dyn_res = ResidualBlock(num_channels, num_channels)
        # value-prefix (reward) LSTM head
        self.rew_conv, self.rew_bn = nn.Conv2d(num_channels, reduced_channels, 1), nn.BatchNorm2d(reduced_channels)
        self.lstm = nn.LSTM(input_size=flat, hidden_size=lstm_hidden)
        self.rew_bn2 = nn.BatchNorm1d(lstm_hidden)
        self.rew_fc = mlp(lstm_hidden, list(fc_layers), S, init_zero=True)
        # prediction (value + policy)
        self.pred_res = ResidualBlock(num_channels, num_channels)
        self.v_conv, self.v_bn = nn.Conv2d(num_channels, reduced_channels, 1), nn.BatchNorm2d(reduced_channels)
        self.p_conv, self.p_bn = nn.Conv2d(num_channels, reduced_channels, 1), nn.BatchNorm2d(reduced_channels)
        self.v_fc = mlp(flat, list(fc_layers), S, init_zero=True)
        self.p_fc = mlp(flat, list(fc_layers), action_dim, init_zero=True)
        # SimSiam projection (target side) + head (online side)
        state_dim = num_channels * state_hw * state_hw
        self.proj = nn.Sequential(
            nn.Linear(state_dim, proj_hidden), nn.BatchNorm1d(proj_hidden), nn.ReLU(),
            nn.Linear(proj_hidden, proj_hidden), nn.BatchNorm1d(proj_hidden), nn.ReLU(),
            nn.Linear(proj_hidden, proj_out), nn.BatchNorm1d(proj_out))
        self.proj_head = nn.Sequential(
            nn.Linear(proj_out, head_hidden), nn.BatchNorm1d(head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, proj_out))
        # thesis variant: transformer token mixing in representation and dynamics
        if trunk == "gtrxl":
            hw2 = state_hw * state_hw
            self.repr_mixer = TokenMixer(num_channels, hw2, mixer_dim, mixer_depth)
            self.dyn_mixer = TokenMixer(num_channels, hw2, mixer_dim, mixer_depth)
        self.lstm_hidden = lstm_hidden

    # --- components ---------------------------------------------------------------------------
    def representation(self, obs: torch.Tensor) -> torch.Tensor:
        s = self.repr_res(self.down(obs))
        if self.trunk == "gtrxl":
            s = self.repr_mixer(s)
        return s

    def dynamics(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        plane = (a.float() / self.action_dim).view(-1, 1, 1, 1).expand(-1, 1, self.hw, self.hw)
        plane = F.relu(self.act_ln(self.act_conv(plane)))
        x = self.dyn_bn(self.dyn_conv(torch.cat([s, plane], dim=1)))
        x = self.dyn_res(F.relu(x + s))
        if self.trunk == "gtrxl":
            x = self.dyn_mixer(x)
        return x

    def value_prefix(self, s: torch.Tensor, hidden):
        x = F.relu(self.rew_bn(self.rew_conv(s))).flatten(1).unsqueeze(0)
        x, hidden = self.lstm(x, hidden)
        x = F.relu(self.rew_bn2(x.squeeze(0)))
        return self.rew_fc(x), hidden

    def prediction(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.pred_res(s)
        v = self.v_fc(F.relu(self.v_bn(self.v_conv(x))).flatten(1))
        p = self.p_fc(F.relu(self.p_bn(self.p_conv(x))).flatten(1))
        return v, p

    def project(self, s: torch.Tensor, with_grad: bool = True) -> torch.Tensor:
        z = self.proj(s.flatten(1))
        return self.proj_head(z) if with_grad else z.detach()

    def init_hidden(self, n: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(1, n, self.lstm_hidden, device=device)
        return (z, z.clone())

    # --- inference API used by search / training ----------------------------------------------
    def initial_inference(self, obs: torch.Tensor):
        """-> state, value logits, policy logits"""
        s = self.representation(obs)
        v, p = self.prediction(s)
        return s, v, p

    def recurrent_inference(self, s: torch.Tensor, a: torch.Tensor, hidden):
        """-> next state, value-prefix logits, value logits, policy logits, lstm hidden"""
        s2 = self.dynamics(s, a)
        vp, hidden = self.value_prefix(s2, hidden)
        v, p = self.prediction(s2)
        return s2, vp, v, p, hidden
