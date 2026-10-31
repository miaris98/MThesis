"""EfficientZero V2 (Atari) networks, with an optional GTrXL token mixer (the thesis variant).

`trunk="resnet"` reproduces the EZ-V2 Atari model (Wang et al. 2024, github.com/Shengjiewang-Jason/
EfficientZeroV2, config ez/config/exp/atari_breakout.yaml) layer for layer: DownSample ResNet to a
64x6x6 state, conv dynamics with a learned action plane, value/policy heads over a 16-channel 1x1
reduction, value-prefix LSTM reward head, 601-bin categorical supports, SimSiam projection 1024-d.

`trunk="gtrxl"` keeps every head, loss and interface identical and inserts a gated-transformer token
mixer (GTrXL blocks over the 36 spatial tokens) after the representation and after each dynamics
step. The gate bias starts closed (GRU gate ~ identity), so at init it behaves like the resnet trunk
and has to earn its contribution - the comparison isolates the transformer, nothing else.

`trunk="qwen"` (TODO B61) puts our Qwen blocks (RMSNorm, attention, SwiGLU, alpha-gated residuals;
src/models/transformer/layers.py) in the same mixer slot. `mixer_size` picks the width: "small" matches
the GTrXL mixer (128 wide, 2 blocks, 4 heads), "30m" is the CARLA head's qwen30m shape (512, 8, 8).

`mixer_where` places the Qwen blocks: "both" (the default, as above), "repr" (representation only, TODO B68 D1), or "pred"
(TODO B69): the CARLA trunk's layout as the prediction function. The conv representation and dynamics are EZ-V2's, as the CNN
encoder is CARLA's, and the value and policy are read from a policy token by the Qwen trunk instead of by the conv heads.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from atari_qwen.models.gtrxl_layers import GTrXLBlock
from src.models.transformer.layers import QwenTransformerBlock, RMSNorm

# TODO B61: (dim, depth, heads, ffn_dim) of the token mixer; "30m" = qwen30m in qwen_wor_policy._MODEL_SIZES
MIXER_SIZES = {"small": (128, 2, 4, 512), "30m": (512, 8, 8, 1536)}


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
    """GTrXL (or Qwen, B61) blocks over the HxW spatial tokens of a (B, C, H, W) state; returns the same shape.

    Learned 2-D position embedding; GRU-gated residuals with a closed gate at init (bg_init=2 ->
    ~88% skip), so the mixer starts near identity on top of the conv state. Qwen blocks start with
    alpha 0.1 on both residual branches and end in an RMSNorm before the output projection."""
    def __init__(self, channels: int, hw: int, dim: int = 128, depth: int = 2, heads: int = 4,
                 bg_init: float = 2.0, out_init_std: float = 0.0, pos_mode: str = "abs",
                 block: str = "gtrxl", ffn_dim: Optional[int] = None, qk_norm: bool = False):
        super().__init__()
        assert pos_mode in ("abs", "sym"), pos_mode
        assert block in ("gtrxl", "qwen"), block
        assert block == "gtrxl" or pos_mode == "abs", "the Qwen block takes no attention bias"
        assert block == "qwen" or not qk_norm, "TODO B67: QK-norm is an option of the Qwen block only"
        self.pos_mode, self.heads, self.block = pos_mode, heads, block
        self.inp = nn.Linear(channels, dim)
        if pos_mode == "abs":
            self.pos = nn.Parameter(torch.zeros(1, hw, dim))
            nn.init.trunc_normal_(self.pos, std=0.02)
        else:
            # TODO B20: attention bias that depends only on the token offset (|dx|, dy), per head. On an s x s grid that is
            # s * (2s - 1) values (66 for s = 6). |dx| makes it mirror-symmetric; dy keeps the sign (rows are not mirrored).
            # Initialised local (conv-like): -0.5 * (|dx| + |dy|), as ConViT's gated positional attention starts local.
            s = int(round(hw ** 0.5))
            assert s * s == hw, "symbias needs a square token grid"
            rows, cols = torch.arange(hw) // s, torch.arange(hw) % s
            dx = (cols[:, None] - cols[None, :]).abs()          # (HW, HW), mirror-invariant
            dy = rows[:, None] - rows[None, :]                  # (HW, HW), -(s-1)..(s-1)
            self.register_buffer("rel_idx", (dx * (2 * s - 1) + (dy + s - 1)).long(), persistent=False)
            ii, dd = torch.arange(s)[:, None].repeat(1, 2 * s - 1), torch.arange(-(s - 1), s)[None, :].repeat(s, 1)
            init = -0.5 * (ii.float() + dd.abs().float()).reshape(-1)     # (s * (2s - 1),)
            self.rel_table = nn.Parameter(init[None, :].repeat(heads, 1))  # (heads, 66) for s = 6
        ffn_dim = ffn_dim or 4 * dim
        if block == "gtrxl":
            self.blocks = nn.ModuleList([GTrXLBlock(dim=dim, num_heads=heads, ffn_dim=ffn_dim, bg_init=bg_init)
                                         for _ in range(depth)])
        else:
            self.blocks = nn.ModuleList([QwenTransformerBlock(dim, num_heads=heads, ffn_dim=ffn_dim, qk_norm=qk_norm) for _ in range(depth)])
            self.norm_out = RMSNorm(dim)
        self.out = nn.Linear(dim, channels)
        # 0 = exact identity at init (the original). With SGD and weight decay the blocks behind a zero output projection
        # got no gradient and decayed to 0 (S-116); out_init_std > 0 lets gradient reach them from the first update (TODO B36).
        if out_init_std > 0:
            nn.init.normal_(self.out.weight, std=out_init_std)
        else:
            nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)            # (B, HW, C)
        if self.pos_mode == "abs":
            h, bias = self.inp(t) + self.pos, None
        else:
            h, bias = self.inp(t), self.rel_table[:, self.rel_idx]   # (heads, HW, HW) additive attention bias
        if self.block == "gtrxl":
            for blk in self.blocks:
                h = blk(h, bias)
        else:
            for blk in self.blocks:
                h = blk(h)
            h = self.norm_out(h)
        t = t + self.out(h)                           # zero-init output: exact identity at init
        return t.transpose(1, 2).reshape(B, C, H, W)


class PolicyTokenTrunk(nn.Module):
    """TODO B69: the CARLA trunk's layout (QwenWaypointTransformer, src/models/world_on_rails/qwen_wor_policy.py) as the prediction
    function. Each of the HxW state cells becomes a token (linear projection + learned position, as CARLA's vision tokens), a learned
    policy token is prepended, Qwen blocks and a final RMSNorm run over the sequence, and the value and policy logits are read from the
    policy token by Linear-GELU-Linear heads, as CARLA reads its waypoints. The heads' last layers start at zero (EZ-V2's init_zero:
    uniform prior and zero value at init, which the search relies on)."""
    def __init__(self, channels: int, hw: int, dim: int, depth: int, heads: int, ffn_dim: int, num_actions: int,
                 support_size: int, qk_norm: bool = False, no_pos: bool = False):
        super().__init__()
        self.inp = nn.Linear(channels, dim)
        self.no_pos = no_pos   # TODO B70b: --mixer-no-pos drops the learned position table from the tokens (the parameter stays, unused)
        self.pos = nn.Parameter(torch.zeros(1, hw, dim))
        self.policy_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.policy_token, std=0.02)
        self.blocks = nn.ModuleList([QwenTransformerBlock(dim, num_heads=heads, ffn_dim=ffn_dim, qk_norm=qk_norm) for _ in range(depth)])
        self.norm_out = RMSNorm(dim)
        self.value_head = nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Linear(256, support_size))
        self.policy_head = nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Linear(256, num_actions))
        for head in (self.value_head, self.policy_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def tokens(self, s: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) state -> (B, 1 + HW, dim) sequence, policy token first."""
        t = self.inp(s.flatten(2).transpose(1, 2)) + (0.0 if self.no_pos else self.pos)
        return torch.cat([self.policy_token.expand(s.shape[0], -1, -1), t], dim=1)

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.tokens(s)
        for blk in self.blocks:
            h = blk(h)
        policy_repr = self.norm_out(h)[:, 0]
        return self.value_head(policy_repr), self.policy_head(policy_repr)


def swap_batchnorm(module: nn.Module, skip=()) -> None:
    """Replace BatchNorm2d(C) by GroupNorm(1, C) (= LayerNorm over C,H,W) and BatchNorm1d(n) by
    LayerNorm(n) in every submodule except those whose top-level name is in `skip`.

    S-058 (S058b diagnosis): in eval mode - how acting, reanalyze and the target network all run -
    BatchNorm uses one set of running stats for every unroll depth, but the value-prefix LSTM's
    features differ strongly between depth 1 and depth 5. The reward head then predicted a phantom
    +0.25 reward for the first imagined step of any action (real 0.005), the search inflated the root
    value 3x and the policy collapsed. Layer statistics are per sample, so train == eval."""
    for name, child in list(module.named_children()):
        if name in skip:
            continue
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(1, child.num_features))
        elif isinstance(child, nn.BatchNorm1d):
            setattr(module, name, nn.LayerNorm(child.num_features))
        else:
            swap_batchnorm(child)


class EZV2Model(nn.Module):
    def __init__(self, action_dim: int, obs_channels: int = 12, num_channels: int = 64,
                 reduced_channels: int = 16, state_hw: int = 6, fc_layers=(32,),
                 lstm_hidden: int = 512, proj_hidden: int = 1024, proj_out: int = 1024,
                 head_hidden: int = 256, action_embed_dim: int = 16, support: Optional[DiscreteSupport] = None,
                 trunk: str = "resnet", mixer_dim: int = 128, mixer_depth: int = 2, norm: str = "batch",
                 mixer_out_init_std: float = 0.0, mixer_bg_init: float = 2.0, simnorm: int = 0,
                 mixer_pos: str = "abs", mixer_size: Optional[str] = None, mixer_qk_norm: bool = False,
                 mixer_where: str = "both", mixer_no_pos: bool = False):
        super().__init__()
        assert trunk in ("resnet", "gtrxl", "qwen")
        assert mixer_where in ("both", "repr", "pred")
        assert mixer_where != "pred" or trunk == "qwen", "TODO B69: the policy-token trunk is built from the Qwen block"
        pred_trunk = trunk == "qwen" and mixer_where == "pred"
        self.action_dim, self.C, self.hw, self.trunk = action_dim, num_channels, state_hw, trunk
        # TODO B22: SimNorm group size on the latent (0 = off, the EZ-V2 default). 8 = TD-MPC2's value.
        self.simnorm = simnorm
        assert simnorm == 0 or (num_channels * state_hw * state_hw) % simnorm == 0
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
        # prediction (value + policy); under mixer_where="pred" the policy-token trunk below replaces these conv heads
        if not pred_trunk:
            self.pred_res = ResidualBlock(num_channels, num_channels)
            self.v_conv, self.v_bn = nn.Conv2d(num_channels, reduced_channels, 1), nn.BatchNorm2d(reduced_channels)
            self.p_conv, self.p_bn = nn.Conv2d(num_channels, reduced_channels, 1), nn.BatchNorm2d(reduced_channels)
            self.v_fc = mlp(flat, list(fc_layers), S, init_zero=True)
            self.p_fc = mlp(flat, list(fc_layers), action_dim, init_zero=True)
        self.pred_mixer = None
        # SimSiam projection (target side) + head (online side)
        state_dim = num_channels * state_hw * state_hw
        self.proj = nn.Sequential(
            nn.Linear(state_dim, proj_hidden), nn.BatchNorm1d(proj_hidden), nn.ReLU(),
            nn.Linear(proj_hidden, proj_hidden), nn.BatchNorm1d(proj_hidden), nn.ReLU(),
            nn.Linear(proj_hidden, proj_out), nn.BatchNorm1d(proj_out))
        self.proj_head = nn.Sequential(
            nn.Linear(proj_out, head_hidden), nn.BatchNorm1d(head_hidden), nn.ReLU(),
            nn.Linear(head_hidden, proj_out))
        # thesis variant: transformer token mixing in representation and dynamics, or the policy-token trunk as the prediction (B69)
        if trunk != "resnet":
            hw2 = state_hw * state_hw
            heads, ffn = 4, None
            if mixer_size is not None:  # B61: a named size overrides mixer_dim / mixer_depth
                mixer_dim, mixer_depth, heads, ffn = MIXER_SIZES[mixer_size]
            kw = dict(heads=heads, bg_init=mixer_bg_init, out_init_std=mixer_out_init_std, pos_mode=mixer_pos, block=trunk, ffn_dim=ffn,
                      qk_norm=mixer_qk_norm)
        if pred_trunk:
            # TODO B69: no token mixer in the representation or the dynamics; the Qwen trunk is the prediction function
            self.repr_mixer = self.dyn_mixer = None
            self.pred_mixer = PolicyTokenTrunk(num_channels, hw2, mixer_dim, mixer_depth, heads, ffn or 4 * mixer_dim, action_dim, S,
                                               qk_norm=mixer_qk_norm, no_pos=mixer_no_pos)
        elif trunk != "resnet":
            self.repr_mixer = TokenMixer(num_channels, hw2, mixer_dim, mixer_depth, **kw)
            # TODO B68 D1: "repr" keeps the mixer in the representation only (the CARLA trunk's role: encode the observation); the
            # dynamics is then EZ-V2's conv + ResNet alone, which removes ~85% of the mixer work in the search
            self.dyn_mixer = TokenMixer(num_channels, hw2, mixer_dim, mixer_depth, **kw) if mixer_where == "both" else None
        self.lstm_hidden = lstm_hidden
        # norm: "batch" = EZ-V2; "heads" = LayerNorm in reward/value/policy heads; "all" = LayerNorm
        # everywhere except the SimSiam projection (its BatchNorm is what prevents collapse).
        assert norm in ("batch", "heads", "all")
        if norm == "heads":
            for n in ("rew_bn", "rew_bn2", "rew_fc", "pred_res", "v_bn", "p_bn", "v_fc", "p_fc"):
                sub = getattr(self, n, None)
                if sub is None:  # TODO B69: no conv prediction heads under mixer_where="pred"
                    continue
                if isinstance(sub, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    setattr(self, n, nn.LayerNorm(sub.num_features) if isinstance(sub, nn.BatchNorm1d)
                            else nn.GroupNorm(1, sub.num_features))
                else:
                    swap_batchnorm(sub)
        elif norm == "all":
            swap_batchnorm(self, skip=("proj", "proj_head"))

    # --- components ---------------------------------------------------------------------------
    def simnorm_latent(self, s: torch.Tensor) -> torch.Tensor:
        """B22: softmax within groups of `self.simnorm` features (TD-MPC2 SimNorm). Identity when off."""
        if self.simnorm == 0:
            return s
        shape = s.shape
        return F.softmax(s.reshape(shape[0], -1, self.simnorm), dim=-1).reshape(shape)

    def representation(self, obs: torch.Tensor) -> torch.Tensor:
        s = self.repr_res(self.down(obs))
        if self.trunk != "resnet" and self.repr_mixer is not None:
            s = self.repr_mixer(s)
        return self.simnorm_latent(s)

    def dynamics(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        plane = (a.float() / self.action_dim).view(-1, 1, 1, 1).expand(-1, 1, self.hw, self.hw)
        plane = F.relu(self.act_ln(self.act_conv(plane)))
        x = self.dyn_bn(self.dyn_conv(torch.cat([s, plane], dim=1)))
        x = self.dyn_res(F.relu(x + s))
        if self.trunk != "resnet" and self.dyn_mixer is not None:
            x = self.dyn_mixer(x)
        return self.simnorm_latent(x)

    def value_prefix(self, s: torch.Tensor, hidden):
        x = F.relu(self.rew_bn(self.rew_conv(s))).flatten(1).unsqueeze(0)
        x, hidden = self.lstm(x, hidden)
        x = F.relu(self.rew_bn2(x.squeeze(0)))
        return self.rew_fc(x), hidden

    def prediction(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pred_mixer is not None:  # TODO B69: value and policy read from the policy token, as CARLA reads its waypoints
            return self.pred_mixer(s)
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
