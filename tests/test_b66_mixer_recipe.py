"""TODO B66: `--mixer-optimizer adamw_carla` trains the Atari Qwen mixer with the recipe the same network uses on CARLA."""
import torch

from atari_qwen.models.ez_model import EZV2Model
from atari_qwen.training.train_ez_offpolicy import carla_lr, mixer_param_groups, optim_state_cpu, parse_args
from src.training.wor_optim import build_warmup_cosine_scheduler


def test_defaults_unchanged():
    a = parse_args([])
    assert a.mixer_optimizer == "sgd" and a.mixer_warmup_frac == 0.05
    b = parse_args(["--mixer-optimizer", "adamw_carla", "--mixer-lr", "1e-4"])
    assert b.mixer_optimizer == "adamw_carla" and b.mixer_lr == 1e-4


def test_schedule_matches_carla():
    peak, epochs, steps = 3e-4, 4, 250                       # 1,000 steps, 50 of them warm-up
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=peak)
    sched = build_warmup_cosine_scheduler(opt, epochs, steps, warmup_frac=0.05, peak_lr=peak, verbose=False)
    for step in range(epochs * steps):
        assert abs(opt.param_groups[0]["lr"] - carla_lr(step, epochs * steps, peak, 0.05)) < 1e-12, step
        opt.step(); sched.step()


def test_decay_groups():
    m = EZV2Model(4, obs_channels=12, trunk="qwen", mixer_size="small")
    decay, no_decay = mixer_param_groups(m, 1e-4)
    ids = {id(q): n for n, q in m.named_parameters() if "mixer" in n}
    got = [id(q) for q in decay["params"]] + [id(q) for q in no_decay["params"]]
    assert sorted(got) == sorted(ids) and len(got) == len(set(got))     # every mixer tensor in exactly one group
    nd = {ids[id(q)] for q in no_decay["params"]}
    d = {ids[id(q)] for q in decay["params"]}
    assert "repr_mixer.pos" in nd and "repr_mixer.blocks.0.alpha_attn" in nd and "repr_mixer.blocks.0.norm1.weight" in nd
    assert "repr_mixer.out.bias" in nd and "repr_mixer.blocks.0.attn.q_proj.weight" in d and "dyn_mixer.out.weight" in d
    assert decay["weight_decay"] == 1e-4 and no_decay["weight_decay"] == 0.0


def test_state_copy_is_detached_and_loadable():
    lin = torch.nn.Linear(3, 2)
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-3)
    lin(torch.randn(4, 3)).sum().backward(); opt.step()
    snap = optim_state_cpu(opt)
    before = snap["state"][0]["exp_avg"].clone()
    lin(torch.randn(4, 3)).sum().backward(); opt.step()                  # training moves on after the save
    assert torch.equal(snap["state"][0]["exp_avg"], before)
    fresh = torch.optim.AdamW(torch.nn.Linear(3, 2).parameters(), lr=1e-3)
    fresh.load_state_dict(snap)
    assert torch.equal(fresh.state_dict()["state"][0]["exp_avg"], before)
