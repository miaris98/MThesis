"""TODO B64: the speed diagnostics are opt-in. With default flags the trainer runs without GPU syncs or a profiler."""
from atari_qwen.training.train_ez_offpolicy import parse_args


def test_profile_flags_default_off():
    a = parse_args([])
    assert a.profile_targets is False and a.profile_at == 0 and a.profile_updates == 20


def test_profile_flags_parse():
    a = parse_args(["--profile-targets", "--profile-at", "2050", "--profile-updates", "10"])
    assert (a.profile_targets, a.profile_at, a.profile_updates) == (True, 2050, 10)
