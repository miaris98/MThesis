"""TODO B62: the anti-collapse fixes are opt-in. With default flags the trainer runs the unchanged EZ-V2 recipe."""
from atari_qwen.training.train_ez_offpolicy import parse_args


def test_fixes_default_off():
    a = parse_args([])
    assert a.stagnation_explore == 0 and a.policy_target_smoothing == 0.0
    assert a.stagnation_burst == 30


def test_fix_flags_parse():
    a = parse_args(["--stagnation-explore", "300", "--stagnation-burst", "20", "--policy-target-smoothing", "0.05"])
    assert (a.stagnation_explore, a.stagnation_burst, a.policy_target_smoothing) == (300, 20, 0.05)
