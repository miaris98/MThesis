"""TODO (2026-11-03): launch-command contract. Gate 1's first launch failed at startup because --reanalyze-merge needs --priority-alpha 0,
a rule the trainer only checked after building the environments and the model. The rule is now checked when the arguments are parsed, so a
bad command fails in milliseconds on the PC and never reaches a rented box. These tests pin the rule and the flags of the gate command."""
import pytest

from atari_qwen.training.train_ez_offpolicy import parse_args

GATE_FLAGS = ["--env-id", "BreakoutNoFrameskip-v4", "--trunk", "conv_transformer", "--mixer-size", "small", "--mixer-where", "pred",
              "--mixer-optimizer", "adamw", "--mixer-lr", "3e-4", "--mixer-qk-norm", "--num-simulations", "16", "--total-steps", "3000",
              "--reanalyze-cache", "--reanalyze-merge", "--compile-search"]


def test_the_gate_command_is_rejected_without_priority_alpha_zero():
    with pytest.raises(ValueError, match="priority-alpha 0"):
        parse_args(GATE_FLAGS)


def test_the_gate_command_with_priority_alpha_zero_parses():
    a = parse_args(GATE_FLAGS + ["--priority-alpha", "0"])
    assert a.reanalyze_merge and a.reanalyze_cache and a.priority_alpha == 0


def test_merge_needs_the_cache_flag_too():
    with pytest.raises(ValueError, match="reanalyze-cache"):
        parse_args(["--reanalyze-merge", "--priority-alpha", "0"])
