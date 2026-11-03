"""TODO (2026-11-03): pin the Gate 2 recipe (the SGD arm B of the 1,100-step screen). Gate 2 failed to match the brief's AdamW only after
the screen; these tests make the recipe a checked contract, so a flag change that silently moves a run off the screened recipe is caught on
the PC. The recipe: default SGD mixer, no QK-norm, mixer weight decay 0, batch norm, no simnorm, uniform replay, the reanalysis speed flags."""
from atari_qwen.training.train_ez_offpolicy import parse_args

GATE2 = ["--env-id", "BreakoutNoFrameskip-v4", "--trunk", "conv_transformer", "--mixer-where", "pred", "--num-simulations", "16",
         "--total-steps", "20000", "--priority-alpha", "0", "--eval-mode", "deferred", "--reanalyze-cache", "--reanalyze-merge",
         "--compile-search", "--norm", "batch", "--simnorm", "0", "--mixer-weight-decay", "0", "--seed", "1", "--mixer-size", "small"]


def test_gate2_recipe_is_the_screened_sgd_arm():
    a = parse_args(GATE2)
    assert a.mixer_optimizer == "sgd"
    assert a.mixer_qk_norm is False
    assert a.mixer_weight_decay == 0
    assert a.norm == "batch" and a.simnorm == 0
    assert a.priority_alpha == 0 and a.reanalyze_merge and a.reanalyze_cache


def test_gate2_recipe_for_both_variants_differs_only_in_size():
    small = vars(parse_args(GATE2))
    mini = vars(parse_args([*GATE2[:-1], "mini"]))
    assert {k for k in small if small[k] != mini[k]} == {"mixer_size"}
