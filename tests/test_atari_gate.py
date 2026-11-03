"""TODO (2026-11-03): the Atari Gate 1 acceptance check as code. The gate used to exist only as prose in a brief, so a run whose attention
had collapsed could still be called healthy (Gate 1 seeds 1-3 never printed attention entropy). These tests pin the gate's verdict on lines
in the trainer's own log format, and that a missing measurement fails the gate instead of passing it."""
from scripts.analysis.atari_gate_check import GATE, parse_log_line, gate_verdict

ATTN_HEALTHY = ("env   1000/3000 upd   1000/3000 | lr 0.200 | loss v 0.460 p 7.293 r 0.065 c -4.396 | gn 0.22 | train-life-ret(20) 0.10 | "
                "V0 0.102 tgt 0.139 | |mixer W| 109.61 | cache hit s/v 0.99/1.00 | time c/t/u 17/6/77% | attn H 0.86/0.84/0.87/0.85 pol 0.90 | 2.7 min")
ATTN_COLLAPSED = ATTN_HEALTHY.replace("attn H 0.86/0.84/0.87/0.85 pol 0.90", "attn H 0.50/0.23/0.71/0.73 pol 0.32")
ATTN_MISSING = ATTN_HEALTHY.replace(" | attn H 0.86/0.84/0.87/0.85 pol 0.90", "")
PLAY_OK = "[play] life len 42 (n 45) | max rf streak 316 | parked(>=1000) 0.0% | act share 0.24/0.26/0.25/0.25 | act-pi H 1.386"
PLAY_NOOP_RIGHT = "[play] life len 42 (n 45) | max rf streak 316 | parked(>=1000) 0.0% | act share 0.50/0.05/0.40/0.05 | act-pi H 1.00"


def test_parse_reads_the_attention_layers_the_policy_row_and_the_loss():
    p = parse_log_line(ATTN_HEALTHY)
    assert p["attn_layers"] == [0.86, 0.84, 0.87, 0.85]
    assert p["attn_policy"] == 0.90
    assert p["loss_p"] == 7.293 and p["loss_v"] == 0.460


def test_parse_reports_a_missing_attention_field_as_none_not_zero():
    assert parse_log_line(ATTN_MISSING)["attn_layers"] is None


def test_parse_reads_the_action_share_of_a_play_line():
    assert parse_log_line(PLAY_OK)["act_share"] == [0.24, 0.26, 0.25, 0.25]


def test_healthy_run_passes_the_gate():
    ok, reasons = gate_verdict([ATTN_HEALTHY, PLAY_OK])
    assert ok and reasons == []


def test_collapsed_attention_fails_the_gate_even_with_a_finite_loss():
    ok, reasons = gate_verdict([ATTN_COLLAPSED, PLAY_OK])
    assert not ok
    assert any("attention" in r for r in reasons)


def test_a_run_that_never_printed_attention_fails_instead_of_passing():
    ok, reasons = gate_verdict([ATTN_MISSING, PLAY_OK])
    assert not ok and any("no attention" in r for r in reasons)


def test_noop_and_right_collapse_fails_the_gate():
    ok, reasons = gate_verdict([ATTN_HEALTHY, PLAY_NOOP_RIGHT])
    assert not ok and any("NOOP/RIGHT" in r for r in reasons)


def test_non_finite_loss_fails_the_gate():
    ok, reasons = gate_verdict([ATTN_HEALTHY.replace("loss v 0.460", "loss v nan"), PLAY_OK])
    assert not ok and any("loss" in r for r in reasons)


def test_thresholds_are_the_brief_values():
    assert GATE["attn_min"] == 0.8 and GATE["noop_right_max"] == 0.85
