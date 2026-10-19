"""TODO A60: lane_plan.py must reproduce neither 2026-10-19 incident.

1. A crash continuation has to name the route the evaluator was actually on (it drives in ascending id order).
2. A lane restarted on a changed route list must never relaunch its label (the evaluator would reset the result file);
   only the routes nobody has driven yet run, under a derived label, and a second restart resumes that label.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
import lane_plan  # noqa: E402

DRIVE, INFRA = "Completed", "Failed - Agent couldn't be set up"


def _result(out: Path, label: str, statuses: dict, total: int):
    recs = [{"route_id": f"RouteScenario_{r}", "status": s, "scores": {"score_composed": 50.0}} for r, s in statuses.items()]
    (out / f"{label}_x.json").write_text(json.dumps({"_checkpoint": {"progress": [len(recs), total], "records": recs}}))


def _routes(out: Path, label: str, routes: str):
    (out / f"{label}_x.routes").write_text(routes + "\n")


def test_norm_sorts_and_drops_duplicates():
    assert lane_plan.fmt(lane_plan.parse("3457, 2050,2509,2050,\n24795")) == "2050,2509,3457,24795"


def test_split_names_the_route_the_evaluator_is_on_for_an_unsorted_list():
    # the evaluator runs 2050, 2509, 23658 in that order: with 1 record done, the crash route is 2509, not 2050
    assert lane_plan.split("23658,2050,2509", 1) == ("2509", "23658")
    assert lane_plan.split("23658,2050,2509", 3) == ("", "")
    assert lane_plan.split("2509", 0) == ("2509", "")


def test_new_label_runs_the_sorted_list(tmp_path):
    assert lane_plan.plan("e15_nC1_2000", "3457,2050", str(tmp_path)) == ("e15_nC1_2000", [2050, 3457])


def test_same_set_in_another_order_resumes_the_label(tmp_path):
    _routes(tmp_path, "L", "3457,2050,2509")
    _result(tmp_path, "L", {2050: DRIVE}, 3)
    assert lane_plan.plan("L", "2509,3457,2050", str(tmp_path)) == ("L", [2050, 2509, 3457])


def test_changed_list_goes_to_a_derived_label_with_only_the_undriven_routes(tmp_path):
    _routes(tmp_path, "L", "2050,2509,3457,24795")
    _result(tmp_path, "L", {2050: DRIVE, 2509: INFRA}, 4)          # 2509 only failed in the harness: still to drive
    assert lane_plan.plan("L", "2050,2509,3457", str(tmp_path)) == ("Ld1", [2509, 3457])


def test_second_restart_on_the_same_changed_list_resumes_the_derived_label(tmp_path):
    _routes(tmp_path, "L", "2050,2509,3457,24795")
    _result(tmp_path, "L", {2050: DRIVE}, 4)
    _routes(tmp_path, "Ld1", "2509,3457")
    _result(tmp_path, "Ld1", {2509: DRIVE}, 2)                     # Ld1 is half done: resuming it loses nothing
    assert lane_plan.plan("L", "2050,2509,3457", str(tmp_path)) == ("Ld1", [2509, 3457])


def test_another_change_opens_the_next_derived_label(tmp_path):
    _routes(tmp_path, "L", "2050,2509,3457,24795")
    _result(tmp_path, "L", {2050: DRIVE}, 4)
    _routes(tmp_path, "Ld1", "2509,3457")
    _result(tmp_path, "Ld1", {2509: DRIVE}, 2)
    assert lane_plan.plan("L", "2050,2509,3457,25318", str(tmp_path)) == ("Ld2", [3457, 25318])


def test_nothing_left_prints_a_dash(tmp_path, capsys):
    _routes(tmp_path, "L", "2050,2509,3457")
    _result(tmp_path, "L", {2050: DRIVE, 2509: DRIVE}, 3)
    assert lane_plan.main(["plan", "L", "2050,2509", str(tmp_path)]) == 0
    assert capsys.readouterr().out.split() == ["Ld1", "-"]


def test_legacy_label_without_a_route_file(tmp_path):
    _result(tmp_path, "L", {2050: DRIVE}, 3)
    assert lane_plan.plan("L", "2050,2509,3457", str(tmp_path))[0] == "L"     # same length, records inside the list
    assert lane_plan.plan("L", "2509,3457", str(tmp_path)) == ("Ld1", [2509, 3457])            # another length
    assert lane_plan.plan("L", "2509,3457,24795", str(tmp_path)) == ("Ld1", [2509, 3457, 24795])  # same length, other routes


def test_unreadable_result_counts_as_nothing_driven(tmp_path):
    _routes(tmp_path, "L", "2050,2509")
    (tmp_path / "L_x.json").write_text("{not json")
    assert lane_plan.plan("L", "2050", str(tmp_path)) == ("Ld1", [2050])


def test_cli_is_callable_the_way_queue_A9c_calls_it(tmp_path):
    py = [sys.executable, str(ROOT / "scripts" / "eval" / "lane_plan.py")]
    out = subprocess.run(py + ["plan", "L", "3457,2050", str(tmp_path)], capture_output=True, text=True, check=True).stdout
    assert out.split() == ["L", "2050,3457"]
    out = subprocess.run(py + ["split", "3457,2050", "0"], capture_output=True, text=True, check=True).stdout
    assert out.split() == ["2050", "3457"]
    assert subprocess.run(py + ["bogus"], capture_output=True, text=True).returncode == 2


@pytest.mark.parametrize("label", ["a9_E15_c03", "e15_a57_nC1_2400", "j20_x"])
def test_derived_labels_keep_the_arm_prefix_for_the_merge(tmp_path, label):
    sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
    import a9_merge
    _routes(tmp_path, label, "2050,2509")
    derived, _ = lane_plan.plan(label, "2509", str(tmp_path))
    assert derived == f"{label}d1"
    assert a9_merge.arm_of(f"{derived}_x.json") == a9_merge.arm_of(f"{label}_x.json")
