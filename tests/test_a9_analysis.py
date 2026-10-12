"""A9 analysis helpers (scripts/analysis/a9_merge.py, a9_abilities.py): result-file labels and the ability success rule."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
a9_merge = pytest.importorskip("a9_merge")
a9_abilities = pytest.importorskip("a9_abilities")


@pytest.mark.parametrize("name,arm", [
    ("a9_E15_c01_x.json", "E15"), ("a9_WOR_E2000t_x.json", "WOR"),
    ("j18_obst_4000_x.json", "J18"), ("j20_b2d20_2800_x.json", "J20"), ("j18H_obst_2000_x.json", "J18H"),
    ("j1s20_obst_2000_x.json", "J20s1"), ("j3s18_b2d20_9600_x.json", "J18s3"),
    ("k20_obst_2000_x.json", "K20"), ("a28m18_b2d20_2000_x.json", "J18med"), ("jh18_obst_x.json", None),
])
def test_arm_labels(name, arm):
    assert a9_merge.arm_of(name) == arm


def _rec(status, **infractions):
    base = {"collisions_vehicle": [], "red_light": [], "stop_infraction": [], "min_speed_infractions": []}
    base.update(infractions)
    return {"status": status, "infractions": base, "scores": {"score_route": 100.0}}


def test_success_rule_matches_official_tool():
    assert a9_abilities.success(_rec("Completed"))
    assert a9_abilities.success(_rec("Perfect"))
    assert a9_abilities.success(_rec("Completed", min_speed_infractions=["slow"]))  # ignored by the official tool
    assert not a9_abilities.success(_rec("Completed", collisions_vehicle=["hit"]))
    assert not a9_abilities.success(_rec("Failed - Agent got blocked"))


def test_official_ability_table_parses():
    ab = a9_abilities.official_abilities()
    assert set(ab) == {"Overtaking", "Merging", "Emergency_Brake", "Give_Way", "Traffic_Signs"}
    assert "Accident" in ab["Overtaking"] and "InvadingTurn" in ab["Give_Way"]
