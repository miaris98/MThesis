"""TODO A74 (2026-11-02): the training-route builder. PDM-Lite's data-collection routes (carla_garage data/50x38_Town12, 50x36_Town13; one
route per XML, 50 per scenario type) are not the 220 Bench2Drive evaluation routes; A74 trains on them in evaluator-sized batches (one evaluator
process per batch, which also resets the leaderboard's memory growth) with obstacle scenarios at a set share, and holds some out to check
the learning curve on routes the head never trained on."""
import os
import xml.etree.ElementTree as ET

from scripts.training.build_a74_routes import build


def _make_src(root, town="Town12", types=("Accident", "ParkedObstacle", "StaticCutIn", "SignalizedJunctionLeftTurn"), per_type=10):
    d = os.path.join(root, f"50x38_{town}")
    for t in types:
        os.makedirs(os.path.join(d, t))
        for i in range(per_type):
            with open(os.path.join(d, t, f"{i}.xml"), "w") as f:
                f.write(f'<routes><route id="{i}" town="{town}"><weathers><weather cloudiness="5.0" /></weathers><waypoints>'
                        f'<position x="{i}.0" y="0.0" z="0.0" /><position x="{i + 200}.0" y="0.0" z="0.0" /></waypoints>'
                        f'<scenarios><scenario name="{t}_1" type="{t}"><trigger_point x="{i}.0" y="0.0" z="0.0" yaw="0" /></scenario></scenarios>'
                        f'</route></routes>')
    return d


def _routes(path):
    return ET.parse(path).getroot().findall("route")


def test_batches_have_the_obstacle_share_unique_ids_and_no_validation_routes(tmp_path):
    _make_src(str(tmp_path / "data"))
    out = tmp_path / "out"
    man = build(str(tmp_path / "data"), str(out), batches=4, per_batch=6, obstacle_share=0.5, val_per_type=2, seed=0)
    assert len(man["batches"]) == 4 and os.path.exists(out / "val.xml")
    val_src = set(man["val"])
    assert len(val_src) == 8                                                     # 2 per scenario type
    for b in man["batches"]:
        routes = _routes(out / b["file"])
        assert len(routes) == 6 and len({r.get("id") for r in routes}) == 6
        assert sum(r.find("scenarios/scenario").get("type") in ("Accident", "ParkedObstacle") for r in routes) == 3
        assert not val_src & set(b["sources"]), "a held-out route must never be trained on"
    val = _routes(out / "val.xml")
    assert len(val) == 8 and len({r.get("id") for r in val}) == 8


def test_each_batch_stays_in_one_town(tmp_path):
    _make_src(str(tmp_path / "data"), town="Town12")
    _make_src(str(tmp_path / "data"), town="Town13")
    man = build(str(tmp_path / "data"), str(tmp_path / "out"), batches=4, per_batch=6, obstacle_share=0.5, val_per_type=1, seed=0)
    towns = [{r.get("town") for r in _routes(tmp_path / "out" / b["file"])} for b in man["batches"]]
    assert all(len(t) == 1 for t in towns) and {next(iter(t)) for t in towns} == {"Town12", "Town13"}   # no map switch inside a batch


def test_build_is_deterministic_and_keeps_the_route_body(tmp_path):
    _make_src(str(tmp_path / "data"))
    m1 = build(str(tmp_path / "data"), str(tmp_path / "a"), batches=2, per_batch=4, obstacle_share=0.5, val_per_type=1, seed=3)
    m2 = build(str(tmp_path / "data"), str(tmp_path / "b"), batches=2, per_batch=4, obstacle_share=0.5, val_per_type=1, seed=3)
    assert [b["sources"] for b in m1["batches"]] == [b["sources"] for b in m2["batches"]]
    r = _routes(tmp_path / "a" / m1["batches"][0]["file"])[0]
    assert r.get("town") == "Town12" and len(r.findall("waypoints/position")) == 2 and r.find("weathers/weather") is not None
