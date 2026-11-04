"""Training routes that sit on a Bench2Drive evaluation route must be found and excluded (user, 2026-11-04)."""
import textwrap

from scripts.analysis.route_overlap_audit import (flag_trajectory, flag_training_route, load_eval_routes, load_training_route,
                                                  point_to_polyline)

EVAL_XML = textwrap.dedent("""\
    <routes><route id="1" town="Town13"><waypoints>
      <position x="0.0" y="0.0" z="0"/><position x="100.0" y="0.0" z="0"/></waypoints>
      <scenarios><scenario name="Accident_1" type="Accident"><trigger_point x="50.0" y="0.0" z="0" yaw="0"/></scenario></scenarios>
    </route>
    <route id="2" town="Town12"><waypoints>
      <position x="0.0" y="0.0" z="0"/><position x="0.0" y="100.0" z="0"/></waypoints>
      <scenarios><scenario name="HardBreakRoute_1" type="HardBreakRoute"><trigger_point x="0.0" y="10.0" z="0" yaw="90"/></scenario></scenarios>
    </route></routes>""")


def _train_xml(town, typ, tx, ty, start=(500.0, 500.0), end=(600.0, 500.0)):
    return (f'<routes><route id="11" town="{town}"><waypoints><position x="{start[0]}" y="{start[1]}" z="0"/>'
            f'<position x="{end[0]}" y="{end[1]}" z="0"/></waypoints><scenarios><scenario name="{typ}_1" type="{typ}">'
            f'<trigger_point x="{tx}" y="{ty}" z="0" yaw="0"/></scenario></scenarios></route></routes>')


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_point_to_polyline_distance():
    line = [(0.0, 0.0), (100.0, 0.0)]
    assert point_to_polyline((50.0, 3.0), line) == 3.0
    assert point_to_polyline((-4.0, 0.0), line) == 4.0
    assert point_to_polyline((103.0, 4.0), line) == 5.0


def test_same_scenario_at_the_eval_trigger_is_flagged(tmp_path):
    ev = load_eval_routes(_write(tmp_path, "b2d.xml", EVAL_XML))
    tr = load_training_route(_write(tmp_path, "t.xml", _train_xml("Town13", "Accident", 55.0, 2.0)))
    f = flag_training_route(tr, ev, trigger_radius=30.0, path_radius=10.0)
    assert f["same_type_near_eval_trigger"] and f["near_eval_path"] and f["leak"]
    assert f["eval_ids"] == ["1"]


def test_other_town_is_never_flagged(tmp_path):
    ev = load_eval_routes(_write(tmp_path, "b2d.xml", EVAL_XML))
    tr = load_training_route(_write(tmp_path, "t.xml", _train_xml("Town12", "Accident", 55.0, 2.0)))
    f = flag_training_route(tr, ev, trigger_radius=30.0, path_radius=10.0)
    assert not f["leak"] and f["eval_ids"] == []


def test_far_route_is_kept(tmp_path):
    ev = load_eval_routes(_write(tmp_path, "b2d.xml", EVAL_XML))
    tr = load_training_route(_write(tmp_path, "t.xml", _train_xml("Town13", "Accident", 400.0, 400.0)))
    assert not flag_training_route(tr, ev, trigger_radius=30.0, path_radius=10.0)["leak"]


def test_driven_trajectory_along_an_eval_route_is_flagged(tmp_path):
    ev = load_eval_routes(_write(tmp_path, "b2d.xml", EVAL_XML))
    along = [(float(x), 1.0) for x in range(0, 61, 2)]  # 60 m driven within 1 m of eval route 1
    f = flag_trajectory("Town13", along, ev, path_radius=10.0, min_overlap_m=30.0)
    assert f["leak"] and f["eval_ids"] == ["1"] and f["overlap_m"]["1"] >= 58.0


def test_crossing_an_eval_route_is_not_a_leak(tmp_path):
    # driving across an evaluation road at a junction shares a few metres, not the route
    ev = load_eval_routes(_write(tmp_path, "b2d.xml", EVAL_XML))
    across = [(50.0, float(y)) for y in range(-40, 41, 2)]
    f = flag_trajectory("Town13", across, ev, path_radius=10.0, min_overlap_m=30.0)
    assert not f["leak"]
    assert not flag_trajectory("Town12", [(float(x), 1.0) for x in range(0, 61, 2)], ev, 10.0, 30.0)["leak"]


def test_data_dir_audit_reads_pos_global_and_town_from_the_folder(tmp_path):
    import gzip, json
    from scripts.analysis.route_overlap_audit import audit_data_dir
    ev = load_eval_routes(_write(tmp_path, "b2d.xml", EVAL_XML))
    for name, y in (("Town13_Rep0_1044_0_route0", 1.0), ("Town13_Rep0_9_0_route0", 300.0), ("Town01_Rep0_5_route0", 1.0)):
        m = tmp_path / "data" / "Accident" / name / "measurements"
        m.mkdir(parents=True)
        for i, x in enumerate(range(0, 61, 2)):
            with gzip.open(m / f"{i:04d}.json.gz", "wt") as fh:
                json.dump({"pos_global": [float(x), y]}, fh)
    rows = {r["dir"].replace("\\", "/").rsplit("/", 1)[-1]: r for r in audit_data_dir(str(tmp_path / "data"), ev, 10.0, 30.0)}
    assert set(rows) == {"Town13_Rep0_1044_0_route0", "Town13_Rep0_9_0_route0"}  # Town01 is not an evaluation town here
    assert rows["Town13_Rep0_1044_0_route0"]["leak"] and not rows["Town13_Rep0_9_0_route0"]["leak"]


def test_scenario_on_the_eval_path_is_flagged_even_with_another_type(tmp_path):
    # a training scenario placed on the evaluation road shows the network that stretch of road with an event on it
    ev = load_eval_routes(_write(tmp_path, "b2d.xml", EVAL_XML))
    tr = load_training_route(_write(tmp_path, "t.xml", _train_xml("Town13", "ParkedObstacle", 80.0, 4.0)))
    f = flag_training_route(tr, ev, trigger_radius=30.0, path_radius=10.0)
    assert f["near_eval_path"] and not f["same_type_near_eval_trigger"] and f["leak"]
