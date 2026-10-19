"""scripts/analysis/thesis_stats.py on synthetic driving scores whose variance components are known."""
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
ts = pytest.importorskip("thesis_stats")


def _seeds(s, n, sg, sw, base=60.0, seed=0):
    rnd = random.Random(seed)
    mu = {r: base + rnd.gauss(0, 20) for r in range(n)}
    return {f"s{k}": {r: mu[r] + g + rnd.gauss(0, sw) for r in range(n)} for k, g in ((k, rnd.gauss(0, sg)) for k in range(s))}, mu


def test_components_recover_seed_and_route_noise():
    seeds, _ = _seeds(s=60, n=300, sg=3.0, sw=12.0)
    c = ts.components(seeds)
    assert c["sw2"] == pytest.approx(144, rel=0.06)
    assert c["sg2_unbiased"] == pytest.approx(9, rel=0.45)          # 60 seeds: the estimate of a variance of 9 is still loose
    assert c["sg2"] == c["sg2_raw"] == pytest.approx(c["sg2_unbiased"] + c["sw2"] / 300)   # default: raw, conservative
    assert ts.components(seeds, unbiased_sg=True)["sg2"] == c["sg2_unbiased"]
    assert c["srun2"] == c["sw2"] and c["sr2"] == 0.0 and c["srun_dof"] == 0   # no repeats: run noise = all of it


def test_run_repeats_split_seed_from_run_noise():
    rnd = random.Random(1)
    seeds, mu = _seeds(s=10, n=200, sg=0.0, sw=10.0)
    repeats = [{r: [mu[r] + rnd.gauss(0, 6) for _ in range(3)] for r in range(200)}]
    c = ts.components(seeds, repeats)
    assert c["srun2"] == pytest.approx(36, rel=0.15) and c["srun_dof"] == 400
    assert c["sr2"] == pytest.approx(100 - 36, rel=0.25)


def test_holm_and_bh_against_hand_computed_values():
    ps = [0.01, 0.04, 0.03, 0.005]
    assert ts.holm(ps) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert ts.bh(ps) == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert ts.holm([0.5]) == [0.5] and ts.bh([0.2, 0.9]) == pytest.approx([0.4, 0.9])


def test_signflip_p():
    assert ts.signflip_p([5.0] * 30, n=4000) < 0.001
    assert ts.signflip_p([1.0, -1.0] * 15, n=4000) > 0.9


def _runs(d):
    return {r: [v] for r, v in d.items()}


def test_one_seed_recipe_interval_carries_the_shared_seed_shift():
    comp = {"sw2": 100.0, "sr2": 64.0, "srun2": 36.0, "sg2": 4.0}
    rnd = random.Random(2)
    x = {r: [70 + rnd.gauss(0, 6)] for r in range(220)}
    y = {r: [60 + rnd.gauss(0, 6)] for r in range(220)}
    o = ts.compare(x, y, comp, boot=2000)
    ck, rf, rp = o["ci_checkpoint_fixed"], o["ci_recipe_fixed"], o["ci_recipe_population"]
    assert ck[0] < o["diff"] < ck[1]
    assert (rf[1] - rf[0]) > (ck[1] - ck[0])                              # seed noise widens it
    # one seed can never average sigma_g away: the recipe interval is at least +-1.96 * sigma_g wide
    assert (rf[1] - rf[0]) / 2 >= 1.96 * 2.0 - 1e-9
    assert (rp[1] - rp[0]) >= (o["ci_route_boot"][1] - o["ci_route_boot"][0])
    assert 0 <= o["p_recipe"] <= 1 and o["p_route"] < 0.01


def test_without_seed_noise_the_recipe_and_checkpoint_intervals_agree():
    comp = {"sw2": 36.0, "sr2": 0.0, "srun2": 36.0, "sg2": 0.0}
    x = {r: [60.0 + (r % 7)] for r in range(50)}
    y = {r: [60.0] for r in range(50)}
    o = ts.compare(x, y, comp, boot=1000)
    assert o["ci_recipe_fixed"] == pytest.approx(o["ci_checkpoint_fixed"])


def test_two_level_bootstrap_for_a_pool_is_wider_than_routes_alone():
    seeds, mu = _seeds(s=4, n=150, sg=4.0, sw=8.0, seed=3)
    ref = {r: mu[r] for r in range(150)}
    pool_runs = {r: [seeds[k][r] for k in seeds] for r in range(150)}
    comp = ts.components(seeds)
    o = ts.compare(pool_runs, _runs(ref), comp, x_pool=seeds, boot=2000)
    d = [ts.mean(pool_runs[r]) - ref[r] for r in range(150)]
    lo, hi = ts.boot_ci(d, n=2000)
    assert (o["ci_recipe_population"][1] - o["ci_recipe_population"][0]) > (hi - lo)
    assert "ci_checkpoint_fixed" not in o                                   # a pool is not one checkpoint


def test_mde_shrinks_with_routes_seeds_and_runs_but_one_seed_has_a_floor():
    comp = {"sr2": 200.0, "srun2": 300.0, "sg2": 2.0}
    m = lambda n, s, k, fixed=True: ts.mde(comp, 50.0, n, s, k, fixed)
    assert m(220, 1, 1) > m(220, 3, 1) > m(220, 5, 1)
    assert m(220, 3, 1) > m(220, 3, 2)
    assert m(19, 1, 1) > m(220, 1, 1)
    assert m(10 ** 9, 1, 1) == pytest.approx(2.8016 * 2 ** 0.5, rel=1e-3)   # infinitely many routes: sqrt(sigma_g^2 / 1)
    assert m(220, 3, 1, fixed=False) > m(220, 3, 1, fixed=True) * 0.9


def test_a_switch_on_one_checkpoint_has_no_seed_term():
    comp = {"sr2": 200.0, "srun2": 150.0, "sg2": 50.0}
    sw = lambda n, k, pop: ts.mde_switch(comp, 100.0, n, k, pop)
    assert sw(220, 1, False) == pytest.approx(2.8016 * (2 * 150 / 220) ** 0.5)
    assert sw(220, 10 ** 6, False) < 0.01                                      # these routes: runs average the noise away
    assert sw(220, 10 ** 6, True) == pytest.approx(2.8016 * (100 / 220) ** 0.5, rel=1e-3)   # routes like these: the interaction floor
    assert sw(220, 1, True) > sw(220, 1, False)


def _write(out: Path, name: str, scores: dict):
    recs = [{"route_id": f"RouteScenario_{r}", "status": "Completed", "infractions": {},
             "scores": {"score_route": 100.0, "score_penalty": v / 100, "score_composed": v}} for r, v in scores.items()]
    (out / name).write_text(json.dumps({"_checkpoint": {"progress": [len(recs), len(recs)], "records": recs}}))


def test_cli_end_to_end(tmp_path):
    out = tmp_path / "live_x"
    out.mkdir()
    seeds, mu = _seeds(s=4, n=60, sg=2.0, sw=10.0, seed=4)
    rids = [2050 + i for i in range(60)]
    for name, k in (("j20_a_x.json", "s0"), ("j1s20_a_x.json", "s1"), ("j2s20_a_x.json", "s2"), ("j3s20_a_x.json", "s3")):
        _write(out, name, {rid: max(0.0, min(100.0, seeds[k][i])) for i, rid in enumerate(rids)})
    _write(out, "a9_WOR_c01_x.json", {rid: max(0.0, min(100.0, mu[i] - 10)) for i, rid in enumerate(rids)})
    _write(out, "a9_E15_c01_x.json", {rid: max(0.0, min(100.0, mu[i] + 1)) for i, rid in enumerate(rids)})
    js = tmp_path / "s.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "analysis" / "thesis_stats.py"), str(out), "--pool", "J=J20,J20s1,J20s2,J20s3",
                        "--pool", "ONE=E15,E15s9", "--compare", "E15-WOR", "--compare", "J-WOR", "--family", "e15=E15-WOR", "--family", "j=J-WOR",
                        "--family", "one=ONE-WOR", "--interaction-switch", "E15-WOR", "--interaction-recipe", "J-E15",
                        "--bootstrap", "2000", "--json-out", str(js)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    data = json.loads(js.read_text())
    assert data["components"]["seeds"] == 4 and data["components"]["routes"] == 60
    assert set(data["compare"]) == {"E15-WOR", "J-WOR"} and data["compare"]["E15-WOR"]["n"] == 60
    assert [row["name"] for row in data["family"]] == ["e15", "j", "one"]          # a pool with one seed present still counts
    assert "only E15 has records" in r.stdout
    assert "n220_s3_k1" in data["mde"]["recipe"]["table"] and "n220_k8" in data["mde"]["switch"]["table"]
    assert "smallest resolvable effect of a switch" in r.stdout
