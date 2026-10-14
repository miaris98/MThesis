import json, glob, re, sys, numpy as np

SP, NEWGLOB, NEWNAME = sys.argv[1], sys.argv[2], sys.argv[3]


INFRA = ("Failed - Agent crashed", "Failed - Agent couldn't be set up", "Failed - Simulation crashed")  # driving outcomes (TickRuntime, blocked, timed out, deviation ...) all count


def recs(globs):
    out = {}
    for g in globs:
        for p in sorted(glob.glob(g)):
            try:
                d = json.load(open(p))
            except Exception:
                continue
            for r in d.get("_checkpoint", {}).get("records", []):
                m = re.search(r"RouteScenario_(\d+)_", r["route_id"])
                sc = r.get("scores", {}).get("score_composed")
                if not m or sc is None or r["status"] in INFRA:
                    continue
                out[int(m.group(1))] = float(sc)
    return out


M = "E:\\MThesis_EXP"
new = recs([NEWGLOB.replace("{SP}", SP)])
base = {
    "plain E15 (2026-09-28 official run)": [M + "\\bench2drive_live_20260928\\a9_E15_c*_x.json"],
    "plain E15 repeat (e15r)": [M + "\\live_2026*\\bench2drive_out\\e15r_*_x.json"],
    "E15 speed x0.9 alone": [M + "\\live_2026*\\bench2drive_out\\e15_ss09v*_x.json"],
    "E15 speed x0.8 alone": [M + "\\live_2026*\\bench2drive_out\\e15_ss08v*_x.json"],
    "E15 newsvendor alone (A57)": [M + "\\live_2026*\\bench2drive_out\\e15_a57_nfr*_x.json"],
}
rng = np.random.default_rng(0)
print(f"{NEWNAME}: {len(new)} routes with a driving record, mean {np.mean(list(new.values())):.1f}")
for name, gl in base.items():
    b = recs(gl)
    common = sorted(set(new) & set(b))
    if len(common) < 10:
        print(f"  vs {name}: only {len(common)} common routes (other has {len(b)})")
        continue
    a = np.array([new[r] for r in common])
    c = np.array([b[r] for r in common])
    d = a - c
    bs = [rng.choice(d, len(d)).mean() for _ in range(4000)]
    print(f"  vs {name}: n={len(common)}  new {a.mean():.1f}  other {c.mean():.1f}  diff {d.mean():+.1f}  95% CI [{np.percentile(bs, 2.5):+.1f}, {np.percentile(bs, 97.5):+.1f}]")
