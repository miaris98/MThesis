"""Write the finished results of the A88/A89 agent workflow to disk (2026-11-24).

Reads the workflow journal (one JSON line per started/finished agent), keeps the latest extract result per paper and the
latest verify verdict, and writes:
  E:\\MThesis_EXP\\thesis_references\\paper_sensors_compute_2026-11-24.csv  (A89, one row per paper extracted so far)
  E:\\MThesis_EXP\\thesis_references\\paper_sensors_compute_2026-11-24.json (rows with sources and open issues)
  E:\\MThesis_EXP\\profile_2026-11-24\\a88_harvest.json and a88_script_summary.txt (A88)
Safe to run repeatedly while the workflow is still going; a row is "verified" only if a checker passed it.

Usage: py -I scripts/analysis/salvage_workflow_a88_a89.py JOURNAL
"""
import csv
import json
import sys
from pathlib import Path

journals = [Path(a) for a in sys.argv[1:]]  # several journals are merged, later ones win
ref = Path(r"E:\MThesis_EXP\thesis_references")
prof = Path(r"E:\MThesis_EXP\profile_2026-11-24")
prof.mkdir(parents=True, exist_ok=True)

labels, results = {}, []
for journal in journals:
    for line in journal.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "started":
            labels[e["key"]] = e.get("label", "")
        elif e.get("type") == "result":
            results.append(e)


def value(e):
    for k in ("result", "value", "output"):
        if k in e:
            v = e[k]
            if isinstance(v, str):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    return v
            return v
    return None


rows, verdicts, harvest, script, fetched = {}, {}, {}, [], []
for e in results:  # journal order = finish order, so later loops overwrite earlier ones
    lab = labels.get(e.get("key"), "")
    v = value(e)
    if lab.startswith("extract:") and isinstance(v, dict):
        for r in v.get("rows", []):
            rows[r["key"]] = r
    elif lab.startswith("verify:") and isinstance(v, dict):
        for p in v.get("papers", []):
            verdicts[p["key"]] = p
    elif lab.startswith("harvest:") and isinstance(v, dict):
        harvest.setdefault(v.get("policy", lab), {})["record"] = v
    elif lab.startswith("check:") and isinstance(v, dict):
        harvest.setdefault(lab.split(":")[1], {})["check"] = v
    elif lab.startswith(("script:", "review:")):
        script.append({"label": lab, "result": v})
    elif lab.startswith("fetch:") and isinstance(v, dict):
        fetched += v.get("papers", [])

FIELDS = ["key", "triage", "verified", "cameras", "lidar", "radar", "map_route", "ego_state", "privileged_inputs",
          "input_resolution", "outputs", "params", "flops_per_inference", "train_hardware", "train_time", "gpu_hours",
          "train_data", "batch_epochs", "inference_latency", "inference_hardware", "memory_train", "memory_infer",
          "benchmark_score", "code_weights", "estimates", "notes", "open_issues", "sources"]
out = []
for k, r in sorted(rows.items()):
    vd = verdicts.get(k)
    r = dict(r)
    r["verified"] = "yes" if vd and vd.get("ok") else ("no" if vd else "not checked yet")
    r["open_issues"] = "; ".join(i if isinstance(i, str) else f"{i['field']}: {i['problem']}" for i in (vd or {}).get("issues", []))
    out.append(r)
with open(ref / "paper_sensors_compute_2026-11-24.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
    w.writeheader()
    for r in out:
        r2 = dict(r)
        r2["sources"] = " | ".join(f"{s['field']} @ {s['location']}: \"{s['quote']}\"" for s in r.get("sources", []))
        w.writerow(r2)
(ref / "paper_sensors_compute_2026-11-24.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
(prof / "a88_harvest.json").write_text(json.dumps(harvest, indent=1, ensure_ascii=False), encoding="utf-8")
(prof / "a88_script_summary.txt").write_text(json.dumps(script, indent=1, ensure_ascii=False), encoding="utf-8")
if fetched:
    (ref / "download_log_carla_2026-11-24_agents.json").write_text(json.dumps(fetched, indent=1), encoding="utf-8")

print(f"results {len(results)} | papers extracted {len(out)} (verified {sum(r['verified']=='yes' for r in out)}, "
      f"failed check {sum(r['verified']=='no' for r in out)}) | harvest records {len(harvest)} | script steps {len(script)} | fetched {len(fetched)}")
