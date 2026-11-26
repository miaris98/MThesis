"""Write docs/design/paper_sensors_compute_2026-11-24.md from the A89 rows on E: (2026-11-24)."""
import json
from pathlib import Path

ref = Path(r"E:\MThesis_EXP\thesis_references")
rows = json.load(open(ref / "paper_sensors_compute_2026-11-24.json", encoding="utf-8"))
fetched = json.load(open(ref / "download_log_carla_2026-11-24_agents.json", encoding="utf-8"))
done = {r["key"] for r in rows}
pending = [f for f in fetched if f["key"] not in done]


def c(s, n=70):
    s = (s or "").replace("|", "/").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def sensors(r):
    parts = [f"cam {r['cameras']}"]
    for k, lab in (("lidar", "LiDAR"), ("radar", "radar"), ("map_route", "map/route")):
        if r[k] not in ("none", "n/a", "not reported", ""):
            parts.append(f"{lab} {r[k]}")
    return c("; ".join(parts), 90)


out = ["# Sensors, hardware and compute of the cited papers (A89, 2026-11-24)", "",
       "Status: **draft.** Extracted by Haiku agents from the papers' text, with a page and a quote for every value. "
       f"Checked by Sonnet: the last 20 rows ({sum(r.get('verified')=='yes' for r in rows)} passed, "
       f"{sum(r.get('verified')=='no' for r in rows)} flagged with open issues, see the open_issues column in the CSV); "
       f"the first {sum(r.get('verified')=='not checked yet' for r in rows)} rows are not checked yet. Check a value against its quote before it goes into the thesis.", "",
       f"- Full data (every field, sources, quotes): `{ref}\\paper_sensors_compute_2026-11-24.csv` and `.json`.",
       f"- Extracted: {len(rows)} of 118 cited works (not extracted: pomerleau1991efficient and sae2021j3016, paywalled): "
       + ", ".join(f"{sum(r['triage']==t for r in rows)} {n}" for t, n in (("a", "driving policies/planners"), ("b", "other models"), ("c", "data sets/benchmarks/surveys"))) + ".",

       "- Regenerate: `py -I scripts/analysis/salvage_workflow_a88_a89.py <journal>` then `py -I scripts/analysis/a89_summary_md.py`.", "",
       "## Driving policies and planners (triage a)", "",
       "| Paper | Sensors | Params | Train hardware | Train time | Inference | Score | Check |", "|---|---|---|---|---|---|---|---|"]
for r in sorted((r for r in rows if r["triage"] == "a"), key=lambda r: r["key"]):
    inf = r["inference_latency"] if r["inference_latency"] == "not reported" else f"{r['inference_latency']} ({r['inference_hardware']})"
    out.append(f"| {r['key']} | {sensors(r)} | {c(r['params'],40)} | {c(r['train_hardware'],50)} | {c(r['train_time'],40)} | {c(inf,50)} | {c(r['benchmark_score'],60)} | {r.get('verified','')} |")
out += ["", "## Other learned models (triage b)", "",
        "| Paper | Params | FLOPs / inference | Train hardware | Train time | Memory (train) |", "|---|---|---|---|---|---|"]
for r in sorted((r for r in rows if r["triage"] == "b"), key=lambda r: r["key"]):
    out.append(f"| {r['key']} | {c(r['params'],45)} | {c(r['flops_per_inference'],40)} | {c(r['train_hardware'],50)} | {c(r['train_time'],40)} | {c(r['memory_train'],30)} |")
out += ["", "## Data sets, benchmarks, simulators, surveys (triage c)", "", "| Paper | Sensors | Data |", "|---|---|---|"]
for r in sorted((r for r in rows if r["triage"] == "c"), key=lambda r: r["key"]):
    out.append(f"| {r['key']} | {sensors(r)} | {c(r['train_data'],80)} |")
p = Path(r"C:\Users\miari\Desktop\MThesis\docs\design\paper_sensors_compute_2026-11-24.md")
p.write_text("\n".join(out) + "\n", encoding="utf-8")
print(p, len(out), "lines")
