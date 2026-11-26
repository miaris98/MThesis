"""Bring the thesis reference library (E:\\MThesis_EXP\\thesis_references) up to date (2026-11-24).

1. Extracts text for every PDF that has none (text\\<stem>.txt), with pypdf.
2. Writes library_index_2026-11-24.csv: one row per PDF with sha256, pages, text status, the cited key (if any),
   the source URL from the download logs, and the log that recorded it. Existing manifests and logs are not changed.
3. Writes library_missing_2026-11-24.md: cited works that are still not in the library, with the reason.

Usage: py -I scripts/ops/update_reference_library.py [--worklist PATH]
Run with -I (the library holds downloaded PDFs; never import from that folder).
"""
import argparse
import csv
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

from pypdf import PdfReader

LIB = Path(r"E:\MThesis_EXP\thesis_references")
TXT = LIB / "text"
TODAY = date(2026, 11, 24).isoformat()


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_text(pdf: Path, out: Path) -> int:
    reader = PdfReader(str(pdf))
    parts = []
    for i, page in enumerate(reader.pages, 1):
        parts.append(f"\n\n=== page {i} ===\n" + (page.extract_text() or ""))
    text = "".join(parts)
    out.write_text(text, encoding="utf-8")
    return len(text)


def load_sources():
    """file name -> (url, log name) from the download logs; key -> file from the worklist."""
    src = {}
    for name in ["download_log.csv", "download_log_carla_2026-11-16.csv", "download_log_carla_2026-11-22.csv"]:
        p = LIB / name
        if p.exists():
            for row in csv.DictReader(open(p, encoding="utf-8")):
                if row.get("file"):
                    src.setdefault(row["file"], (row.get("url", ""), name))
    agent = LIB / "download_log_carla_2026-11-24_agents.json"
    if agent.exists():
        for x in json.load(open(agent, encoding="utf-8")):
            pdf = x.get("pdf") or ""
            if pdf:
                src.setdefault(Path(pdf).name, (x.get("url", ""), agent.name))
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worklist", default=None)
    args = ap.parse_args()

    pdfs = sorted(LIB.glob("*.pdf"))
    extracted = []
    for pdf in pdfs:
        out = TXT / (pdf.stem + ".txt")
        if not out.exists():
            n = extract_text(pdf, out)
            extracted.append((pdf.name, n))

    key_of = {}
    if args.worklist:
        for o in json.load(open(args.worklist, encoding="utf-8")):
            if o.get("pdf"):
                key_of[Path(o["pdf"]).name] = o["key"]
    agent_log = LIB / "download_log_carla_2026-11-24_agents.json"
    agent_entries = json.load(open(agent_log, encoding="utf-8")) if agent_log.exists() else []
    for x in agent_entries:
        if x.get("pdf"):
            key_of.setdefault(Path(x["pdf"]).name, x["key"])
    src = load_sources()

    rows = []
    for pdf in pdfs:
        t = TXT / (pdf.stem + ".txt")
        url, log = src.get(pdf.name, ("", "none (not in any log)"))
        pages = len(PdfReader(str(pdf)).pages)
        chars = len(t.read_text(encoding="utf-8")) if t.exists() else 0
        rows.append({
            "file": pdf.name, "key": key_of.get(pdf.name, ""), "pages": pages, "bytes": pdf.stat().st_size,
            "sha256": sha256(pdf), "text_file": t.name if t.exists() else "", "text_chars": chars,
            "text_ok": "yes" if chars > 2000 else "NO (scanned or empty?)", "url": url, "recorded_in": log,
        })
    idx = LIB / f"library_index_{TODAY}.csv"
    with open(idx, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # cited works with no PDF in the library, and why
    have_keys = {r["key"] for r in rows if r["key"]}
    reasons = {
        "pomerleau1991efficient": "paywalled (Neural Computation, MIT Press returns 403; CMU page not reachable 2026-11-24)",
        "sae2021j3016": "paywalled SAE standard (needs SAE account)",
        "carlaleaderboard2025bench2drivespace": "web page, not a PDF: text snapshot only (bench2drive_leaderboard\\)",
    }
    cited = sorted(set(key_of.values()) | set(reasons))
    missing = [k for k in cited if k not in have_keys]
    md = [f"# Reference library: cited works without a PDF ({TODAY})", "",
          f"Library: `{LIB}`. Index of every PDF: `library_index_{TODAY}.csv`.", ""]
    md += [f"- `{k}`: {reasons.get(k, 'no PDF found')}" for k in missing] or ["- none"]
    (LIB / f"library_missing_{TODAY}.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"PDFs {len(pdfs)} | text extracted now {len(extracted)} | index {idx}")
    print("cited without a PDF:", missing)
    for name, n in extracted:
        print(f"  text: {name} ({n:,} chars)")
    bad = [r["file"] for r in rows if r["text_ok"] != "yes"]
    print("text too short or missing:", bad)
    return 0


if __name__ == "__main__":
    sys.exit(main())
