#!/usr/bin/env python3
"""Build and maintain the Graphify Knowledge Graph & Wiki for the External MThesis Experiment Archive (E:\\MThesis_EXP).

Usage:
    python scripts/sync/build_external_graph.py
    python scripts/sync/build_external_graph.py --wiki-only
"""
import os
import sys
import json
import argparse
from pathlib import Path

from graphify.detect import detect
from graphify.extract import extract
from graphify.build import build_from_json
from graphify.cluster import cluster, score_all
from graphify.analyze import god_nodes, surprising_connections, suggest_questions
from graphify.report import generate
from graphify.export import to_json

def update_catalog(exp_root: Path):
    """Generate or update EXPERIMENTS_CATALOG.md before graph extraction."""
    checkpoints = []
    ckpt_dir = exp_root / 'checkpoints'
    if ckpt_dir.exists():
        for item in sorted(os.listdir(ckpt_dir)):
            p = ckpt_dir / item
            if p.is_file():
                size_mb = p.stat().st_size / (1024 * 1024)
                checkpoints.append(f'- **File**: `{item}` ({size_mb:.1f} MB)')
            elif p.is_dir():
                files = os.listdir(p)
                checkpoints.append(f'- **Directory**: `{item}/` ({len(files)} items)')

    b2d_runs = []
    b2d_dir = exp_root / 'bench2drive_eval'
    if b2d_dir.exists():
        for item in sorted(os.listdir(b2d_dir)):
            p = b2d_dir / item
            if p.is_dir():
                b2d_runs.append(f'- `{item}/`')

    mlruns_dir = exp_root / 'mlruns'
    exp_dirs = []
    if mlruns_dir.exists():
        for item in sorted(os.listdir(mlruns_dir)):
            p = mlruns_dir / item
            if p.is_dir() and item != '.trash':
                exp_dirs.append(f'- Experiment ID `{item}`')

    catalog_md = f'''# MThesis External Experiments Catalog & Knowledge Base

This document catalogs the historical experiments, model checkpoints, evaluations, and logs stored at `E:\\MThesis_EXP`.

## Checkpoints Repository (`E:\\MThesis_EXP\\checkpoints`)
{chr(10).join(checkpoints) if checkpoints else 'None found'}

## Bench2Drive Evaluations (`E:\\MThesis_EXP\\bench2drive_eval`)
{chr(10).join(b2d_runs) if b2d_runs else 'None found'}

### Key Evaluation Files:
- `bench2drive_eval/run_20260916/bench2drive_out/b2d20_cnn.json`: ResNet-34 CNN Baseline (20 routes).
- `bench2drive_eval/run_20260916/bench2drive_out/b2d20_qwen.json`: Qwen-30M Decision Transformer Policy (20 routes).
- `bench2drive_eval/run_20260916/bench2drive_out/b2d20_tfpp_v2.json`: TransFuser++ Teacher Reference (20 routes).
- `bench2drive_eval/run_20260916/route_representativeness_report.txt`: Statistical representativeness audit across Town01-Town05.
- `bench2drive_eval/run_20260916/crash_rate.py`: Post-hoc collision, infraction, and route completion analyzer.
- `bench2drive_eval/run_20260916/classify_failures.py`: Failure mode taxonomy and route-level categorization.

## MLflow Runs & Experiments (`E:\\MThesis_EXP\\mlruns`)
{chr(10).join(exp_dirs) if exp_dirs else 'None found'}

## Atari Experiments (`E:\\MThesis_EXP\\atari_gtrxl_optuna_20260917`)
- Optuna Bayesian optimization study for `ImpalaGTrXLAgent` on `BreakoutNoFrameskip-v4`.
- SQLite database: `atari_gtrxl_optuna.db`.
- Runner: `optimize_gtrxl_optuna.py`.
'''
    (exp_root / 'EXPERIMENTS_CATALOG.md').write_text(catalog_md, encoding='utf-8')
    print('Updated EXPERIMENTS_CATALOG.md')

def main():
    parser = argparse.ArgumentParser(description="Build external graphify knowledge graph on E:\\MThesis_EXP")
    parser.add_argument("--root", default=r"E:\MThesis_EXP", help="External experiment root")
    parser.add_argument("--wiki-only", action="store_true", help="Only re-export wiki from existing graph.json")
    args = parser.parse_args()

    exp_root = Path(args.root)
    out_dir = exp_root / "graphify-out"
    out_dir.mkdir(exist_ok=True)

    if not args.wiki_only:
        update_catalog(exp_root)

        print("Step 1: Detecting files in", exp_root)
        detect_res = detect(exp_root)
        (out_dir / ".graphify_detect.json").write_text(json.dumps(detect_res, ensure_ascii=False), encoding="utf-8")
        print(f"Detected {detect_res.get('total_files', 0)} files")

        files_to_extract = []
        for cat in ["code", "document"]:
            for f in detect_res.get("files", {}).get(cat, []):
                p = Path(f)
                if p.is_file():
                    files_to_extract.append(p)

        print(f"Step 2: Extracting {len(files_to_extract)} files with AST/parsers...")
        ast_result = extract(files_to_extract, cache_root=exp_root)
        nodes = ast_result.get("nodes", [])
        edges = ast_result.get("edges", [])
        print(f"AST extracted: {len(nodes)} nodes, {len(edges)} edges")

        (out_dir / ".graphify_extract.json").write_text(json.dumps(ast_result, indent=2, ensure_ascii=False), encoding="utf-8")

        print("Step 3: Building graph...")
        G = build_from_json(ast_result, root=str(exp_root), directed=False)
        print(f"Built Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        if G.number_of_nodes() == 0:
            print("Error: graph is empty.")
            return 1

        print("Step 4: Clustering & Analyzing...")
        communities = cluster(G)
        cohesion = score_all(G, communities)
        gods = god_nodes(G)
        surprises = surprising_connections(G, communities)

        labels = {}
        for cid, node_ids in communities.items():
            node_labels = [G.nodes[nid].get("label", "") for nid in node_ids[:10]]
            combined = " ".join(node_labels).lower()
            if "bench2drive" in combined or "crash" in combined or "route" in combined:
                labels[cid] = "Bench2Drive Evaluation & Metrics"
            elif "mlrun" in combined or "experiment" in combined:
                labels[cid] = "MLflow Runs & Tracking"
            elif "optuna" in combined or "atari" in combined or "gtrxl" in combined:
                labels[cid] = "Atari GTrXL Optuna Tuning"
            elif "checkpoint" in combined or "pth" in combined or "model" in combined:
                labels[cid] = "Checkpoints & Model Weights"
            elif "wor" in combined or "rail" in combined:
                labels[cid] = "World-on-Rails Training"
            else:
                labels[cid] = f"Experiments Community {cid}"

        questions = suggest_questions(G, communities, labels)

        print("Step 5: Exporting graph.json and GRAPH_REPORT.md...")
        to_json(G, communities, str(out_dir / "graph.json"), community_labels=labels)
        (out_dir / ".graphify_labels.json").write_text(json.dumps({str(k): v for k, v in labels.items()}, ensure_ascii=False), encoding="utf-8")

        report = generate(G, communities, cohesion, labels, gods, surprises, detect_res, {"input": 0, "output": 0}, str(exp_root), suggested_questions=questions)
        (out_dir / "GRAPH_REPORT.md").write_text(report, encoding="utf-8")

        (out_dir / ".graphify_python").write_text(sys.executable, encoding="utf-8")
        (out_dir / ".graphify_root").write_text(str(exp_root), encoding="utf-8")

    # Step 6: Export wiki and HTML
    print("Step 6: Exporting wiki & HTML visualization...")
    import subprocess
    py_exe = sys.executable
    subprocess.run([py_exe, "-m", "graphify", "export", "wiki", "--graph", str(out_dir / "graph.json"), "--labels", str(out_dir / ".graphify_labels.json")], cwd=str(exp_root), check=True)
    subprocess.run([py_exe, "-m", "graphify", "export", "html", "--graph", str(out_dir / "graph.json"), "--labels", str(out_dir / ".graphify_labels.json")], cwd=str(exp_root), check=True)

    print("Success! External Knowledge Graph & Wiki ready at:", out_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main())
