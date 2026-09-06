# Experiment Storage & MLflow Sync Guide

Where experiment output goes, and how to get vast.ai runs onto the external disk
(`E:\MThesis_EXP`) so every run — local and remote — sits in one MLflow dashboard.

---

## 1. The experiment root

One resolver, [src/config/paths.py](src/config/paths.py), decides where output goes.
Everything else asks it instead of hardcoding a path.

| Machine                          | Root                | mlruns                       |
|----------------------------------|---------------------|------------------------------|
| This PC, external disk mounted   | `E:\MThesis_EXP`    | `E:\MThesis_EXP\mlruns`      |
| vast.ai instance (`/workspace`)  | `/workspace`        | `/workspace/MThesis/mlruns`  |
| Anywhere else                    | `<repo>/experiments`| `<repo>/experiments/mlruns`  |

`MTHESIS_EXP_ROOT` overrides all of it:

```powershell
$env:MTHESIS_EXP_ROOT = "D:\MThesis_EXP"
```

Check what a machine resolved:

```bash
python src/config/paths.py
```

**The vast.ai layout is deliberately unchanged.** `/workspace/runs`,
`/workspace/checkpoints` and `/workspace/MThesis/mlruns` are exactly where
`run_multi_carla_training.sh` and the tmux MLflow server already expect them.

### Layout under the root

```
E:\MThesis_EXP\
  mlruns\        MLflow store - local runs AND imported vast.ai runs, merged
  runs\          TensorBoard events + telemetry CSVs
  checkpoints\   .pth weights + train_state.json
  videos\        evaluation videos
  imports\       raw per-instance snapshots, kept as pulled
```

---

## 2. Training locally → straight onto the disk

Nothing to configure. With the disk mounted, the defaults already point at it:

```bash
python train_wor.py --data_dir dataset/wor_trajectories --epochs 50
# checkpoints, TensorBoard events and MLflow runs all land under E:\MThesis_EXP
```

Override per run if you want:

```bash
python train_wor.py --save_dir "E:/MThesis_EXP/checkpoints/wor_qwen100m"
```

If the disk is **not** mounted the resolver falls back to `<repo>/experiments`
(gitignored) rather than failing — but then it is on C:, so plug the disk in first.

---

## 3. Training on vast.ai → pull it down afterwards

```bash
# Metrics + telemetry only (small, seconds)
python sync_experiments.py --ssh-cmd "ssh -p 12345 root@ssh5.vast.ai"

# Everything: mlruns, telemetry, TensorBoard events, checkpoints, videos
python sync_experiments.py --ssh-cmd "ssh -p 12345 root@ssh5.vast.ai" --include all

# Pick and choose
python sync_experiments.py --ssh-cmd "..." --include mlruns checkpoints
```

Groups: `mlruns`, `telemetry`, `tensorboard`, `checkpoints`, `videos`, or `all`.

What happens:

1. **Pull** — one `tar`-over-`ssh` stream into `E:\MThesis_EXP\imports\<host-port_date>\`.
   A single stream, not thousands of `scp` round-trips: an MLflow store is mostly
   tiny files and `scp -r` is painfully slow on it. Paths that do not exist yet on
   the instance are skipped rather than failing the transfer.
2. **Merge** — the snapshot's `mlruns` is folded into `E:\MThesis_EXP\mlruns`.

Useful flags:

| Flag | Effect |
|------|--------|
| `--no-merge` | Pull the snapshot only, leave the local store alone |
| `--merge-only <dir>` | Merge a snapshot pulled earlier, no network |
| `--overwrite` | Replace runs already present locally (default: skip them) |
| `--list` | Show what the local store holds |
| `--relink` | Repair artifact paths after a drive-letter change |
| `--tag <name>` | Name the snapshot directory yourself |
| `--dest <path>` | Sync into a different root |

Re-running a pull is safe: runs already merged are skipped.

### How the merge avoids collisions

MLflow 2.x numbers experiments `0, 1, 2…`, so the instance's experiment `1` and a
local experiment `1` are different things with the same id. The merge therefore:

- matches experiments **by name**, since MLflow requires names to be unique — remote
  `WoR_Offline_Training` joins the local experiment of that name;
- keeps the source experiment id when it is free locally (MLflow 3 hands out large
  random ids, which never collide), and renumbers only on a real clash;
- rewrites every run's `artifact_uri` to its new home on the disk, so artifacts
  actually open instead of pointing at `/workspace/...`;
- tags each imported run with `source_instance` so you can tell in the UI which
  machine produced it.

---

## 4. Viewing everything

```powershell
powershell -File mlflow_local.ps1
```

Serves `E:\MThesis_EXP\mlruns` on <http://localhost:10100> and opens a browser.

Options: `-Port 10101`, `-Root "D:\MThesis_EXP"`, `-Python <path>`, `-NoBrowser`,
`-Relink`.

**One-time setup** — MLflow is not installed in `carla_rl` yet:

```powershell
& "$env:USERPROFILE\anaconda3\envs\carla_rl\python.exe" -m pip install mlflow
```

For the *live* dashboard on a running instance, [tunnel_mlflow.ps1](tunnel_mlflow.ps1)
still does what it always did (SSH port-forward), and the trainer prints a
Cloudflare URL. `mlflow_local.ps1` is for the permanent archive.

---

## 5. Gotchas

**MLflow 3.16+ refuses file stores.** It puts the filesystem backend in
maintenance mode and raises unless `MLFLOW_ALLOW_FILE_STORE=true` is set. Every
store here is a file store, so the opt-out is set for you in
[ExperimentLogger](src/logging/experiment_logger.py),
[run_multi_carla_training.sh](run_multi_carla_training.sh) and
[mlflow_local.ps1](mlflow_local.ps1). Setting it by hand:

```powershell
$env:MLFLOW_ALLOW_FILE_STORE = "true"
mlflow ui --backend-store-uri file:///E:/MThesis_EXP/mlruns
```

**Drive letters are baked in.** MLflow stores absolute artifact paths. If the disk
comes back as `F:` instead of `E:`:

```bash
python sync_experiments.py --relink --dest "F:/MThesis_EXP"
```

**Back up `mlruns\`, not just checkpoints.** The metrics history of every run lives
there and nowhere else once the instance is destroyed.

---

## 6. Related scripts

| Script | Purpose |
|--------|---------|
| [sync_experiments.py](sync_experiments.py) | Pull + merge vast.ai runs into the archive |
| [mlflow_local.ps1](mlflow_local.ps1) | Serve the local archive |
| [src/config/paths.py](src/config/paths.py) | Resolve the experiment root |
| [download_from_vastai.py](download_from_vastai.py) | Older per-file `scp` downloader; now defaults to the experiment root. `sync_experiments.py` supersedes it for anything MLflow-shaped |
| [tunnel_mlflow.ps1](tunnel_mlflow.ps1) | SSH-forward the *live* dashboard off an instance |
| [start_mlflow_tunnel.sh](start_mlflow_tunnel.sh) | Cloudflare tunnel for the live dashboard (runs on the instance) |
