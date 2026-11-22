# Deploy manifests for `scripts/ops/box_deploy.sh`

`scripts/ops/box_deploy.sh PORT HOST scripts/ops/manifests/NAME.txt` ships the repo and these files to a box and checks every
sha256 on both ends; `box_deploy.sh --check NAME.txt` only checks that the local files exist. `${EXP}` is the experiment archive
(`MTHESIS_EXP_ROOT`, default `E:/MThesis_EXP`), `${REPO}` this repository, `${RWS}` the box workspace (`REMOTE_WORKSPACE`, default
`/workspace`); any other `${NAME}` must be set in the environment. Each file replaces a per-experiment deploy script (retired
2026-11-22; verbatim copies in `E:\MThesis_EXP\local_repo_logs_20261122\retired_scripts`).

| Manifest | Was | Box script |
|---|---|---|
| `a81.txt` | `a81_deploy.sh` | `a81_box.sh` |
| `a84.txt` | `a84_deploy.sh code` + `ckpt E15 vf_clean vf_clean_s1 vf_clean_s2 vf_ctrl vf_s2d` | `a84_box.sh`, `a84_train_chain.sh` |
| `arms.txt` | `arms_deploy.sh` | `arms_box.sh` |
| `ladder.txt` | `ladder_deploy.sh PORT HOST SCRATCH` (`SCRATCH` = the folder with `ids220.txt`, `fam_routes.txt`) | `ladder_chain.sh` |
| `ladder_r1.txt` | `W8_SHIP=0 ladder_deploy.sh ...` | `r1_chain.sh` |
