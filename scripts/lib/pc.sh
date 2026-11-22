# PC-side (Git Bash) defaults for the scripts that run on this PC and talk to the boxes. Source it, do not run it:
#   . "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib/pc.sh"
#
# The box-side counterpart is common.sh. Every value is a default the environment can override; the Python side reads the
# same MTHESIS_EXP_ROOT (src/config/paths.py), so one variable moves both to another disk. Tested by tests/test_scripts_pc.py.

: "${MTHESIS_EXP_ROOT:=E:/MThesis_EXP}"     # the experiment archive (stage 1 of the sync pipeline)
# EXP: the archive as a Git Bash path (/e/MThesis_EXP). scp and rsync read "E:/..." as a host named E, so pass EXP to them,
# never MTHESIS_EXP_ROOT.
if command -v cygpath > /dev/null 2>&1; then EXP=$(cygpath -u "$MTHESIS_EXP_ROOT"); else EXP=$MTHESIS_EXP_ROOT; fi
: "${GDRIVE_ROOT:=gdrive:MThesis_backups}"   # rclone remote:folder that stage 2 pushes to (docs/ops/google_drive_sync.md)
export MTHESIS_EXP_ROOT EXP GDRIVE_ROOT

# The vastai CLI: $VAST_CLI, else vastai on PATH, else a conda install under $HOME (pip puts vastai.exe in <env>/Scripts,
# which Git Bash does not have on PATH). Same order as scripts/lib/vast_cli.py.
vast_cli() {
  if [ -n "${VAST_CLI:-}" ]; then printf '%s\n' "$VAST_CLI"; return 0; fi
  command -v vastai 2> /dev/null && return 0
  local c
  for c in "$HOME"/anaconda3/envs/*/Scripts/vastai.exe "$HOME"/anaconda3/Scripts/vastai.exe \
           "$HOME"/miniconda3/envs/*/Scripts/vastai.exe "$HOME"/miniconda3/Scripts/vastai.exe; do
    [ -f "$c" ] && { printf '%s\n' "$c"; return 0; }
  done
  echo "vastai CLI not found: set VAST_CLI or put vastai on PATH" >&2
  return 1
}

# rclone: on PATH, else the WinGet package folder under %LOCALAPPDATA% (winget installs it there without a PATH entry in
# some shells).
rclone_cli() {
  command -v rclone 2> /dev/null && return 0
  local base c
  base=$(cygpath -u "${LOCALAPPDATA:-$HOME/AppData/Local}" 2> /dev/null || echo "$HOME/AppData/Local")
  for c in "$base"/Microsoft/WinGet/Packages/Rclone.Rclone_*/rclone-*/rclone.exe; do
    [ -f "$c" ] && { printf '%s\n' "$c"; return 0; }
  done
  echo "rclone not found: install it (winget install Rclone.Rclone) or put it on PATH" >&2
  return 1
}
