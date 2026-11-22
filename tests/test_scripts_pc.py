"""PC-side helpers: scripts/lib/pc.sh, scripts/lib/vast_cli.py, box_deploy.sh manifest expansion, agent_smoke.check."""
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
needs_bash = pytest.mark.skipif(BASH is None, reason="bash not available")
sys.path.insert(0, str(REPO))


def sh(script, **env):
    full = {**os.environ, **env}
    r = subprocess.run([BASH, "-c", f". '{(REPO / 'scripts/lib/pc.sh').as_posix()}'; {script}"],
                       capture_output=True, text=True, env=full, timeout=60)
    return r.returncode, r.stdout.strip()


@needs_bash
def test_pc_sh_archive_root_defaults_and_overrides():
    env = {k: v for k, v in os.environ.items() if k not in ("MTHESIS_EXP_ROOT", "GDRIVE_ROOT")}
    r = subprocess.run([BASH, "-c", f". '{(REPO / 'scripts/lib/pc.sh').as_posix()}'; echo \"$MTHESIS_EXP_ROOT|$GDRIVE_ROOT\""],
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.stdout.strip() == "E:/MThesis_EXP|gdrive:MThesis_backups"
    rc, out = sh('echo "$GDRIVE_ROOT"', GDRIVE_ROOT="gdrive:other")
    assert out == "gdrive:other"
    if shutil.which("cygpath"):   # Git Bash: EXP is the scp-safe form of the same folder
        rc, out = sh('echo "$EXP"', MTHESIS_EXP_ROOT="E:/MThesis_EXP")
        assert out.lower() == "/e/mthesis_exp"


@needs_bash
def test_pc_sh_vast_cli_order(tmp_path):
    rc, out = sh("vast_cli", VAST_CLI="/opt/vastai")
    assert (rc, out) == (0, "/opt/vastai")
    fake_home = tmp_path / "home"
    exe = fake_home / "anaconda3" / "envs" / "tools" / "Scripts" / "vastai.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    env = {k: v for k, v in os.environ.items() if k != "VAST_CLI"}
    env["HOME"] = fake_home.as_posix()
    env["PATH"] = "/usr/bin:/bin"   # no vastai on PATH
    r = subprocess.run([BASH, "-c", f". '{(REPO / 'scripts/lib/pc.sh').as_posix()}'; vast_cli"], capture_output=True, text=True,
                       env=env, timeout=60)
    assert r.stdout.strip().endswith("anaconda3/envs/tools/Scripts/vastai.exe")


def test_vast_cli_py_order(tmp_path, monkeypatch):
    from scripts.lib.vast_cli import vast_cli
    monkeypatch.setenv("VAST_CLI", "/opt/vastai")
    assert vast_cli() == "/opt/vastai"
    monkeypatch.delenv("VAST_CLI")
    monkeypatch.delenv("VAST", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    exe = tmp_path / "anaconda3" / "envs" / "tools" / "Scripts" / "vastai.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    assert Path(vast_cli(home=tmp_path)) == exe
    with pytest.raises(FileNotFoundError):
        vast_cli(home=tmp_path / "empty")


@needs_bash
def test_box_deploy_check_expands_and_refuses_unset_variables(tmp_path):
    data = tmp_path / "data.bin"
    data.write_text("x")
    man = tmp_path / "m.txt"
    man.write_text("# comment\n${SCRATCH}/data.bin|${RWS}/x/data.bin\nREADY ${RWS}/x\n", newline="\n")
    deploy = (REPO / "scripts/ops/box_deploy.sh").as_posix()
    env = {**os.environ, "SCRATCH": tmp_path.as_posix()}
    r = subprocess.run([BASH, deploy, "--check", man.as_posix()], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0 and "MANIFEST_OK" in r.stdout and "1 local files present" in r.stdout, r.stdout + r.stderr
    env.pop("SCRATCH")
    r = subprocess.run([BASH, deploy, "--check", man.as_posix()], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 1 and "SCRATCH" in r.stderr
    (tmp_path / "data.bin").unlink()
    r = subprocess.run([BASH, deploy, "--check", man.as_posix()], capture_output=True, text=True,
                       env={**env, "SCRATCH": tmp_path.as_posix()}, timeout=60)
    assert r.returncode == 1 and "MISSING on the PC" in r.stdout


def test_agent_smoke_checks():
    torch = pytest.importorskip("torch")
    from scripts.eval.agent_smoke import check
    net = types.SimpleNamespace(speed_moe=types.SimpleNamespace(w2=torch.zeros(3), num_experts=4),
                                trunk=types.SimpleNamespace(moe=None))
    assert "zero init" in check(net, ["speed_moe.w2"])
    net.speed_moe.w2 = torch.ones(3)
    assert check(net, ["speed_moe.w2"], ["speed_moe.num_experts=4"]) is None
    assert "missing" in check(net, ["trunk.moe.w2"])
    assert "expected 8" in check(net, [], ["speed_moe.num_experts=8"])
