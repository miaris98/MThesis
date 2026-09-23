"""remote_config.py — Shared remote-box configuration loader.

All scripts in scripts/setup/ and scratch/ import this module instead of
hardcoding SSH host/port/paths. Box-specific values live in a single
`.remote` file at the repository root (git-ignored).

Usage:
    from remote_config import load_config
    cfg = load_config()
    subprocess.run(cfg.ssh("nvidia-smi"), ...)
    subprocess.run(cfg.scp("local/file.py", "remote/path/file.py"), ...)

Precedence (highest → lowest):
    1. Environment variables (REMOTE_HOST, REMOTE_PORT, …)
    2. .remote config file (repo root or any ancestor directory)
    3. Built-in defaults
"""

from __future__ import annotations

import configparser
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Config file discovery
# ---------------------------------------------------------------------------

def _find_remote_file() -> Optional[Path]:
    """Walk up from the calling script's directory to find .remote."""
    # Start from the directory of the script that called us (not this file).
    start = Path(sys.argv[0]).resolve().parent if sys.argv[0] else Path.cwd()
    candidates = [start, *start.parents]
    for d in candidates:
        p = d / ".remote"
        if p.is_file():
            return p
    return None


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class RemoteConfig:
    host: str = "UNSET"
    port: str = "22"
    user: str = "root"
    workspace: str = "/workspace/MThesis"
    venv: str = "/venv/main/bin"

    # CPU core ranges for taskset (Gate 2 parallel variants)
    cpu_cores_a: str = "16-20"
    cpu_cores_b: str = "21-25"
    cpu_cores_c: str = "26-31"

    # Additional venvs (e.g. CARLA uses a separate Python 3.8 venv)
    venv_carla: str = "/venv/carla_py38/bin"

    def __post_init__(self):
        if self.host == "UNSET":
            raise ValueError(
                "Remote host is not configured.\n"
                "Create a .remote file at the repository root (copy .remote.example)\n"
                "or set the REMOTE_HOST environment variable."
            )

    # ------------------------------------------------------------------
    # SSH / SCP helpers
    # ------------------------------------------------------------------

    def _ssh_base(self) -> List[str]:
        return [
            "ssh", "-p", self.port,
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=15",
            "-o", "BatchMode=yes",
            f"{self.user}@{self.host}",
        ]

    def ssh(self, remote_cmd: str) -> List[str]:
        """Return a subprocess-ready SSH command list."""
        return self._ssh_base() + [remote_cmd]

    def ssh_T(self, remote_cmd: str) -> List[str]:
        """SSH with pseudo-TTY disabled (-T), for piped stdin usage."""
        return [
            "ssh", "-T", "-p", self.port,
            "-o", "StrictHostKeyChecking=no",
            f"{self.user}@{self.host}",
            remote_cmd,
        ]

    def scp(self, local_path: str, remote_path: str, recursive: bool = False) -> List[str]:
        """Return a subprocess-ready SCP upload command list."""
        cmd = ["scp", "-P", self.port, "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes"]
        if recursive:
            cmd.append("-r")
        cmd.append(local_path)
        cmd.append(f"{self.user}@{self.host}:{remote_path}")
        return cmd

    def scp_download(self, remote_path: str, local_path: str, recursive: bool = False) -> List[str]:
        """Return a subprocess-ready SCP download command list."""
        cmd = ["scp", "-P", self.port, "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes"]
        if recursive:
            cmd.append("-r")
        cmd.append(f"{self.user}@{self.host}:{remote_path}")
        cmd.append(local_path)
        return cmd

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def remote_path(self, *parts: str) -> str:
        """Join paths under the remote workspace root."""
        return "/".join([self.workspace.rstrip("/")] + [p.lstrip("/") for p in parts])

    def python(self) -> str:
        """Absolute path to the remote Python interpreter."""
        return f"{self.venv.rstrip('/')}/python"

    def python_carla(self) -> str:
        """Absolute path to the CARLA-specific remote Python interpreter."""
        return f"{self.venv_carla.rstrip('/')}/python"

    def run_ssh(self, cmd_str: str, check: bool = False, timeout: Optional[int] = None):
        """Run a remote command and return CompletedProcess. Prints the command."""
        print(f"--> [SSH {self.host}:{self.port}] {cmd_str[:120]}")
        return subprocess.run(
            self.ssh(cmd_str),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=check,
        )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config() -> RemoteConfig:
    """Load remote config from env vars → .remote file → defaults."""
    raw: dict = {}

    # 1. Try .remote file
    rc_file = _find_remote_file()
    if rc_file:
        parser = configparser.ConfigParser()
        parser.read(rc_file, encoding="utf-8")
        section = "remote" if parser.has_section("remote") else parser.sections()[0] if parser.sections() else None
        if section:
            raw = dict(parser[section])

    # 2. Env var overrides (always win over file)
    env_map = {
        "host": "REMOTE_HOST",
        "port": "REMOTE_PORT",
        "user": "REMOTE_USER",
        "workspace": "REMOTE_WORKSPACE",
        "venv": "REMOTE_VENV",
        "cpu_cores_a": "REMOTE_CORES_A",
        "cpu_cores_b": "REMOTE_CORES_B",
        "cpu_cores_c": "REMOTE_CORES_C",
        "venv_carla": "REMOTE_VENV_CARLA",
    }
    for field_name, env_var in env_map.items():
        val = os.environ.get(env_var)
        if val:
            raw[field_name] = val

    return RemoteConfig(**{k: v for k, v in raw.items() if k in RemoteConfig.__dataclass_fields__})
