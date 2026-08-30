"""
Download Training Artifacts & Telemetry from Vast.ai / Cloud Instance to Local PC.

Usage examples:
    # 1. Interactive mode (prompts for SSH port and host):
    python download_from_vastai.py

    # 2. Command-line flags:
    python download_from_vastai.py --ssh-cmd "ssh -p 12345 root@ssh5.vast.ai"
    python download_from_vastai.py -p 12345 -H ssh5.vast.ai --all
    python download_from_vastai.py -p 12345 -H ssh5.vast.ai --telemetry-only
    python download_from_vastai.py -p 12345 -H ssh5.vast.ai --checkpoints-only
"""
import os
import sys
import re
import argparse
import subprocess
from pathlib import Path


DEFAULT_REMOTE_PATHS = {
    "telemetry": "/workspace/runs/training_telemetry.csv",
    "runs_dir": "/workspace/runs",
    "best_checkpoint": "/workspace/checkpoints/ppo_carla_best.pth",
    "latest_checkpoint": "/workspace/checkpoints/ppo_carla_latest.pth",
    "train_state": "/workspace/checkpoints/train_state.json",
    "checkpoints_dir": "/workspace/checkpoints",
    "eval_video": "/workspace/eval_video.mp4",
}


def parse_ssh_cmd(ssh_cmd: str):
    """Extract port, user, and host from a typical ssh string like 'ssh -p 12345 root@ssh5.vast.ai'."""
    port_match = re.search(r'-p\s+(\d+)', ssh_cmd)
    port = port_match.group(1) if port_match else "22"
    
    # Extract user@host
    host_match = re.search(r'([a-zA-Z0-9_\-]+@[a-zA-Z0-9\.\-]+)', ssh_cmd)
    if host_match:
        user_host = host_match.group(1)
    else:
        # Fallback to host without user
        parts = [p for p in ssh_cmd.split() if not p.startswith('-') and p != 'ssh' and p != port]
        user_host = f"root@{parts[-1]}" if parts else "root@localhost"
        
    return port, user_host


def run_scp_download(port: str, user_host: str, remote_file_or_dir: str, local_dest: str, is_dir: bool = False):
    """Execute SCP command on Windows/Linux."""
    os.makedirs(os.path.dirname(os.path.abspath(local_dest)), exist_ok=True)
    
    cmd = ["scp", "-P", str(port)]
    if is_dir:
        cmd.append("-r")
    cmd.extend([f"{user_host}:{remote_file_or_dir}", local_dest])
    
    print(f"\n--> Downloading: {remote_file_or_dir} -> {local_dest}")
    print(f"    Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"✓ Download complete: {local_dest}")
        return True
    else:
        print(f"⚠️  Download failed or remote file does not exist: {remote_file_or_dir}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download CARLA RL artifacts & telemetry from Vast.ai to local PC.")
    parser.add_argument("--ssh-cmd", type=str, help="Full SSH command string from Vast.ai console (e.g. 'ssh -p 12345 root@ssh5.vast.ai')")
    parser.add_argument("-p", "--port", type=str, help="SSH Port (e.g. 12345)")
    parser.add_argument("-H", "--host", type=str, help="SSH Host or user@host (e.g. root@ssh5.vast.ai)")
    parser.add_argument("--dest", type=str, default=str(Path(__file__).parent.resolve()), help="Local destination directory (default: repo root)")
    parser.add_argument("--telemetry-only", action="store_true", help="Download only training_telemetry.csv")
    parser.add_argument("--checkpoints-only", action="store_true", help="Download only checkpoint weights & state")
    parser.add_argument("--video-only", action="store_true", help="Download only evaluation video")
    parser.add_argument("--all", action="store_true", help="Download all runs, checkpoints, telemetry, and videos")
    args = parser.parse_args()

    print("=" * 65)
    print("   📥 Vast.ai -> Local PC Artifact Downloader (CARLA RL)     ")
    print("=" * 65)

    port = args.port
    user_host = args.host

    if args.ssh_cmd:
        port, user_host = parse_ssh_cmd(args.ssh_cmd)
    elif not port or not user_host:
        print("\nPaste your Vast.ai SSH command or enter connection details.")
        print("Example: ssh -p 38472 root@ssh5.vast.ai\n")
        user_input = input("Enter SSH command or press Enter to input Port/Host manually: ").strip()
        if user_input:
            port, user_host = parse_ssh_cmd(user_input)
        else:
            port = input("Enter SSH Port (e.g. 38472): ").strip()
            host = input("Enter Host (e.g. ssh5.vast.ai or root@ssh5.vast.ai): ").strip()
            user_host = host if "@" in host else f"root@{host}"

    if not port or not user_host:
        print("Error: SSH port and host are required.")
        sys.exit(1)

    print(f"\n[Target Instance]: {user_host} on port {port}")
    print(f"[Local Destination]: {args.dest}")

    local_runs_dir = os.path.join(args.dest, "runs")
    local_checkpoints_dir = os.path.join(args.dest, "checkpoints")
    os.makedirs(local_runs_dir, exist_ok=True)
    os.makedirs(local_checkpoints_dir, exist_ok=True)

    # 1. Telemetry CSV
    if args.telemetry_only or args.all or (not args.checkpoints_only and not args.video_only):
        csv_dest = os.path.join(local_runs_dir, "training_telemetry.csv")
        run_scp_download(port, user_host, DEFAULT_REMOTE_PATHS["telemetry"], csv_dest)

    # 2. Checkpoints
    if args.checkpoints_only or args.all or (not args.telemetry_only and not args.video_only):
        # Best model
        best_dest = os.path.join(local_checkpoints_dir, "ppo_carla_best.pth")
        run_scp_download(port, user_host, DEFAULT_REMOTE_PATHS["best_checkpoint"], best_dest)
        
        # Latest model
        latest_dest = os.path.join(local_checkpoints_dir, "ppo_carla_latest.pth")
        run_scp_download(port, user_host, DEFAULT_REMOTE_PATHS["latest_checkpoint"], latest_dest)

        # Train state JSON
        state_dest = os.path.join(local_checkpoints_dir, "train_state.json")
        run_scp_download(port, user_host, DEFAULT_REMOTE_PATHS["train_state"], state_dest)

    # 3. Eval Video
    if args.video_only or args.all:
        video_dest = os.path.join(args.dest, "eval_video.mp4")
        if not run_scp_download(port, user_host, DEFAULT_REMOTE_PATHS["eval_video"], video_dest):
            # Check eval_video_best.mp4
            run_scp_download(port, user_host, "/workspace/eval_video_best.mp4", os.path.join(args.dest, "eval_video_best.mp4"))

    print("\n" + "=" * 65)
    print("   ✨ Download Routine Complete!                             ")
    print("=" * 65)


if __name__ == "__main__":
    main()
