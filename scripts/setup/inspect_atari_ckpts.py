import subprocess
from remote_config import load_config

cfg = load_config()


def test_ckpts():
    py_code = (
        "import torch; "
        "variants = ['S049a_mcts_sim20', 'S049b_mcts_sim35', 'S049c_mcts_sim50']; "
        "[print(f'VALID CHECKPOINT: {v} -> ' + str(len(torch.load("
        f"f'{cfg.remote_path(\"results/100k_benchmark\")}" + "/{v}_s42/mcts_offpolicy_BreakoutNoFrameskip-v4_s42_1790086674/checkpoints/model_latest.pt', "
        "map_location='cpu'))) + ' tensors') for v in variants]"
    )
    # Build a self-contained remote python one-liner via heredoc approach
    remote_cmd = (
        f"cd {cfg.workspace} && "
        f"{cfg.python()} -c \""
        "import torch; "
        "for v in ['S049a_mcts_sim20', 'S049b_mcts_sim35', 'S049c_mcts_sim50']: "
        f"  p = f'{cfg.remote_path(\"results/100k_benchmark\")}" + "/{v}_s42/mcts_offpolicy_BreakoutNoFrameskip-v4_s42_1790086674/checkpoints/model_latest.pt'; "
        "  sd = torch.load(p, map_location='cpu', weights_only=False); "
        "  print(f'VALID CHECKPOINT: {v} -> {len(sd)} tensors')"
        "\""
    )
    r = subprocess.run(cfg.ssh(remote_cmd), capture_output=True, text=True)
    print(r.stdout)
    if r.stderr:
        print("STDERR:", r.stderr)


if __name__ == "__main__":
    test_ckpts()
