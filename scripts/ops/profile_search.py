"""B23 step 1: where does one batched search step spend its time? (run on a box, repo root = cwd)"""
import sys, time
sys.path.insert(0, ".")
import numpy as np, torch
from atari_qwen.training.eval_ez_checkpoints import load_model
from atari_qwen.training.train_ez_offpolicy import run_search, to_input, make_vector_atari_envs

ck = sys.argv[1]; dev = torch.device("cuda")
for sims in (16, 64):
    _, a, model, mcts = load_model(__import__("pathlib").Path(ck), dev, sims)
    model.eval()
    envs = make_vector_atari_envs(a.env_id, num_envs=30, seed=10_000, clip_reward=False, episodic_life=False,
                                  frame_size=a.frame_size, grayscale=False, fire_reset=False, max_episode_steps=27000)
    obs, _ = envs.reset(seed=10_000)
    for _ in range(40):  # a few real steps so the roots are not the first frame
        _, _, act = run_search(model, mcts, to_input(obs, dev), add_noise=False)
        obs, *_ = envs.step(act)
    x = to_input(obs, dev)
    def timed(f, n=10):
        f(); torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(n): f()
        torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1000
    with torch.no_grad():
        t_init = timed(lambda: model.initial_inference(x))
        s, v, p = model.initial_inference(x)
        h = model.init_hidden(30, dev)
        act = torch.zeros(30, dtype=torch.long, device=dev)
        t_rec = timed(lambda: model.recurrent_inference(s, act, h))
    t_search = timed(lambda: run_search(model, mcts, x, add_noise=False))
    print(f"sims {sims}: initial_inference {t_init:.1f} ms, recurrent_inference {t_rec:.1f} ms x {sims} = {t_rec * sims:.0f} ms, "
          f"whole run_search {t_search:.0f} ms  -> model share {100 * (t_init + t_rec * sims) / t_search:.0f}%", flush=True)
    if sims == 16:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            run_search(model, mcts, x, add_noise=False); torch.cuda.synchronize()
        ev = prof.key_averages()
        n_sync = sum(e.count for e in ev if e.key in ("aten::item", "aten::_local_scalar_dense"))
        n_launch = sum(e.count for e in ev if e.key == "cudaLaunchKernel")
        gpu_us = sum(e.device_time_total for e in ev if e.device_type.name == "CUDA") if hasattr(ev[0], "device_type") else 0
        print(f"  one 16-sim search: {n_sync} host syncs (.item/bool), {n_launch} kernel launches")
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=8, max_name_column_width=40))
    envs.close()
