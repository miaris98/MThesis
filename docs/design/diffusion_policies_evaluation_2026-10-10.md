# Should we use a diffusion model for CARLA or Atari? (evaluation of 2026-10-10, S-139)

Question from the user: diffusion models are said to give good trajectories and may fit CARLA; evaluate them, and evaluate them as policy networks for reinforcement learning as well.

Short answer (my judgement): **not as a replacement of the policy network and not for Atari or RL. For CARLA only as an anchor-based, few-step trajectory head, and only behind a cheap data check that tells us whether multi-modality is our actual failure.** Reasons and the evidence follow. Numbers marked *paper* come from the abstracts or tables of the cited papers (fetched 2026-10-10); sensor setups of the Bench2Drive entries were not checked, so they are context, not comparisons with our single-camera system.

## 1. What our failures are

- E and J drive straight into stopped vehicles and obstacles; 80% of J's collision points are systematic (S-120). Obstacle / passing routes are the gap to TF++ (S-134: ~70 on the screen subset against TF++ 84.2 official).
- A regression head that averages a "wait" mode and a "pass" mode produces the average of the two, a path through the obstacle. That is the textbook case for a multi-modal head, and it is the reason TODO A41 (two-mode head) and A55 (generative head) exist.
- Head-only changes so far did not beat E15: A15 plain / cascade 64.9 / 65.2, A38 65.4, A39 62.7-64.1 against E15 ~64.5, with head-only seed noise ~4 DS (S-132). The vision input is also a suspect: 16 average-pooled tokens (A54, A45).
- Not yet shown: that our expert data are bimodal at the failure frames and that our head predicts the mean between the modes. That is the cheap test in Section 5.

## 2. Evidence for diffusion / flow heads in driving

| source | what it shows | caveat |
|---|---|---|
| DiffusionDrive, CVPR 2025 (*paper*, NAVSIM navtest, ResNet-34, 20 anchors) | TransFuser (regression) 84.0 PDMS; **vanilla diffusion policy on the same TransFuser 84.6 (+0.6, 20 steps, 7 FPS, 11% mode diversity)**; **truncated diffusion from anchors + cascade decoder 88.1 (+4.1), 2 steps (1 step 87.9, 3 steps 88.1), 45 FPS on a 4090, 74% mode diversity**; nuScenes open loop L2 0.57 m / collisions 0.08% against VAD 0.72 / 0.22 | NAVSIM is a non-reactive pseudo-simulation, not Bench2Drive closed loop |
| BridgeDrive, ICLR 2026 (*paper* abstract) | diffusion-bridge refinement of anchor trajectories, Bench2Drive closed-loop success rate +7.72 points over prior art with PDM-Lite data and +2.45 with LEAD data; works with ODE samplers for real time | absolute numbers not in the abstract; a secondary snippet quotes SR 74.99 (PDM-Lite) / 89.25 (LEAD) and DS 86.87, unchecked |
| DiffRefiner, AAAI 2026 (*paper* abstract) | proposal decoder with anchors + diffusion refiner: Bench2Drive **DS 87.1, SR 71.4** | sensors and data of the entry not checked |
| BevAD flow matching (from our own literature note, `policy_options_datasets_timeseries_2026-10-04.md`) | flow matching let performance keep scaling with data: 51.7% -> 56.2% SR on identical data | one paper, via our note |

What this says: the gain is **not from "diffusion"**; vanilla diffusion added 0.6 PDMS. It is from **anchors (a multi-mode prior), a truncated schedule (2 steps) and conditioning on scene features**, i.e. a small mixture-of-anchors model refined by a couple of denoising steps. The Bench2Drive entries with strong closed-loop numbers all refine anchors.

## 3. Diffusion / flow as the policy network (no RL), by family

| head | multi-modal | closed-loop determinism | latency | tractable log-likelihood | fit to our setup |
|---|---|---|---|---|---|
| regression + binned speed (now) | no | yes | ~0 | yes (speed bins) | the baseline; averages modes |
| anchor classifier + offsets (A41 / MDN / WTA) | yes, discrete | yes (pick top mode) | ~0 | yes | cheapest multi-modal option; needs a mode-selection rule |
| truncated diffusion / flow from anchors (DiffusionDrive-style) | yes | needs a fixed seed or mean of samples + a scorer | 2 steps ~ 8 ms on a 4090 (*paper*) | no (ELBO only) | feasible; the selector (which sample to execute) is the open design problem, DiffusionDrive uses a learned confidence |
| vanilla DDPM, 20+ steps | yes | stochastic | 130 ms (*paper*) | no | no: slower, +0.6 PDMS |
| autoregressive trajectory tokens (VLA style) | yes | yes | grows with length | yes | needs a language/VLA model, outside our scope |

Closed-loop specifics that open-loop papers do not test: a stochastic sample can flip between "wait" and "pass" from one control step to the next (dithering in front of an obstacle); we would need per-route committed mode or anchor-conditioned selection with hysteresis (this is what A41's filtered intent posterior does at lower cost).

## 4. Diffusion for reinforcement learning

- **CARLA closed-loop RL (TODO A20):** DPPO fine-tunes a pre-trained diffusion policy with PPO by treating denoising as an MDP and using the per-step Gaussian likelihood (*paper*, ICLR 2025, robotics). A later study (*paper*) found that from random initialisation DPPO can be less sample-efficient than a plain MLP policy with standard RL, so the claim is for fine-tuning a good pre-trained policy. Our binding constraint is simulator throughput (~20-110 route-runs per hour per box), far too slow for online RL sample counts, whichever policy class. Offline variants (Diffusion-QL: a Q-guided diffusion loss, state of the art on most D4RL tasks per its authors) need a reward-labelled offline dataset and a critic; our PDM-Lite data are expert-only. **Verdict: no.**
- **Atari Breakout (4 discrete actions):** the policy is a categorical distribution trained on MCTS visit counts; there is nothing for a diffusion policy to model better. **Diffusion as a world model** (DIAMOND, NeurIPS 2024): best among world-model agents on Atari-100k with mean HNS 1.46 but **Breakout 132.5** against EfficientZero-V2 400.1 and our 359.4, ~2.9 days per run on a 4090, and a denoising loop inside every MCTS node expansion would multiply search cost (we run 16-64 simulations per step). **Verdict: no.**

## 5. Recommendation

1. **Do not start a diffusion project.** It cannot fix the vision bottleneck and our head-only record is flat.
2. **Gate (1 box-hour, no training): is multi-modality the failure?** On the dumps of J / E15 collisions with stopped vehicles (TODO A1 / A44), compare the expert trajectory at the same frames (PDM-Lite pass vs wait) with the head's output: if the expert branches into two clusters and the head predicts their mean (path between them), multi-modality is the problem; if the head predicts one mode and still collides, it is perception/speed and no head will help.
3. **If the gate is positive:** build the cheap discrete version first (A41: stay / pass anchors, gated by intent, deterministic); only if it helps, test a 2-step flow-matching refinement on the same anchors (A55) on the pooled-token cache (A59), 2 seeds, J-cache control, then the 220 routes at Epic (A61).
4. **If the gate is negative:** spend the effort on A54 (tokens) and the Epic re-baseline.
5. Keep Atari on the EfficientZero line: more seeds at 100k on the standard protocol (n = seeds), not new model classes.
