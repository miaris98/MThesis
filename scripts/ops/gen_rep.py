import random
obst = open("routes_obst.txt").read().strip().split(",")
r19 = [r for r in open("routes_b2d20.txt").read().strip().replace("\n", ",").split(",") if r]
crash = {"25854", "25896", "25955", "24816"}
routes = [r for r in dict.fromkeys(obst + r19) if r not in crash]
print(len(routes))
random.Random(3).shuffle(routes)
lanes = [("O", 2000 + 400 * i, i % 2) for i in range(8)] + [("P", 2000 + 400 * i, i % 4) for i in range(12)]
out = {"O": [], "P": []}
E = "carla_armE_aug1_hires/model_epoch_015.pth"
J = ["carla_armJ_ft_obst", "carla_armJ_ft_obst_s1", "carla_armJ_ft_obst_s2", "carla_armJ_ft_obst_s3"]
for k, (b, port, gpu) in enumerate(lanes):
    rl = ",".join(routes[3 * k: 3 * k + 3])
    jobs = [f"CKL:a9_E15_r{b}a_{port}:{E}:{rl}"]
    jobs += [f"CKL:{'j20' if s == 0 else f'j{s}s20'}_r{b}_{port}:{J[s]}/model_epoch_020.pth:{rl}" for s in range(4)]
    jobs += [f"CKL:a9_E15_r{b}b_{port}:{E}:{rl}"]
    out[b].append(f"{b} {port} {gpu} " + " ".join(jobs))
for b in out:
    open(f"lanes_{b}_rep.txt", "w", newline="\n").write("\n".join(out[b]) + "\n")
