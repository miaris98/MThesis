"""Labels-only bundle restore (A59): measurements_<set>.tgz + the arm-M cache json -> a data dir of measurements plus empty image stubs (the pooled cache supplies the features).
usage: restore_bundle.py BUNDLE_DIR PARENT SET     SET = wor_traj_m (J + lead families) or wor_trajectories (the J recipe); both are restored under the directory name `wor_traj_m`
because the one cache (wor_traj_m_*.json, built over the superset) is matched to the data dir by name; frames the dataset does not index are simply unused.
Then train with:  --data_dir PARENT/wor_traj_m --pooled_cache BUNDLE_DIR/wor_traj_m_288x768o4ro_regnety_032_g4x4"""
import json, os, subprocess, sys
bundle, parent, which = sys.argv[1:4]
dest = os.path.join(parent, "wor_traj_m")
os.makedirs(dest, exist_ok=True)
tgz = os.path.join(bundle, f"measurements_{which}.tgz")
subprocess.check_call(["tar", "-xzf", tgz, "-C", dest])
meta = json.load(open(os.path.join(bundle, "wor_traj_m_288x768o4ro_regnety_032_g4x4.json")))
keys = set(meta["keys"])
members = subprocess.check_output(["tar", "-tzf", tgz]).decode().split("\n")
n = 0
for m in members:
    if not m.endswith(".json.gz"):
        continue
    k = m.lstrip("./").replace("/measurements/", "/rgb/").replace(".json.gz", ".jpg")
    if k in keys:
        p = os.path.join(dest, k)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").close()
        n += 1
print("stubs", n, "of", len(members) - 1, "measurement files ->", dest)
