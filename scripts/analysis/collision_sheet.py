"""Contact sheet: for sampled unique collisions of each class, the front-camera frames 2 s, 1 s before and at impact (the policy's view)."""
import csv, glob, io, json, os, random, sys
import numpy as np
from PIL import Image, ImageDraw

S = sys.argv[1]
OUT = sys.argv[2]
U = list(csv.DictReader(open(os.path.join(S, "collision_unique.csv"))))
paths = {os.path.basename(p): p for p in glob.glob(os.path.join(S, "clips_epic220", "*", "clips", "*", "*.npz"))}
random.seed(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
classes = [("front/ego", lambda r: r["zone"] == "front" and r["mover"] in ("ego", "both")),
           ("side/other", lambda r: r["zone"] == "side" and r["mover"] == "other"),
           ("side/ego", lambda r: r["zone"] == "side" and r["mover"] in ("ego", "both")),
           ("rear", lambda r: r["zone"] == "rear"),
           ("never visible", lambda r: not (r["vis_now"] == "True" or r["vis_1s"] == "True" or r["vis_2s"] == "True"))]
W, H = 256, 128
rows = []
for name, fn in classes:
    sel = [r for r in U if fn(r)]
    for r in random.sample(sel, min(3, len(sel))):
        z = np.load(paths[r["file"]], allow_pickle=True)
        nb = int(z["n_before"])
        idx = [max(nb - 9, 0), max(nb - 5, 0), nb - 1]          # ~2 s, ~1 s before, at impact (4 Hz)
        ims = [Image.open(io.BytesIO(z["jpegs"][i])).convert("RGB").resize((W, H)) for i in idx]
        row = Image.new("RGB", (3 * W + 330, H), (20, 20, 20))
        for k, im in enumerate(ims):
            row.paste(im, (k * W, 0))
        d = ImageDraw.Draw(row)
        d.text((3 * W + 6, 4), f"{name}\n{r['arm']} {r['scenario']}\nfwd {r['fwd']} right {r['right']}\nego {r['ego_speed']} m/s other {r['other_speed']}\n"
                               f"closing ego {r['ego_closing']} other {r['other_closing']}\nvis now/1s/2s {r['vis_now'][0]}/{r['vis_1s'][0]}/{r['vis_2s'][0]}\n"
                               f"{r['other_type'][:30]}", fill=(255, 255, 0))
        rows.append(row)
sheet = Image.new("RGB", (rows[0].width, H * len(rows)))
for i, r in enumerate(rows):
    sheet.paste(r, (0, i * H))
sheet.save(OUT, quality=85)
print("sheet", OUT, len(rows), "rows")
