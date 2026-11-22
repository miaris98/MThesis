#!/usr/bin/env bash
# CARLA render test for an eval box (rebuilt 2026-11-21 from the 11-20 render_test.log; the original lived in a cleared scratchpad).
# Checks the install, then starts CARLA off-screen (Epic, port 2900), connects a client, loads Town12 and renders camera frames in
# synchronous mode, loads Town03 and spawns the four vehicles that crashed boxes before. Prints RENDER_OK frames N on success.
: "${WORKSPACE:=/workspace}"; WORKSPACE=${WORKSPACE%/}   # the box layout (scripts/lib/common.sh); WORKSPACE=... overrides
C=$WORKSPACE/carla; PYC=$WORKSPACE/venv_carla/bin/python; P=2900
echo "CarlaUE4.sh: $(ls $C/CarlaUE4.sh 2>/dev/null)"
echo "maps: $(ls $C/CarlaUE4/Content/Carla/Maps 2>/dev/null | tr '\n' ' ' | cut -c1-200)"
echo "Town12/13 packages: $(ls -d $C/CarlaUE4/Content/Carla/Maps/Town12 $C/CarlaUE4/Content/Carla/Maps/Town13 2>/dev/null | wc -l)"
echo "nvidia icd: $(ls /etc/vulkan/icd.d/ /usr/share/vulkan/icd.d/ 2>/dev/null | grep -i nvidia | tr '\n' ' ')"
echo "libGLX_nvidia: $(ls /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 2>/dev/null)"
LOG=$WORKSPACE/ladder/render_test_carla.log
# CARLA refuses to run as root: start it as carlauser with its own XDG_RUNTIME_DIR and -vulkan, as run_bench2drive.sh's shim does
X=/tmp/runtime-carlauser-render-test; mkdir -p $X; chown carlauser:carlauser $X; chmod 700 $X; touch $LOG; chmod 666 $LOG
(setsid nohup su carlauser -c "export XDG_RUNTIME_DIR=$X; cd $C && ./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=$P -vulkan -quality-level=Epic" > $LOG 2>&1 < /dev/null &)
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":$P " && break; sleep 5; done
echo "listening after ~$((i * 5)) s: $(ss -ltn 2>/dev/null | grep -c ":$P ")"
timeout 600 $PYC - $P <<'PY'
import sys, time
import carla
c = carla.Client("localhost", int(sys.argv[1])); c.set_timeout(120.0)
w = c.get_world()
print(f"CLIENT_OK server {c.get_server_version()} map {w.get_map().name}", flush=True)
t0 = time.time(); w = c.load_world("Town12"); print(f"LOAD_TOWN12_OK in {time.time() - t0:.0f} s {w.get_map().name}", flush=True)
# rendering on the small Town03 map (spawning on the large Town12 map in synchronous mode segfaulted this test on 2026-11-21)
t0 = time.time(); w = c.load_world("Town03"); print(f"LOAD_TOWN03_OK in {time.time() - t0:.0f} s {w.get_map().name}", flush=True)
bp = w.get_blueprint_library(); pts = w.get_map().get_spawn_points()
ego = w.spawn_actor(bp.find("vehicle.lincoln.mkz_2020"), pts[0])
cam_bp = bp.find("sensor.camera.rgb"); cam_bp.set_attribute("image_size_x", "1024"); cam_bp.set_attribute("image_size_y", "512"); cam_bp.set_attribute("fov", "110")
cam = w.spawn_actor(cam_bp, carla.Transform(carla.Location(x=-1.5, z=2.0)), attach_to=ego)
frames = []
cam.listen(lambda im: frames.append(im.frame))
s = w.get_settings(); s.synchronous_mode = True; s.fixed_delta_seconds = 0.05; w.apply_settings(s)
for _ in range(20):
    w.tick()
time.sleep(1.0)
cam.stop(); cam.destroy(); ego.destroy()
print(("RENDER_OK" if len(frames) >= 10 else "RENDER_FAILED") + f" frames {len(frames)}", flush=True)
s.synchronous_mode = False; s.fixed_delta_seconds = None; w.apply_settings(s)
ok = 0
for i, name in enumerate(["vehicle.bh.crossbike", "vehicle.diamondback.century", "vehicle.gazelle.omafiets", "vehicle.lincoln.mkz_2020"]):
    a = w.try_spawn_actor(bp.find(name), pts[i + 1])
    print(f"spawn {name} {'OK' if a else 'FAILED'}", flush=True)
    if a:
        ok += 1; a.destroy()
print(f"SPAWN_TEST_DONE {ok} of 4", flush=True)
PY
echo "py rc=$?"
echo "== carla log errors:"; grep -i -E "error|signal|crash" $LOG | head -5
for pid in $(pgrep -f "CarlaUE4.*-carla-rpc-port=$P"); do kill $pid; done; sleep 8
for pid in $(pgrep -f "CarlaUE4.*-carla-rpc-port=$P"); do kill -9 $pid; done
echo "gpu after stop: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1)"
