# CARLA Headless Rendering & NVIDIA Vulkan Driver Troubleshooting Guide

This guide covers the case where **CARLA exits immediately with `EXIT CODE: 1` and no log file** on a Vast.ai (or any Docker/container) GPU instance, while CUDA training works perfectly. The root cause is that the container has the NVIDIA **compute** stack but not the **graphics** stack, so CARLA's renderer has no Vulkan driver to talk to.

First diagnosed: 2026-09-06, Vast.ai instance `C.50057329` (RTX A4000, driver 595.71.05, Ubuntu 24.04 Noble).

---

## 1. Symptom

`CarlaUE4.sh` dies instantly. The only output is the engine banner and an unrelated `xdg-user-dir` warning:

```
4.26.2-0+++UE4+Release-4.26 522 0
Disabling core dumps.
sh: 1: xdg-user-dir: not found
EXIT CODE: 1
```

Key characteristics that identify *this* failure specifically:

- **No `CarlaUE4.log` is ever written** (`/workspace/carla/CarlaUE4/Saved/Logs/` stays empty or absent). The process dies before Unreal initializes logging, so there is no engine-level error to read.
- Exit code is **`1`, not a signal.** The shell does *not* print `Illegal instruction` or `Segmentation fault`, meaning the binary called `exit(1)` voluntarily rather than crashing.
- **Both `-vulkan` and `-opengl` fail identically.** Switching renderers does not help, which is a strong hint the problem is below the renderer choice.
- CUDA/PyTorch training works fine on the same box.

---

## 2. What This Is NOT

These were all checked and ruled out on the original diagnosis. Do not spend time on them unless a check below actually fails:

| Suspected cause | How to rule it out | Result in this case |
|---|---|---|
| Running as root | `Refusing to run with the root privileges.` appears explicitly if so — run via `su carlauser` | Not the cause (message absent once using `carlauser`) |
| Missing AVX2/FMA CPU flags (see `.agents/rules/carla_hardware_compatibility.md`) | `cat /proc/cpuinfo \| grep -m1 flags \| tr ' ' '\n' \| grep -E '^(avx2\|fma)$'` | Both present (i7-13700) |
| Disk full | `df -h /workspace` | 120 GB free |
| File ownership / write permissions | `su carlauser -c "mkdir -p /workspace/carla/CarlaUE4/Saved/Logs && echo WRITE_OK"` | `WRITE_OK` |
| GPU device node permissions | `ls -la /dev/nvidia*` then `su carlauser -c "nvidia-smi"` | Device nodes are `crw-rw-rw-`; `nvidia-smi` works as `carlauser` |
| Stale process holding port 2000 | `ss -tlnp \| grep 2000` | Nothing listening |
| Wrong renderer flag | Try `-opengl` instead of `-vulkan` | Fails identically |

---

## 3. Diagnosis

### 3.1 Trace the syscalls

Since Unreal writes no log, `strace` is the only way to see the real failure point:

```bash
apt-get install -y strace
su carlauser -c "strace -f -e trace=open,openat,mmap,exit_group \
  /workspace/carla/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping CarlaUE4 \
  -carla-port=2000 -RenderOffScreen -nosound -vulkan -quality-level=Low" 2>&1 | tail -60
```

The signature of this issue is the Vulkan loader scanning every ICD directory, finding **no driver manifest**, and immediately calling `exit_group(1)`:

```
openat(AT_FDCWD, "/usr/share/vulkan/icd.d", ...) = -1 ENOENT (No such file or directory)
exit_group(1)                           = ?
```

### 3.2 Confirm with vulkaninfo

```bash
apt-get install -y vulkan-tools
vulkaninfo --summary
```

Two distinct failure modes, which need **different** fixes:

- `vkCreateInstance: Found no drivers!` with **no ICD manifest on disk** → the graphics stack was never installed (§4).
- `ERROR_INCOMPATIBLE_DRIVER` with a manifest present → userspace/kernel **version mismatch** (§4.2). This is also what you get if you "fix" the first case with the wrong apt package.

> **Third variant, seen 2026-09-08 on a different Vast.ai host (V100, driver 580.178.04):**
> the manifest was present (`/etc/vulkan/icd.d/nvidia_icd.json`) but the library it points
> at — `libGLX_nvidia.so.0` — **did not exist anywhere on the filesystem**. `vulkaninfo`'s
> loader still reports `ERROR_INCOMPATIBLE_DRIVER`, identically to the real mismatch case,
> because that is the loader's generic "the ICD I found didn't work" message — it does not
> distinguish "wrong version" from "no file at all". Check which one you actually have
> before following §4.2's mismatch framing:
> ```bash
> cat /etc/vulkan/icd.d/*.json                              # what the manifest expects
> ls -la /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so*         # what actually exists
> ```
> An empty `ls` means "never installed" (this variant), not "wrong version" — proceed
> straight to §4.2's `.run` installer; there is nothing to purge first, because nothing was
> ever there to conflict with it. The fix is identical, but the diagnosis in §3.3 below
> (`dpkg -l | grep -i nvidia`) is the more reliable signal for this variant, since it will
> show no `libnvidia-gl-*` package at all rather than a wrong-version one.

### 3.3 Confirm the underlying cause

```bash
dpkg -l | grep -i nvidia
```

If this lists only CUDA/NCCL/Nsight packages and **no `libnvidia-gl-*`**, the container was started with `NVIDIA_DRIVER_CAPABILITIES=compute,utility`. The NVIDIA container runtime injected the compute libraries (so PyTorch works) but not the OpenGL/Vulkan libraries (so CARLA cannot render).

---

## 4. The Fix

### 4.1 Do NOT just `apt-get install libnvidia-gl-<version>`

This is the intuitive move and it **makes things worse**. The apt package version will almost never match the host's kernel module exactly:

```bash
cat /proc/driver/nvidia/version     # kernel module, e.g. 595.71.05
ls -la /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so*   # apt gave us, e.g. 595.84
```

NVIDIA requires the userspace libraries and the kernel module to match **exactly**. A mismatch installs a valid-looking ICD manifest that then fails with `ERROR_INCOMPATIBLE_DRIVER` — arguably a worse state than having no manifest at all, because the failure now looks like a Vulkan bug rather than a missing stack.

If you already did this, purge it first:

```bash
apt-get purge -y libnvidia-gl-<version> libnvidia-common-<version>
```

### 4.2 Install the exactly-matching userspace driver

Read the kernel module version and download that precise `.run` installer. Three-part versions (e.g. `595.71.05`) are datacenter-branch drivers and live under `/tesla/`; two-part versions live under `/XFree86/`:

```bash
DRIVER_VERSION=$(cat /proc/driver/nvidia/version | grep -oP '(?<=Module for x86_64  )[0-9.]+')
echo "Kernel driver: $DRIVER_VERSION"

cd /workspace
wget -q --show-progress "https://us.download.nvidia.com/tesla/${DRIVER_VERSION}/NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run" \
  || wget -q --show-progress "https://us.download.nvidia.com/XFree86/Linux-x86_64/${DRIVER_VERSION}/NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run"
```

The installer needs `modprobe` present even though we are not building a kernel module, so install `kmod` first or it aborts with `Unable to find the module utility 'modprobe'`:

```bash
apt-get install -y kmod

sh /workspace/NVIDIA-Linux-x86_64-${DRIVER_VERSION}.run \
  --no-kernel-module --silent --install-libglvnd \
  --no-nvidia-modprobe --no-rebuild-initramfs
```

> **`--no-kernel-module` is mandatory.** The kernel module belongs to the Vast.ai host and is shared with other containers. Only the userspace libraries may be installed from inside a container.

### 4.3 Verify

```bash
ls -la /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so*   # must match the kernel version exactly
vulkaninfo --summary                                  # must list the GPU, no ERROR_INCOMPATIBLE_DRIVER
```

Then launch CARLA:

```bash
su carlauser -c "/workspace/carla/CarlaUE4.sh -carla-port=2000 -RenderOffScreen -nosound -vulkan -quality-level=Low" \
  > /workspace/carla_server.log 2>&1 &
sleep 20
python /workspace/MThesis/test_connection.py --host 127.0.0.1 --port 2000
```

Expected:

```
Successfully connected to Carla Simulator!
Client Version: 0.9.15
Server Version: 0.9.15
```

---

## 5. Alternatives If the `.run` Installer Is Unavailable

If the exact driver version is not publicly downloadable (both URLs 404), this container cannot render and the options are:

1. **Use the CARLA Docker image**, which ships its own matched GL stack — already supported by `carla_runner.py`:
   ```bash
   python carla_runner.py --docker --port 2000
   ```
   Requires Docker to be usable inside the instance.

2. **Rent an instance with graphics capability enabled** (`NVIDIA_DRIVER_CAPABILITIES=all`). This is the durable fix, since a compute-only container will hit this on every fresh box.

Before destroying an instance, save trained artifacts so they do not need to be recomputed:

```bash
tar czf /workspace/wor_checkpoints.tar.gz \
  -C /workspace/MThesis checkpoints/ weights/wor_nc.pth
```

---

## 6. Quick Reference

| Check | Command | Healthy result |
|---|---|---|
| Kernel driver version | `cat /proc/driver/nvidia/version` | e.g. `595.71.05` |
| Userspace lib version | `ls /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so*` | **same** version as kernel |
| Vulkan ICD manifest | `ls /usr/share/vulkan/icd.d/` | contains `nvidia_icd.json` |
| Vulkan works | `vulkaninfo --summary` | lists the GPU |
| Graphics stack installed | `dpkg -l \| grep libnvidia-gl` | present, or driver installed via `.run` |
| CARLA reachable | `python test_connection.py --host 127.0.0.1 --port 2000` | `Successfully connected` |

---

## 7. Note for `setup_vastai.sh`

`setup_vastai.sh` currently installs `libvulkan1` (the Vulkan **loader**) but not an NVIDIA Vulkan **driver/ICD**, and does not verify that Vulkan can actually enumerate a GPU. On containers launched with the graphics capability this is fine, because the runtime injects the driver; on compute-only containers it silently produces the failure documented above, which then surfaces much later as an unexplained CARLA exit. Adding a `vulkaninfo --summary` check to step 7 (alongside `test_connection.py`) would catch it at setup time rather than at first evaluation run.
