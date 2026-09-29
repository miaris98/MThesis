"""scripts/lib/machine.py: container limits and the job sizes derived from them."""
import math

from scripts.lib import machine as mc


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_cpu_quota_cgroup_v2(tmp_path):
    _write(tmp_path / "cpu.max", "1150000 100000")  # box W: 11.5 CPUs (S-086)
    assert mc.cpu_quota(tmp_path) == 11.5


def test_cpu_quota_v2_unlimited_is_none(tmp_path):
    _write(tmp_path / "cpu.max", "max 100000")
    assert mc.cpu_quota(tmp_path) is None


def test_cpu_quota_cgroup_v1(tmp_path):
    _write(tmp_path / "cpu" / "cpu.cfs_quota_us", "7020000")
    _write(tmp_path / "cpu" / "cpu.cfs_period_us", "100000")
    assert mc.cpu_quota(tmp_path) == 70.2  # box AB's quota (S-098)


def test_cpu_quota_v1_unlimited_is_none(tmp_path):
    _write(tmp_path / "cpu" / "cpu.cfs_quota_us", "-1")
    _write(tmp_path / "cpu" / "cpu.cfs_period_us", "100000")
    assert mc.cpu_quota(tmp_path) is None


def test_pids_and_memory_limits(tmp_path):
    _write(tmp_path / "pids.max", "2816")
    _write(tmp_path / "memory.max", str(64 * 10**9))
    meminfo = tmp_path / "meminfo"
    _write(meminfo, "MemTotal:       263000000 kB\n")
    assert mc.pids_max(tmp_path) == 2816
    assert math.isclose(mc.memory_limit_gb(tmp_path, meminfo), 64.0)


def test_unlimited_pids_is_none(tmp_path):
    _write(tmp_path / "pids.max", "max")
    assert mc.pids_max(tmp_path) is None


def test_parse_nvidia_smi_with_and_without_compute_cap():
    with_cc = mc.parse_nvidia_smi("0, NVIDIA GeForce RTX 3090, 24576, 1024, 8.6\n"
                                  "1, NVIDIA GeForce RTX 2080 Ti, 22528, 0, 7.5\n", with_cc=True)
    assert [g.index for g in with_cc] == [0, 1]
    assert with_cc[0].bf16 is True and with_cc[1].bf16 is False
    assert math.isclose(with_cc[0].memory_free_gb, 23.0)
    old_driver = mc.parse_nvidia_smi("0, A10, 23028, 500\n", with_cc=False)
    assert old_driver[0].bf16 is None


def test_ezv2_cpus_capped_by_pids():
    m = mc.Machine(cpus=48.0, host_cpus=48, pids_max=2816, memory_gb=64.0)
    assert mc.recommend(m)["ezv2_num_cpus"] == 2816 // mc.PIDS_PER_RAY_CPU  # 12, what box W needed


def test_ezv2_cpus_follow_quota():
    m = mc.Machine(cpus=11.5, host_cpus=48, pids_max=None, memory_gb=64.0)
    assert mc.recommend(m)["ezv2_num_cpus"] == 12


def test_carla_lanes_limited_by_cpu_and_vram():
    gpus = [mc.Gpu(0, "A10", 24.0, 0.0, 8.6)]  # one 24 GB GPU: 20.4 GB usable -> 2 lanes of 7.5 GB
    assert mc.recommend(mc.Machine(70.0, 128, None, 200.0, gpus))["carla_lanes"] == 2
    assert mc.recommend(mc.Machine(9.0, 128, None, 200.0, gpus))["carla_lanes"] == 1  # 9 cores / 5 per lane
    assert mc.recommend(mc.Machine(70.0, 128, None, 200.0, []))["carla_lanes"] == 0


def test_dataloader_workers_bounds():
    assert mc.recommend(mc.Machine(64.0, 64, None, None))["dataloader_workers"] == 16
    assert mc.recommend(mc.Machine(11.5, 48, None, None))["dataloader_workers"] == 9
    assert mc.recommend(mc.Machine(2.0, 2, None, None))["dataloader_workers"] == 1


def test_bf16_requires_every_gpu():
    mixed = [mc.Gpu(0, "3090", 24.0, 0.0, 8.6), mc.Gpu(1, "2080 Ti", 22.0, 0.0, 7.5)]
    assert mc.recommend(mc.Machine(16.0, 16, None, None, mixed))["bf16"] is False
