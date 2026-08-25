"""Hardware telemetry monitor querying GPU VRAM, System RAM, and CPU usage."""
from typing import Dict
import torch


class HardwareMonitor:
    """Hardware resource telemetry monitor for GPU and CPU utilization."""

    @staticmethod
    def get_metrics() -> Dict[str, float]:
        """Fetch real-time GPU VRAM, System RAM, and CPU load metrics."""
        metrics = {
            "gpu_mem_used_mb": 0.0,
            "gpu_mem_total_mb": 0.0,
            "gpu_mem_pct": 0.0,
            "sys_cpu_pct": 0.0,
            "sys_ram_used_gb": 0.0,
            "sys_ram_total_gb": 0.0
        }
        
        try:
            if torch.cuda.is_available():
                mem_used = torch.cuda.memory_allocated() / (1024.0 ** 2)
                mem_total = torch.cuda.get_device_properties(0).total_memory / (1024.0 ** 2)
                metrics["gpu_mem_used_mb"] = round(mem_used, 1)
                metrics["gpu_mem_total_mb"] = round(mem_total, 1)
                metrics["gpu_mem_pct"] = round((mem_used / max(1.0, mem_total)) * 100.0, 1)
        except Exception:
            pass

        try:
            import psutil
            metrics["sys_cpu_pct"] = round(psutil.cpu_percent(interval=None), 1)
            vm = psutil.virtual_memory()
            metrics["sys_ram_used_gb"] = round(vm.used / (1024.0 ** 3), 2)
            metrics["sys_ram_total_gb"] = round(vm.total / (1024.0 ** 3), 2)
        except Exception:
            pass

        return metrics
