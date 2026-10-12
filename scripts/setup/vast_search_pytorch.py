#!/usr/bin/env python3
"""
vast_search_pytorch.py — Search, filter, inspect, and export available Vast.ai instances.

Features:
- Configurable filters matching Vast.ai console:
    * Cloud & verification: secure/datacenter, verified, rented, external
    * Disk & storage: allocated storage / container size (default: 157.59 GB), min disk space
    * Host reliability: min reliability % (e.g., 85.34%)
    * Duration: min instance duration (hours/days)
    * Software & Drivers: min cuda version, min driver version, ubuntu OS version
    * Pricing limits: max $/hour, max $/TFLOPS/hr, max $/TB up/down
    * GPU Specs: GPU count, min TFLOPs, per-GPU VRAM, total VRAM, memory bandwidth, PCIe bandwidth, NVLink bandwidth, DLPerf
    * CPU & Host: min CPU cores, min CPU RAM, min CPU GHz, disk bandwidth, inet up/down speed, open direct ports
- PyTorch template presets (Vast official PyTorch, PyTorch cuDNN Devel/Runtime)
- Detailed pricing breakdown:
    * Total $/hr
    * GPU $/hr
    * Storage cost ($/hr for specified disk + $/GB/month)
    * Network upload & download costs ($/TB and $/GB)
- Output formats:
    * Rich CLI table / summaries
    * JSON export
    * CSV export
    * Rent command generator (ready-to-run CLI commands)
"""

import sys
import os
import json
import csv
import argparse
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

# Locate vastai CLI executable
VAST_CLI_CANDIDATES = [
    r"C:\Users\miari\anaconda3\envs\graphtools\Scripts\vastai.exe",
    r"C:\Users\miari\anaconda3\Scripts\vastai.exe",
    shutil.which("vastai") or "vastai"
]

def get_vast_cli() -> str:
    for cand in VAST_CLI_CANDIDATES:
        if cand and Path(cand).exists():
            return cand
    return "vastai"

# Default template presets
PYTORCH_TEMPLATES = {
    "vast": {
        "id": 266995,
        "hash": "305ac3ffd3e42e0d9ad1f4ae14729ec2",
        "name": "PyTorch (Vast Official)",
        "image": "vastai/pytorch:@vastai-automatic-tag",
    },
    "devel": {
        "id": 215736,
        "hash": "b773dcf2ce56e1b863e081be8e51dac3",
        "name": "PyTorch 2.5 cuDNN Devel",
        "image": "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
    },
    "runtime": {
        "id": 212406,
        "hash": "8c25ceee8d33d6bf923b21c7597cfb08",
        "name": "PyTorch 2.5 cuDNN Runtime",
        "image": "pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel",
    }
}


def parse_driver_tuple(driver_str: Optional[str]) -> tuple:
    if not driver_str:
        return (0, 0, 0)
    parts = []
    for p in str(driver_str).replace(",", ".").split("."):
        try:
            parts.append(int(p))
        except ValueError:
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def query_offers(
    cli: str,
    allocated_storage: float,
    pricing_type: str = "on-demand",
    limit: int = 1500,
    api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Query offers from Vast.ai CLI with raw JSON output."""
    cmd = [
        cli, "search", "offers",
        "-n",
        "--storage", str(allocated_storage),
        "--raw",
        "--limit", str(limit),
        "--type", pricing_type
    ]
    if api_key:
        cmd.extend(["--api-key", api_key])
    
    # Query base rentable offers
    cmd.append("rentable=true")

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
        if isinstance(data, list):
            return data
        return []
    except subprocess.CalledProcessError as e:
        print(f"[Error] Failed to execute vastai: {e.stderr or e.stdout}", file=sys.stderr)
        return []
    except json.JSONDecodeError as e:
        print(f"[Error] Failed to parse JSON from vastai: {e}", file=sys.stderr)
        return []


def filter_offers(offers: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    filtered = []
    for o in offers:
        # 1. Cloud & Verification
        if args.datacenter_only and not o.get("datacenter"):
            continue
        
        # In raw vast API output, 'verification' can be 'verified', 'unverified', or 'deverified'
        # and 'verified' may be None or bool.
        is_verified = (o.get("verification") == "verified") or (o.get("verified") is True)
        if args.verified_only and not is_verified:
            continue
        if not args.include_external and o.get("external") is True:
            continue
        if args.static_ip and not o.get("static_ip"):
            continue

        # 2. Disk Space & Storage
        disk_space = o.get("disk_space") or 0.0
        if disk_space < args.min_disk_space:
            continue

        # 3. Reliability
        # Reliability is reported as 0.0 - 1.0 (or percentage if > 1)
        rel = o.get("reliability") or 0.0
        rel_pct = rel * 100.0 if rel <= 1.0 else rel
        if rel_pct < args.min_reliability:
            continue

        # 4. Duration
        # duration in vast is in days or seconds depending on schema; vast duration is max rental days
        duration_days = o.get("duration") or 0.0
        duration_hours = duration_days * 24.0
        if duration_hours < args.min_duration_hours:
            continue

        # 5. Software & Versions
        cuda_vers = o.get("cuda_vers") or o.get("cuda_max_good") or 0.0
        if args.min_cuda and cuda_vers < args.min_cuda:
            continue

        if args.min_driver:
            cand_driver = parse_driver_tuple(o.get("driver_version"))
            req_driver = parse_driver_tuple(args.min_driver)
            if cand_driver < req_driver:
                continue

        if args.ubuntu_version:
            os_ver = str(o.get("os_version") or o.get("ubuntu_version") or "")
            if args.ubuntu_version not in os_ver:
                continue

        # 6. Pricing Filters
        total_dph = o.get("dph_total") or 0.0
        if args.max_dph and total_dph > args.max_dph:
            continue

        flops_usd = o.get("flops_per_dphtotal") or o.get("flops_usd") or 0.0
        # $/TFLOPS/hr = total_dph / total_flops
        total_flops = o.get("total_flops") or 0.0
        dph_per_tflop = (total_dph / total_flops) if total_flops > 0 else 9999.0
        if args.max_usd_per_tflops and dph_per_tflop > args.max_usd_per_tflops:
            continue

        net_down_cost_tb = o.get("internet_down_cost_per_tb")
        if net_down_cost_tb is None:
            net_down_cost_tb = (o.get("inet_down_cost") or 0.0) * 1024.0
        if args.max_tb_download_cost and net_down_cost_tb > args.max_tb_download_cost:
            continue

        net_up_cost_tb = o.get("internet_up_cost_per_tb")
        if net_up_cost_tb is None:
            net_up_cost_tb = (o.get("inet_up_cost") or 0.0) * 1024.0
        if args.max_tb_upload_cost and net_up_cost_tb > args.max_tb_upload_cost:
            continue

        # 7. GPU Resources
        num_gpus = o.get("num_gpus") or 0
        if args.gpu_count is not None and num_gpus != args.gpu_count:
            continue
        if num_gpus < args.min_gpus or num_gpus > args.max_gpus:
            continue

        if args.gpu_name:
            target_name = args.gpu_name.lower().replace("_", " ")
            cand_name = (o.get("gpu_name") or "").lower().replace("_", " ")
            if target_name not in cand_name:
                continue

        if total_flops < args.min_tflops:
            continue

        # GPU RAM in GB (sometimes in MB if > 1000)
        gpu_ram = o.get("gpu_ram") or 0.0
        if gpu_ram > 1024:  # vast reports MB sometimes
            gpu_ram /= 1024.0
        if gpu_ram < args.min_gpu_ram:
            continue

        gpu_total_ram = o.get("gpu_total_ram") or (gpu_ram * num_gpus)
        if gpu_total_ram > 1024 * 32:  # MB check
            gpu_total_ram /= 1024.0
        if gpu_total_ram < args.min_gpu_total_ram:
            continue

        mem_bw = o.get("gpu_mem_bw") or 0.0
        if mem_bw < args.min_gpu_mem_bw:
            continue

        pcie_bw = o.get("pcie_bw") or 0.0
        if pcie_bw < args.min_pcie_bw:
            continue

        nvlink_bw = o.get("bw_nvlink") or 0.0
        if nvlink_bw < args.min_nvlink_bw:
            continue

        dlperf = o.get("dlperf") or 0.0
        if dlperf < args.min_dlperf:
            continue

        # 8. Machine Resources
        cpu_cores = o.get("cpu_cores") or 0
        if cpu_cores < args.min_cpu_cores:
            continue

        cpu_ram = o.get("cpu_ram") or 0.0
        if cpu_ram > 1024:  # MB -> GB
            cpu_ram /= 1024.0
        if cpu_ram < args.min_cpu_ram:
            continue

        cpu_ghz = o.get("cpu_ghz") or 0.0
        if cpu_ghz < args.min_cpu_ghz:
            continue

        disk_bw = o.get("disk_bw") or 0.0
        if disk_bw < args.min_disk_bw:
            continue

        inet_up = o.get("inet_up") or 0.0
        if inet_up < args.min_inet_up:
            continue

        inet_down = o.get("inet_down") or 0.0
        if inet_down < args.min_inet_down:
            continue

        direct_ports = o.get("direct_port_count") or 0
        if direct_ports < args.min_direct_ports:
            continue

        filtered.append(o)

    return filtered


def format_offer_row(o: Dict[str, Any], allocated_storage: float) -> Dict[str, Any]:
    """Calculate structured metrics and detailed pricing for an offer."""
    num_gpus = o.get("num_gpus") or 1
    gpu_name = o.get("gpu_name") or "Unknown"
    
    # GPU RAM
    gpu_ram = o.get("gpu_ram") or 0.0
    if gpu_ram > 1024:
        gpu_ram /= 1024.0
    
    total_ram = o.get("gpu_total_ram") or (gpu_ram * num_gpus)
    if total_ram > 1024 * 32:
        total_ram /= 1024.0

    # CPU RAM
    cpu_ram = o.get("cpu_ram") or 0.0
    if cpu_ram > 1024:
        cpu_ram /= 1024.0

    # Pricing components
    dph_total = o.get("dph_total") or 0.0
    gpu_dph = o.get("dph_base") or 0.0
    storage_mo = o.get("storage_cost") or 0.0  # $/GB/month
    storage_hr = o.get("storage_total_cost") or 0.0  # $/hr for allocated storage
    
    down_cost_tb = o.get("internet_down_cost_per_tb")
    if down_cost_tb is None:
        down_cost_tb = (o.get("inet_down_cost") or 0.0) * 1024.0
        
    up_cost_tb = o.get("internet_up_cost_per_tb")
    if up_cost_tb is None:
        up_cost_tb = (o.get("inet_up_cost") or 0.0) * 1024.0

    rel = o.get("reliability") or 0.0
    rel_pct = rel * 100.0 if rel <= 1.0 else rel

    return {
        "id": o.get("id"),
        "machine_id": o.get("machine_id"),
        "gpu": f"{num_gpus}x {gpu_name}",
        "gpu_ram_per": round(gpu_ram, 1),
        "gpu_total_ram": round(total_ram, 1),
        "total_flops": round(o.get("total_flops") or 0.0, 1),
        "dlperf": round(o.get("dlperf") or 0.0, 1),
        "cpu_cores": o.get("cpu_cores") or 0,
        "cpu_ram_gb": round(cpu_ram, 1),
        "cpu_ghz": round(o.get("cpu_ghz") or 0.0, 2),
        "disk_space_gb": round(o.get("disk_space") or 0.0, 1),
        "allocated_disk_gb": allocated_storage,
        "inet_down_mbps": round(o.get("inet_down") or 0.0, 1),
        "inet_up_mbps": round(o.get("inet_up") or 0.0, 1),
        "direct_ports": o.get("direct_port_count") or 0,
        "cuda_max": o.get("cuda_max_good") or o.get("cuda_vers") or "N/A",
        "driver_ver": o.get("driver_version") or "N/A",
        "os_ver": o.get("os_version") or o.get("ubuntu_version") or "N/A",
        "reliability_pct": round(rel_pct, 1),
        "static_ip": bool(o.get("static_ip")),
        "verified": bool(o.get("verified")),
        "location": o.get("geolocation") or "Unknown",
        # Pricing Breakdown
        "total_cost_hr": round(dph_total, 4),
        "gpu_cost_hr": round(gpu_dph, 4),
        "storage_cost_hr": round(storage_hr, 4),
        "storage_cost_gb_month": round(storage_mo, 3),
        "net_down_cost_tb": round(down_cost_tb, 2),
        "net_up_cost_tb": round(up_cost_tb, 2),
    }


def print_table(rows: List[Dict[str, Any]], template_info: Dict[str, Any]):
    if not rows:
        print("\nNo instances matched your filter criteria.")
        return

    print("\n" + "=" * 120)
    print(f" MATCHING VAST.AI INSTANCES ({len(rows)} found) — Template: {template_info['name']}")
    print("=" * 120)
    print(f"{'ID':<10} {'GPU Model':<22} {'vCPU':<5} {'RAM':<7} {'Disk':<7} {'Rel%':<6} "
          f"{'Total $/hr':<11} {'GPU $/hr':<10} {'Disk $/hr':<10} {'Storage $/mo':<13} {'Net Dn/Up $/TB'}")
    print("-" * 120)

    for r in rows:
        net_pricing = f"${r['net_down_cost_tb']:.1f} / ${r['net_up_cost_tb']:.1f}"
        print(f"{r['id']:<10} {r['gpu']:<22} {r['cpu_cores']:<5} {r['cpu_ram_gb']:<5.0f}G "
              f"{r['disk_space_gb']:<6.0f}G {r['reliability_pct']:<6.1f} "
              f"${r['total_cost_hr']:<10.4f} ${r['gpu_cost_hr']:<9.4f} ${r['storage_cost_hr']:<9.4f} "
              f"${r['storage_cost_gb_month']:<4.2f}/GB/mo   {net_pricing}")

    print("-" * 120)
    print("💡 To rent any instance directly, run:")
    sample_id = rows[0]["id"]
    print(f"   vastai create instance {sample_id} --template_hash {template_info['hash']} --disk {rows[0]['allocated_disk_gb']:.0f}")
    print("=" * 120 + "\n")


def generate_rent_commands(rows: List[Dict[str, Any]], template_info: Dict[str, Any], filepath: Path):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write(f"# Vast.ai Rent Commands — Generated for Template: {template_info['name']}\n")
        f.write(f"# Template Hash: {template_info['hash']}\n\n")
        for r in rows:
            f.write(f"# ID {r['id']} | {r['gpu']} | Rel: {r['reliability_pct']}% | Total: ${r['total_cost_hr']:.4f}/hr "
                    f"(GPU: ${r['gpu_cost_hr']:.4f}, Disk: ${r['storage_cost_hr']:.4f}, Net: ${r['net_down_cost_tb']:.1f}/TB)\n")
            f.write(f"vastai create instance {r['id']} --template_hash {template_info['hash']} --disk {r['allocated_disk_gb']:.0f}\n\n")
    print(f"✓ Saved {len(rows)} ready-to-run rent commands to {filepath}")


def export_json(rows: List[Dict[str, Any]], template_info: Dict[str, Any], filepath: Path):
    payload = {
        "template": template_info,
        "count": len(rows),
        "instances": rows
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"✓ Exported {len(rows)} instances to JSON: {filepath}")


def export_csv(rows: List[Dict[str, Any]], filepath: Path):
    if not rows:
        return
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"✓ Exported {len(rows)} instances to CSV: {filepath}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Search, filter, and inspect rentable Vast.ai GPU instances with PyTorch templates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Template Options
    t_group = p.add_argument_group("PyTorch Template Options")
    t_group.add_argument("--template", choices=["vast", "devel", "runtime"], default="vast",
                         help="PyTorch template preset: 'vast' (Official Vast PyTorch), 'devel' (cuDNN devel), 'runtime' (cuDNN runtime)")
    t_group.add_argument("--template-hash", type=str, default=None,
                         help="Custom Vast.ai template hash (overrides preset)")

    # Filter Options - Storage & Disk
    d_group = p.add_argument_group("Disk Space & Container Size")
    d_group.add_argument("--container-size", "--disk", type=float, default=157.59,
                         help="Allocated container/storage size in GB for pricing & launch")
    d_group.add_argument("--min-disk-space", type=float, default=157.59,
                         help="Minimum available host disk space in GB")

    # Cloud & Machine Verification
    c_group = p.add_argument_group("Cloud & Machine Options")
    c_group.add_argument("--datacenter-only", "--secure-cloud", action="store_true",
                         help="Show secure cloud / datacenter instances only")
    c_group.add_argument("--verified-only", action="store_true", default=False,
                         help="Show only verified machines")
    c_group.add_argument("--include-external", action="store_true",
                         help="Include external offers")
    c_group.add_argument("--static-ip", action="store_true",
                         help="Require static/stable IP address")

    # Reliability & Duration
    r_group = p.add_argument_group("Reliability & Duration")
    r_group.add_argument("--min-reliability", type=float, default=85.34,
                         help="Minimum host reliability percentage (e.g. 85.34)")
    r_group.add_argument("--min-duration-hours", type=float, default=1.0,
                         help="Minimum rental duration in hours (e.g. 1h)")

    # Software & Driver Versions
    s_group = p.add_argument_group("Software & Driver Versions")
    s_group.add_argument("--min-cuda", type=float, default=11.4,
                         help="Minimum supported CUDA version (e.g. 11.4, 12.1, 12.4)")
    s_group.add_argument("--min-driver", type=str, default=None,
                         help="Minimum Nvidia driver version (e.g. 535.86.05, 592.72.22)")
    s_group.add_argument("--ubuntu-version", type=str, default=None,
                         help="Ubuntu version filter (e.g. '22.04', '24.04')")

    # Pricing Limits
    pr_group = p.add_argument_group("Pricing Limits")
    pr_group.add_argument("--pricing-type", choices=["on-demand", "bid", "reserved"], default="on-demand",
                          help="Pricing contract type")
    pr_group.add_argument("--max-dph", type=float, default=128.00,
                          help="Maximum total rental $/hour")
    pr_group.add_argument("--max-usd-per-tflops", type=float, default=20.00,
                          help="Maximum $/TFLOPS/hour")
    pr_group.add_argument("--max-tb-upload-cost", type=float, default=100.00,
                          help="Maximum upload bandwidth cost in $/TB")
    pr_group.add_argument("--max-tb-download-cost", type=float, default=100.00,
                          help="Maximum download bandwidth cost in $/TB")

    # GPU Resources
    gpu_group = p.add_argument_group("GPU Resources")
    gpu_group.add_argument("--gpu-name", type=str, default=None,
                           help="Filter by GPU model name substring (e.g. 'RTX 3090', 'RTX 4090', 'RTX 5090', 'A4000', 'A100')")
    gpu_group.add_argument("--gpu-count", type=int, default=None,
                           help="Exact number of GPUs required")
    gpu_group.add_argument("--min-gpus", type=int, default=1,
                           help="Minimum GPU count")
    gpu_group.add_argument("--max-gpus", type=int, default=64,
                           help="Maximum GPU count")
    gpu_group.add_argument("--min-tflops", type=float, default=2.0,
                           help="Minimum total TFLOPs")
    gpu_group.add_argument("--min-gpu-ram", type=float, default=0.0,
                           help="Minimum per-GPU VRAM in GB (e.g. 21.86)")
    gpu_group.add_argument("--min-gpu-total-ram", type=float, default=0.0,
                           help="Minimum total GPU VRAM across all GPUs in GB")
    gpu_group.add_argument("--min-gpu-mem-bw", type=float, default=10.0,
                           help="Minimum GPU RAM bandwidth in GB/s")
    gpu_group.add_argument("--min-pcie-bw", type=float, default=1.0,
                           help="Minimum PCIe bandwidth in GB/s")
    gpu_group.add_argument("--min-nvlink-bw", type=float, default=0.0,
                           help="Minimum NVLink bandwidth in GB/s")
    gpu_group.add_argument("--min-dlperf", type=float, default=1.06,
                           help="Minimum Deep Learning Performance score")

    # Host & CPU Resources
    cpu_group = p.add_argument_group("Host & Machine Resources")
    cpu_group.add_argument("--min-cpu-cores", type=int, default=4,
                           help="Minimum virtual CPU cores")
    cpu_group.add_argument("--min-cpu-ram", type=float, default=16.0,
                           help="Minimum host CPU RAM in GB")
    cpu_group.add_argument("--min-cpu-ghz", type=float, default=0.128,
                           help="Minimum CPU clock speed in GHz")
    cpu_group.add_argument("--min-disk-bw", type=float, default=1.0,
                           help="Minimum disk read bandwidth in MB/s")
    cpu_group.add_argument("--min-inet-up", type=float, default=1.0,
                           help="Minimum internet upload speed in Mbps")
    cpu_group.add_argument("--min-inet-down", type=float, default=1.0,
                           help="Minimum internet download speed in Mbps")
    cpu_group.add_argument("--min-direct-ports", type=int, default=0,
                           help="Minimum open direct TCP/UDP router ports")

    # General & Output Controls
    out_group = p.add_argument_group("Output & Export Options")
    out_group.add_argument("--limit", type=int, default=1000,
                           help="Max offers to fetch from Vast API")
    out_group.add_argument("--sort-by", choices=["price", "reliability", "dlperf", "gpu_ram", "tflops"],
                           default="price", help="Sort order of displayed results")
    out_group.add_argument("--export-json", type=str, default=None,
                           help="Path to export matching offers as JSON")
    out_group.add_argument("--export-csv", type=str, default=None,
                           help="Path to export matching offers as CSV")
    out_group.add_argument("--export-rent-script", type=str, default=None,
                           help="Path to save executable bash script with rent commands")
    out_group.add_argument("--api-key", type=str, default=None,
                           help="Explicit Vast.ai API key")

    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    cli = get_vast_cli()

    # Determine template
    if args.template_hash:
        template_info = {
            "name": "Custom Template",
            "hash": args.template_hash,
            "image": "custom"
        }
    else:
        template_info = PYTORCH_TEMPLATES[args.template]

    print(f"\n🔍 Querying Vast.ai marketplace...")
    print(f"   Storage context: {args.container_size:.2f} GB | Template: {template_info['name']}")

    raw_offers = query_offers(
        cli=cli,
        allocated_storage=args.container_size,
        pricing_type=args.pricing_type,
        limit=args.limit,
        api_key=args.api_key
    )

    if not raw_offers:
        print("[!] No offers returned from Vast.ai API.")
        sys.exit(1)

    print(f"   Fetched {len(raw_offers)} raw offers. Applying filters...")
    filtered_offers = filter_offers(raw_offers, args)
    formatted_rows = [format_offer_row(o, args.container_size) for o in filtered_offers]

    # Sorting
    if args.sort_by == "price":
        formatted_rows.sort(key=lambda x: x["total_cost_hr"])
    elif args.sort_by == "reliability":
        formatted_rows.sort(key=lambda x: x["reliability_pct"], reverse=True)
    elif args.sort_by == "dlperf":
        formatted_rows.sort(key=lambda x: x["dlperf"], reverse=True)
    elif args.sort_by == "gpu_ram":
        formatted_rows.sort(key=lambda x: x["gpu_total_ram"], reverse=True)
    elif args.sort_by == "tflops":
        formatted_rows.sort(key=lambda x: x["total_flops"], reverse=True)

    print_table(formatted_rows, template_info)

    # Exports
    if args.export_json:
        export_json(formatted_rows, template_info, Path(args.export_json))
    if args.export_csv:
        export_csv(formatted_rows, Path(args.export_csv))
    if args.export_rent_script:
        generate_rent_commands(formatted_rows, template_info, Path(args.export_rent_script))


if __name__ == "__main__":
    main()
