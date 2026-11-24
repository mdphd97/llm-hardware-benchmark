#!/usr/bin/env python3
"""
Host Power & Thermal Profiler.
Samples GPU power/temperature via nvidia-smi and optionally CPU Intel RAPL.
"""

import subprocess
import time
from typing import Any, Dict, List, Optional


def parse_nvidia_smi_power(raw_csv: str) -> Dict[str, Any]:
    """Parse 'power.draw, temperature.gpu, memory.used' CSV output."""
    parts = [p.strip() for p in raw_csv.split(",")]
    return {
        "power_watts": float(parts[0]) if parts[0] not in ("", "[N/A]") else 0.0,
        "temp_c": int(float(parts[1])) if len(parts) > 1 and parts[1] not in ("", "[N/A]") else 0,
        "vram_used_mb": int(float(parts[2])) if len(parts) > 2 and parts[2] not in ("", "[N/A]") else 0,
    }


def sample_gpu_metrics() -> Optional[Dict[str, Any]]:
    """Query current NVIDIA GPU power draw, temperature, and VRAM usage."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=power.draw,temperature.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if out:
            return parse_nvidia_smi_power(out.splitlines()[0])
    except Exception:
        pass
    return None


def read_rapl_energy_uj() -> Optional[int]:
    """Read Intel RAPL package-0 energy counter in microjoules if permitted."""
    try:
        with open("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj", "r") as f:
            return int(f.read().strip())
    except Exception:
        return None


class EnergyMonitor:
    """Context manager / sampler tracking GPU power and thermal delta."""

    def __init__(self):
        self.start_time = 0.0
        self.end_time = 0.0
        self.samples: List[Dict[str, Any]] = []
        self.rapl_start: Optional[int] = None
        self.rapl_end: Optional[int] = None

    def start(self):
        self.start_time = time.perf_counter()
        self.rapl_start = read_rapl_energy_uj()
        s = sample_gpu_metrics()
        if s:
            self.samples.append(s)

    def sample(self):
        s = sample_gpu_metrics()
        if s:
            self.samples.append(s)

    def stop(self) -> Dict[str, Any]:
        self.end_time = time.perf_counter()
        self.rapl_end = read_rapl_energy_uj()
        s = sample_gpu_metrics()
        if s:
            self.samples.append(s)

        duration_s = max(self.end_time - self.start_time, 1e-6)
        avg_gpu_watts = (
            round(sum(x["power_watts"] for x in self.samples) / len(self.samples), 2)
            if self.samples
            else 0.0
        )
        peak_gpu_watts = (
            round(max(x["power_watts"] for x in self.samples), 2)
            if self.samples
            else 0.0
        )
        peak_temp_c = (
            max(x["temp_c"] for x in self.samples) if self.samples else 0
        )
        peak_vram_mb = (
            max(x["vram_used_mb"] for x in self.samples) if self.samples else 0
        )
        gpu_energy_j = round(avg_gpu_watts * duration_s, 2)

        cpu_energy_j = None
        if self.rapl_start is not None and self.rapl_end is not None and self.rapl_end >= self.rapl_start:
            cpu_energy_j = round((self.rapl_end - self.rapl_start) / 1e6, 2)

        return {
            "duration_s": round(duration_s, 2),
            "avg_gpu_watts": avg_gpu_watts,
            "peak_gpu_watts": peak_gpu_watts,
            "gpu_energy_joules": gpu_energy_j,
            "cpu_energy_joules": cpu_energy_j,
            "peak_gpu_temp_c": peak_temp_c,
            "peak_vram_mb": peak_vram_mb,
        }
