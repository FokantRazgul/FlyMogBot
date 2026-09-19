"""Hardware and environment probe.

This is the first command the operator runs on their own machine. Its JSON
output becomes the "Hardware" section of docs/M0_REPORT.md, because the
development environment has no accelerator and cannot measure the real one.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from typing import Any

import psutil


def _nvidia_smi() -> dict[str, Any]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return {"present": False, "detail": "nvidia-smi not on PATH"}
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"present": True, "detail": f"nvidia-smi failed: {exc}"}
    if out.returncode != 0:
        return {
            "present": True,
            "detail": f"nvidia-smi exited {out.returncode}: {out.stderr.strip()}",
        }
    gpus = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    return {"present": True, "gpus": gpus}


def _apple_gpu() -> dict[str, Any]:
    """On macOS, unified memory means there is no separate VRAM figure."""
    if platform.system() != "Darwin":
        return {"present": False, "detail": "not macOS"}
    exe = shutil.which("system_profiler")
    if exe is None:
        return {"present": False, "detail": "system_profiler not on PATH"}
    try:
        out = subprocess.run(
            [exe, "SPDisplaysDataType"], capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"present": True, "detail": f"system_profiler failed: {exc}"}
    wanted = ("Chipset Model", "Total Number of Cores")
    lines = [ln.strip() for ln in out.stdout.splitlines() if any(marker in ln for marker in wanted)]
    return {"present": True, "details": lines}


def _torch_report() -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        return {"installed": False, "detail": str(exc)}

    mps_backend = getattr(torch.backends, "mps", None)
    report: dict[str, Any] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(mps_backend is not None and mps_backend.is_available()),
        "num_threads": torch.get_num_threads(),
    }
    if torch.cuda.is_available():
        devices = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            devices.append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / 1024**3, 2),
                    "capability": f"{props.major}.{props.minor}",
                }
            )
        report["cuda_devices"] = devices
        report["cuda_version"] = torch.version.cuda
    return report


def probe_environment() -> dict[str, Any]:
    """Collect everything needed to interpret a benchmark on this machine."""
    vm = psutil.virtual_memory()
    disk = psutil.disk_usage(os.getcwd())
    from flymog.sim.backends import available_devices, resolve_backend_kind, select_device

    torch_report = _torch_report()
    selected = None
    resolved_backend = None
    if torch_report.get("installed"):
        device = select_device()
        selected = device.type
        resolved_backend = resolve_backend_kind("auto", device)

    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
        },
        "cpu": {
            "logical_cores": psutil.cpu_count(logical=True),
            "physical_cores": psutil.cpu_count(logical=False),
        },
        "memory": {
            "total_gb": round(vm.total / 1024**3, 2),
            "available_gb": round(vm.available / 1024**3, 2),
        },
        "disk": {
            "total_gb": round(disk.total / 1024**3, 2),
            "free_gb": round(disk.free / 1024**3, 2),
        },
        "nvidia": _nvidia_smi(),
        "apple_gpu": _apple_gpu(),
        "torch": torch_report,
        "devices_available": available_devices() if torch_report.get("installed") else None,
        "selected_device": selected,
        "auto_backend": resolved_backend,
    }
