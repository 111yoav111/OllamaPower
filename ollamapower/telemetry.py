"""Hardware facts and live telemetry, for the hub and for agents.

GPUs are found best-effort, in this order:

* **NVIDIA** through NVML (``nvidia-ml-py``): name, VRAM, utilisation,
  temperature, driver and CUDA version.
* **AMD on Linux** through sysfs (``/sys/class/drm/card*/device``): VRAM,
  busy %, temperature. No extra tools needed.
* **Apple Silicon**: unified memory; about 75 % of RAM is usable by Metal,
  reported as ``vram_total`` with ``unified: true``. No utilisation figure.

Anything that fails is simply left out; the CPU is always reported. The link
the agent uses to reach the hub is classified as wired, Wi-Fi or unknown,
because split mode over Wi-Fi is slow and the owner must be warned.
"""

from __future__ import annotations

import contextlib
import logging
import os
import platform
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import psutil

from .llamacpp.variants import native_machine

log = logging.getLogger("ollamapower.telemetry")

APPLE_GPU_SHARE = 0.75  # Metal's recommended working set is ~75% of unified memory
SYSFS_DRM = Path("/sys/class/drm")
SYSFS_NET = Path("/sys/class/net")


# ----- NVIDIA ----------------------------------------------------------------------------


class _Nvml:
    """Lazily initialised NVML; ``available`` is False when there is no NVIDIA driver."""

    def __init__(self) -> None:
        self.available = False
        self.mod: Any = None
        try:
            import pynvml

            pynvml.nvmlInit()
        except Exception:  # noqa: BLE001 - no library, no driver, no GPU: all mean "none"
            return
        self.mod = pynvml
        self.available = True

    def _text(self, value: Any) -> str:
        return value.decode() if isinstance(value, bytes) else str(value)

    def static(self) -> list[dict[str, Any]]:
        n = self.mod
        gpus = []
        driver = cuda = ""
        with contextlib.suppress(Exception):
            driver = self._text(n.nvmlSystemGetDriverVersion())
        with contextlib.suppress(Exception):
            v = n.nvmlSystemGetCudaDriverVersion()
            cuda = f"{v // 1000}.{(v % 1000) // 10}"
        for i in range(n.nvmlDeviceGetCount()):
            h = n.nvmlDeviceGetHandleByIndex(i)
            mem = n.nvmlDeviceGetMemoryInfo(h)
            gpus.append(
                {
                    "index": i,
                    "name": self._text(n.nvmlDeviceGetName(h)),
                    "vendor": "nvidia",
                    "backend": "cuda",
                    "vram_total": int(mem.total),
                    "driver": driver,
                    "cuda": cuda,
                }
            )
        return gpus

    def sample(self) -> list[dict[str, Any]]:
        n = self.mod
        out = []
        for i in range(n.nvmlDeviceGetCount()):
            h = n.nvmlDeviceGetHandleByIndex(i)
            mem = n.nvmlDeviceGetMemoryInfo(h)
            entry: dict[str, Any] = {"index": i, "vram_total": int(mem.total), "vram_used": int(mem.used)}
            with contextlib.suppress(Exception):
                entry["util"] = int(n.nvmlDeviceGetUtilizationRates(h).gpu)
            with contextlib.suppress(Exception):
                entry["temp"] = int(n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU))
            out.append(entry)
        return out


# ----- AMD (Linux sysfs) -------------------------------------------------------------------


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _amd_cards(root: Path = SYSFS_DRM) -> list[Path]:
    cards = []
    for card in sorted(root.glob("card[0-9]*")):
        dev = card / "device"
        try:
            vendor = (dev / "vendor").read_text().strip()
        except OSError:
            continue
        if vendor == "0x1002" and (dev / "mem_info_vram_total").exists():
            cards.append(dev)
    return cards


def amd_static(root: Path = SYSFS_DRM) -> list[dict[str, Any]]:
    gpus = []
    for i, dev in enumerate(_amd_cards(root)):
        name = "AMD GPU"
        with contextlib.suppress(OSError):
            name = (dev / "product_name").read_text().strip() or name
        gpus.append(
            {
                "index": i,
                "name": name,
                "vendor": "amd",
                "backend": "rocm",
                "vram_total": _read_int(dev / "mem_info_vram_total") or 0,
            }
        )
    return gpus


def amd_sample(root: Path = SYSFS_DRM) -> list[dict[str, Any]]:
    out = []
    for i, dev in enumerate(_amd_cards(root)):
        entry: dict[str, Any] = {
            "index": i,
            "vram_total": _read_int(dev / "mem_info_vram_total") or 0,
            "vram_used": _read_int(dev / "mem_info_vram_used") or 0,
        }
        busy = _read_int(dev / "gpu_busy_percent")
        if busy is not None:
            entry["util"] = busy
        for temp in sorted(dev.glob("hwmon/hwmon*/temp1_input"))[:1]:
            milli = _read_int(temp)
            if milli is not None:
                entry["temp"] = milli // 1000
        out.append(entry)
    return out


# ----- Apple Silicon -----------------------------------------------------------------------


def apple_static() -> list[dict[str, Any]]:
    if sys.platform != "darwin" or native_machine() != "arm64":  # also under Rosetta
        return []
    return [
        {
            "index": 0,
            "name": f"{cpu_model()} GPU",
            "vendor": "apple",
            "backend": "metal",
            "vram_total": int(psutil.virtual_memory().total * APPLE_GPU_SHARE),
            "unified": True,
        }
    ]


# ----- CPU ---------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def cpu_model() -> str:
    try:
        if sys.platform == "darwin":
            return (
                subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=2, check=False
                ).stdout.strip()
                or platform.processor()
            )
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


# ----- network -----------------------------------------------------------------------------


def interface_for_ip(ip: str) -> str | None:
    for name, addrs in psutil.net_if_addrs().items():
        if any(a.address.split("%")[0] == ip for a in addrs):
            return name
    return None


@lru_cache(maxsize=1)
def _darwin_wifi_devices() -> frozenset[str]:
    """macOS: devices whose hardware port is Wi-Fi (``en0`` is Wi-Fi on most laptops, wired on iMacs)."""
    try:
        out = subprocess.run(
            ["networksetup", "-listallhardwareports"], capture_output=True, text=True, timeout=3, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    wifi, port = set(), ""
    for line in out.splitlines():
        if line.startswith("Hardware Port:"):
            port = line.split(":", 1)[1].strip()
        elif line.startswith("Device:") and port.lower() in ("wi-fi", "airport"):
            wifi.add(line.split(":", 1)[1].strip())
    return frozenset(wifi)


def link_kind(interface: str, platform_name: str = sys.platform, sysfs: Path = SYSFS_NET) -> str:
    """``wired``, ``wifi`` or ``unknown`` for a network interface name."""
    low = interface.lower()
    if platform_name.startswith("linux"):
        if (sysfs / interface / "wireless").exists() or (sysfs / interface / "phy80211").exists():
            return "wifi"
        if low.startswith(("wl", "wlan")):
            return "wifi"
        if low.startswith(("eth", "en", "em")):
            return "wired"
        return "unknown"
    if platform_name == "darwin":
        if interface in _darwin_wifi_devices():
            return "wifi"
        return "wired" if low.startswith("en") and _darwin_wifi_devices() else "unknown"
    if any(word in low for word in ("wi-fi", "wifi", "wlan", "wireless")):
        return "wifi"
    if "ethernet" in low:
        return "wired"
    return "unknown"


def network_info(local_ip: str | None) -> dict[str, Any]:
    """The interface used for ``local_ip`` (the hub connection), its kind and speed."""
    if not local_ip:
        return {"kind": "unknown"}
    name = interface_for_ip(local_ip)
    if name is None:
        return {"kind": "loopback" if local_ip.startswith(("127.", "::1")) else "unknown", "ip": local_ip}
    info: dict[str, Any] = {"interface": name, "ip": local_ip, "kind": link_kind(name)}
    if local_ip.startswith(("127.", "::1")):
        info["kind"] = "loopback"
    with contextlib.suppress(Exception):
        speed = psutil.net_if_stats()[name].speed
        if speed:
            info["speed_mbps"] = int(speed)
    return info


# ----- the probe -----------------------------------------------------------------------------


class Telemetry:
    """Static facts once, live samples on demand. Cheap enough to sample every 2 s."""

    def __init__(self) -> None:
        self._nvml = _Nvml()
        psutil.cpu_percent(interval=None)  # prime: the first reading is meaningless

    def gpus(self) -> list[dict[str, Any]]:
        if self._nvml.available:
            try:
                return self._nvml.static()
            except Exception as exc:  # noqa: BLE001 - a flaky driver must not kill the agent
                log.warning("NVML failed: %s", type(exc).__name__)
        return amd_static() or apple_static()

    def static(self) -> dict[str, Any]:
        gpus = self.gpus()
        backends = sorted({g["backend"] for g in gpus}) + ["cpu"]
        return {
            "os": platform.system(),
            "os_release": platform.release(),
            "arch": platform.machine(),
            "hostname": platform.node(),
            "cpu": {
                "model": cpu_model(),
                "cores": psutil.cpu_count(logical=False) or 0,
                "threads": psutil.cpu_count() or 0,
            },
            "ram_total": int(psutil.virtual_memory().total),
            "gpus": gpus,
            "backends": backends,
        }

    def sample(self) -> dict[str, Any]:
        vm = psutil.virtual_memory()
        gpus: list[dict[str, Any]] = []
        if self._nvml.available:
            with contextlib.suppress(Exception):
                gpus = self._nvml.sample()
        if not gpus:
            gpus = amd_sample()
        if not gpus:
            # Unified memory: what the GPU could still take is what the system has available.
            gpus = [
                {
                    "index": g["index"],
                    "vram_total": g["vram_total"],
                    "vram_used": g["vram_total"] - min(g["vram_total"], int(vm.available)),
                }
                for g in apple_static()
            ]
        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_total": int(vm.total),
            "ram_used": int(vm.total - vm.available),
            "gpus": gpus,
            "load": os.getloadavg()[0] if hasattr(os, "getloadavg") else None,
        }
