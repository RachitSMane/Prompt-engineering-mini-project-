"""In-process thermal guard for long local-inference loops (Quadro M2200, 4 GB VRAM laptop GPU).

Call ``GpuGuard.wait_until_safe()`` before every model call. It reads the GPU temperature with ``nvidia-smi``:
  * >= ``pause_c``  : sleeps (checking every 20 s) until the GPU has cooled to ``resume_c``;
  * >= ``abort_c``  : raises ``GpuTooHot`` immediately so the batch stops cleanly (stored results are append-only,
                      so at most the in-flight item is lost and a re-run resumes);
  * temperature unreadable (no NVIDIA driver, CPU-only machine): does nothing;
  * free system RAM < ``min_free_ram_mb`` (default 2 GB): raises ``LowMemory`` (a ``GpuTooHot`` subclass) so the batch
    stops cleanly instead of being killed mid-call.
Every reading is appended to ``logs/gpu_guard.csv``.
"""

from __future__ import annotations

import subprocess
import time
from datetime import datetime
from pathlib import Path

NO_WINDOW = 0x08000000 if hasattr(subprocess, "CREATE_NO_WINDOW") else 0


class GpuTooHot(RuntimeError):
    pass


class LowMemory(GpuTooHot):
    """Free system RAM fell below the floor; stop cleanly before the next model call (subclass so callers stop the same way)."""


def free_ram_mb() -> int | None:
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return None
        return int(m.ullAvailPhys // (1024 * 1024))
    except Exception:  # non-Windows or no ctypes access
        return None


def read_gpu_temp() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, creationflags=NO_WINDOW).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


class GpuGuard:
    def __init__(self, log_path: Path | None = None, pause_c: int = 88, resume_c: int = 75, abort_c: int = 94, max_wait_s: int = 1800,
                 min_free_ram_mb: int = 2048):
        self.log_path = log_path
        self.min_free_ram_mb = min_free_ram_mb
        self.pause_c, self.resume_c, self.abort_c, self.max_wait_s = pause_c, resume_c, abort_c, max_wait_s
        self.max_seen: int | None = None

    def _log(self, temp: int | None, note: str) -> None:
        if temp is not None:
            self.max_seen = max(self.max_seen or temp, temp)
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.log_path.exists()
        with self.log_path.open("a", encoding="utf-8") as fh:
            if new:
                fh.write("time,temp_c,note\n")
            fh.write(f"{datetime.now().isoformat(timespec='seconds')},{'' if temp is None else temp},{note}\n")

    def wait_until_safe(self) -> int | None:
        ram = free_ram_mb()
        if ram is not None and ram < self.min_free_ram_mb:
            self._log(read_gpu_temp(), f"low_memory_{ram}MB")
            raise LowMemory(f"only {ram} MB of RAM free (floor {self.min_free_ram_mb} MB); stopping before the next model call")
        temp = read_gpu_temp()
        self._log(temp, f"check ram_free_mb={ram}")
        if temp is None:
            return None
        if temp >= self.abort_c:
            raise GpuTooHot(f"GPU at {temp} °C (abort threshold {self.abort_c} °C); stopping before the next model call")
        if temp < self.pause_c:
            return temp
        print(f"  [gpu-guard] GPU {temp} °C >= {self.pause_c} °C: pausing until <= {self.resume_c} °C", flush=True)
        waited = 0
        while True:  # hysteresis: once paused, wait for resume_c, not just below pause_c
            time.sleep(20)
            waited += 20
            temp = read_gpu_temp()
            self._log(temp, "cooling")
            if temp is not None and temp >= self.abort_c:
                raise GpuTooHot(f"GPU at {temp} °C while cooling; stopping")
            if temp is not None and temp <= self.resume_c:
                break
            if waited >= self.max_wait_s:
                raise GpuTooHot(f"GPU did not cool below {self.resume_c} °C within {self.max_wait_s} s")
        return temp
