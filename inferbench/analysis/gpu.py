from __future__ import annotations

import re
import subprocess
from typing import Callable


def parse_used_mem_mib(nvidia_smi_csv: str) -> float:
    """Parse used GPU memory (MiB) from nvidia-smi output, return GB."""
    for line in nvidia_smi_csv.splitlines():
        m = re.search(r"(\d+)", line)
        if m:
            return int(m.group(1)) / 1024.0  # MiB -> GB
    return 0.0


def make_gpu_mem_fn(index: int = 0) -> Callable[[], float]:
    """Return a fn that queries nvidia-smi for used GB on the given GPU index.

    Integration helper (requires a real GPU host); not unit-tested.
    """
    def gpu_mem() -> float:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={index}", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"], text=True)
        return parse_used_mem_mib(out)
    return gpu_mem
