"""CPU thread bounds (interfaces section 4)."""

import os

DEFAULT_THREADS = 4


def bound_threads(cpu_threads=DEFAULT_THREADS, torch_module=None):
    """Pin OMP/MKL and torch to the profile's cpuThreads."""
    count = int(cpu_threads or DEFAULT_THREADS)
    if count < 1:
        raise ValueError(f"cpuThreads must be >= 1, got {cpu_threads!r}")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(count)
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError:
            return count
    torch_module.set_num_threads(count)
    return count
