from __future__ import annotations


def self_host_cost_per_1m(gpu_usd_per_hr: float, output_tps: float) -> float:
    if output_tps <= 0:
        return float("inf")
    tokens_per_hr = output_tps * 3600.0
    return gpu_usd_per_hr / (tokens_per_hr / 1_000_000.0)


def crossover_tps(gpu_usd_per_hr: float, managed_usd_per_1m: float) -> float:
    # Solve self_host_cost_per_1m == managed price for tps.
    # gpu/hr / (tps*3600/1e6) = managed  =>  tps = gpu/hr * 1e6 / (managed * 3600)
    if managed_usd_per_1m <= 0:
        return float("inf")
    return gpu_usd_per_hr * 1_000_000.0 / (managed_usd_per_1m * 3600.0)
