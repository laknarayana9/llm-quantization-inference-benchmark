import math

from inferbench.analysis.cost import self_host_cost_per_1m, crossover_tps


def test_self_host_cost():
    # $2/hr, 100 tok/s -> 360k tok/hr -> $/1M = 2 / 0.36 = 5.555...
    assert math.isclose(self_host_cost_per_1m(2.0, 100.0), 2.0 / 0.36, rel_tol=1e-9)
    assert self_host_cost_per_1m(2.0, 0.0) == float("inf")


def test_crossover_point():
    # self-host beats managed when cost_per_1m < managed price
    tps = crossover_tps(gpu_usd_per_hr=2.0, managed_usd_per_1m=5.0)
    assert math.isclose(self_host_cost_per_1m(2.0, tps), 5.0, rel_tol=1e-9)
