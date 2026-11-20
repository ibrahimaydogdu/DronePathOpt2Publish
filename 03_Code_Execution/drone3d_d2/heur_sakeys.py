"""Simulated annealing on the random-key encoding of heur_keys.FleetProblem (same decoder/budget as the library ABC/TLBO/GWO).
Separates search paradigm (SA vs population) from encoding (keys vs native trip lists) in the 2x2 factorial.
Neighbour: perturb 1-2 coordinates (50% resample uniformly, 50% Gaussian sigma=0.1), clip to [0,1]. Geometric cooling t0 -> t1 over the budget."""
from __future__ import annotations
import math
import numpy as np
from .heur_keys import Result, Budget, init_population, PENALTY


def sa_keys(problem, limit, seed, N=30, t0=30.0, t1=0.1):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 4]))
    B = Budget(problem, limit)
    x = init_population(problem.D, 1, seed)[0]
    fx = B.f(x)
    alpha = (t1 / t0) ** (1.0 / max(1, limit - 1))
    t = t0
    while not B.done:
        y = x.copy()
        for _ in range(int(rng.integers(1, 3))):
            d = int(rng.integers(problem.D))
            y[d] = rng.random() if rng.random() < 0.5 else np.clip(y[d] + rng.normal(0, 0.1), 0.0, 1.0)
        fy = B.f(y)
        if fy <= fx or rng.random() < math.exp(-(fy - fx) / t):
            x, fx = y, fy
        t *= alpha
    return Result(B.best, B.bx, B.hist, B.used)
