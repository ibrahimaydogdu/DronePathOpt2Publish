"""Component ablation of the simulated annealing of heur.py (same neighbourhood, cost model, iterations).
Components: unl = restart 0 from the unlimited-fleet solution packed onto k drones (Proposition 1c);
            p1  = phase 1 (makespan repair) before the energy search;
            hard = makespan bound a HARD constraint in phase 2 (violating neighbours rejected) instead of the penalty PEN*(makespan - T).
Without hard, phase 2 accepts on E + PEN*violation and the best time-feasible solution seen is returned.
Without unl, the extra restart keeps the number of annealing runs equal (restarts + 1)."""
from __future__ import annotations
import math
import random
from .heur import _Eval, _cost, _neighbor, initial, _phase1, _pack, anneal


def anneal_abl(leg, reqs, dt, tm, k, T, iters, seed, evaluator, p1, hard, start=None, alpha=0.9995):
    rng = random.Random(seed)
    ev_ = evaluator
    n = len(reqs)
    cur = start if start is not None else initial(ev_, n, k)
    if cur is None:
        return None
    if p1:
        cur = _phase1(ev_, cur, T, k, reqs, n, rng, 3 * iters)
        if cur is None:
            return None
    cc = _cost(ev_, cur, T)
    if cc is None:
        return None
    best = (cc[1], cc[2], cur) if cc[2] <= T + 1e-9 else None
    temp = max(1.0, 0.02 * cc[1])
    for _ in range(iters):
        sol = _neighbor(cur, rng, reqs, n, k)
        if sol is None:
            continue
        c = _cost(ev_, sol, T)
        if c is None:
            continue
        if hard and c[2] > T + 1e-9:
            continue
        if c[0] <= cc[0] or rng.random() < math.exp(-(c[0] - cc[0]) / temp):
            cur, cc = sol, c
            if c[2] <= T + 1e-9 and (best is None or c[1] < best[0]):
                best = (c[1], c[2], sol)
        temp = max(temp * alpha, 1e-3)
    return best


def anneal_multi_abl(leg, reqs, dt, tm, k, T, unl, p1, hard, restarts=3, iters=20000, seed0=1001):
    ev_ = _Eval(leg, reqs, dt, tm)
    n = len(reqs)
    bst = None
    starts = []
    if unl:
        inf = anneal(leg, reqs, dt, tm, n, T, iters=iters, seed=seed0 + 777, evaluator=ev_, phase1_iters=3 * iters)
        if inf is not None:
            trips = [list(t) for plan in inf[2] for t in plan]
            starts.append(_pack(ev_, trips, k))
            if k >= len(trips) and inf[1] <= T + 1e-9:
                bst = (inf[0], inf[1], inf[2])
        starts += [None] * max(restarts - len(starts), 1)
    else:
        starts = [None] * (restarts + 1)
    for j, st in enumerate(starts):
        r = anneal_abl(leg, reqs, dt, tm, k, T, iters, seed0 + 1000 * j, ev_, p1, hard, start=st)
        if r and (bst is None or r[0] < bst[0]):
            bst = r
    return bst
