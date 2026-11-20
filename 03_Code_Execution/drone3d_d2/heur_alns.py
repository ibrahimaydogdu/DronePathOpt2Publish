"""ALNS (ruin-and-recreate) baseline on the native trip-list encoding, same cost model and evaluation budget (calls of _cost) as heur.anneal / heur_pop.
Ruin: q = 2..5 requests, either random or all requests of one random trip plus random others ('trip removal').
Recreate: for each removed request (random order) m candidate insertions (same insertion operators as the SA relocate move: into an existing trip at a random position or as a new trip on a random drone) are evaluated; the best by objective E + PEN*max(0, makespan - T) is kept.
No phase-1 repair: the makespan bound enters through the penalty only. Acceptance: simulated-annealing criterion, geometric cooling. Returns the best time-feasible (E, makespan, solution) or None.
Written for this study; textbook ALNS skeleton (Ropke and Pisinger 2006) without adaptive operator weights."""
from __future__ import annotations
import math
import random
from .analysis import _req_events
from .heur import _Eval, _cost, _remove, _insert_events, initial, PEN


def _recreate(ev_, sol, r, reqs, k, T, rng, m, run):
    evs = _req_events(reqs, r)
    flat = [(d, i) for d in range(k) for i in range(len(sol[d]))]
    best, bc = None, None
    for _ in range(m):
        cand = [[list(t) for t in plan] for plan in sol]
        if flat and rng.random() < 0.6:
            d, i = rng.choice(flat)
            cand[d][i] = _insert_events(cand[d][i], evs, rng)
        else:
            d = rng.randrange(k)
            cand[d].insert(rng.randint(0, len(cand[d])), list(evs))
        c = run.g(cand, T)
        if c is not None and (bc is None or c[0] < bc[0]):
            best, bc = cand, c
    if best is None:    # every candidate infeasible: open a new trip on a random drone
        best = [[list(t) for t in plan] for plan in sol]
        best[rng.randrange(k)].append(list(evs))
    return best


class _Count:
    def __init__(self, ev_, budget):
        self.ev, self.budget, self.used, self.best = ev_, budget, 0, None

    def f(self, sol, T):
        self.used += 1
        c = _cost(self.ev, sol, T)
        if c is not None and c[2] <= T + 1e-9 and (self.best is None or c[1] < self.best[0]):
            self.best = (c[1], c[2], sol)
        return c

    def g(self, sol, T):
        """Evaluation of a PARTIAL solution (some removed requests not yet reinserted): counts budget, never recorded as best."""
        self.used += 1
        return _cost(self.ev, sol, T)

    @property
    def done(self):
        return self.used >= self.budget


def alns(leg, reqs, dt, tm, k, T, budget=80000, seed=1001, m=8, evaluator=None):
    rng = random.Random(seed)
    ev_ = evaluator or _Eval(leg, reqs, dt, tm)
    n = len(reqs)
    run = _Count(ev_, budget)
    cur = initial(ev_, n, k)
    if cur is None:
        return None
    cc = run.f(cur, T)
    if cc is None:
        return None
    temp = max(1.0, 0.02 * cc[1])
    iters_est = max(1, budget // (3 * m + 1))
    alpha = (1e-3 / temp) ** (1.0 / iters_est) if temp > 1e-3 else 1.0
    while not run.done:
        q = rng.randint(2, max(2, min(5, n // 2)))
        sol = [[list(t) for t in plan] for plan in cur]
        if rng.random() < 0.5:
            rem = rng.sample(range(n), q)
        else:
            flat = [t for plan in sol for t in plan]
            rem = list({e[0] for e in rng.choice(flat)}) if flat else []
            while len(rem) < q:
                x = rng.randrange(n)
                if x not in rem:
                    rem.append(x)
            rem = rem[:max(q, 2)]
        for r in rem:
            sol = _remove(sol, r)
        rng.shuffle(rem)
        for r in rem:
            sol = _recreate(ev_, sol, r, reqs, k, T, rng, m, run)
        c = run.f(sol, T)
        if c is None:
            temp = max(temp * alpha, 1e-3)
            continue
        if c[0] <= cc[0] or rng.random() < math.exp(-(c[0] - cc[0]) / temp):
            cur, cc = sol, c
        temp = max(temp * alpha, 1e-3)
    return run.best
