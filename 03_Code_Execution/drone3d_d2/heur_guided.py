"""Ablation of problem-specific guidance on the library ABC (heur_keys.abc, random keys + fleet decoder).

Two guides, each from the structure of the problem (Proposition 1 / Remark 1 of the manuscript), switchable independently:
  seed   : the population contains the unlimited-fleet solution (energy-optimal without the fleet limit when every trip meets T; obtained by a
           short annealing with k = n drones), packed longest-first onto the k drones and written back as random keys (plus four noisy copies).
           The evaluations of that annealing are charged to the budget.
  repair : inside the decoder, if some drone cannot meet T, requests are moved from the most violating drone to another drone (best move by
           total violation, at most MAXREP moves). Every tried move is charged as one evaluation.
Arms: base (heur_keys.abc), seed, repair, seed+repair. Equal total budget of decoder calls."""
from __future__ import annotations
import numpy as np
from .heur import anneal, _pack
from .heur_keys import FleetProblem, PENALTY, abc, init_population

MAXREP = 15


class FleetProblemG(FleetProblem):
    def __init__(self, *a, repair=False, **kw):
        super().__init__(*a, **kw)
        self.repair = repair

    def _viol(self, seq):
        if not seq:
            return 0.0, 0.0
        fr = self._front(seq)
        if not fr:
            return INF_V, 0.0
        ok = [E for E, t in fr if t <= self.T + 1e-9]
        if ok:
            return 0.0, min(ok)
        return min(t for E, t in fr) - self.T, 0.0

    def decode(self, x):
        if not self.repair:
            return super().decode(x)
        n, k = self.n, self.k
        order = np.argsort(x[:n], kind="stable")
        rank = {int(r): i for i, r in enumerate(order)}
        drone = np.minimum((np.asarray(x[n:]) * k).astype(int), k - 1)
        seqs = [tuple(int(r) for r in order if drone[r] == d) for d in range(k)]
        info = [self._viol(s) for s in seqs]
        for _ in range(MAXREP):
            tot = sum(v for v, _ in info)
            if tot <= 0 or tot >= INF_V:
                break
            d = max(range(k), key=lambda i: info[i][0])
            best = None
            for r in seqs[d]:
                s1 = tuple(q for q in seqs[d] if q != r)
                v1 = self._viol(s1)
                for d2 in range(k):
                    if d2 == d:
                        continue
                    s2 = list(seqs[d2]); pos = 0
                    while pos < len(s2) and rank[s2[pos]] < rank[r]:
                        pos += 1
                    s2.insert(pos, r); s2 = tuple(s2)
                    v2 = self._viol(s2)
                    self.evals += 1
                    nt = tot - info[d][0] - info[d2][0] + v1[0] + v2[0]
                    if best is None or nt < best[0]:
                        best = (nt, d, d2, s1, s2, v1, v2)
            if best is None or best[0] >= tot - 1e-9:
                break
            _, d, d2, s1, s2, v1, v2 = best
            seqs[d], seqs[d2], info[d], info[d2] = s1, s2, v1, v2
        viol = sum(v for v, _ in info)
        if viol >= INF_V:
            return PENALTY * 10, None
        if viol > 0:
            return PENALTY + viol, None
        return sum(e for _, e in info), None


INF_V = 1.0e12


def seed_keys(prob, k, T, seed, iters=3000):
    """Unlimited-fleet solution -> packed on k drones -> random keys. Returns (keys, evaluations charged)."""
    ev_ = prob.ev
    n = prob.n
    before = len(ev_.cache)
    r = anneal(ev_.leg, ev_.reqs, ev_.dt, ev_.tm, n, T, iters=iters, seed=seed + 777, evaluator=ev_, phase1_iters=3 * iters)
    charged = 4 * iters   # upper bound: iters + phase 1 (3 x iters)
    if r is None:
        return None, charged
    trips = [list(t) for plan in r[2] for t in plan]
    plans = _pack(ev_, trips, k)
    reqs_order = []
    assign = np.zeros(n)
    for d, plan in enumerate(plans):
        for trip in plan:
            for e in trip:
                if e[0] not in reqs_order:
                    reqs_order.append(e[0]); assign[e[0]] = (d + 0.5) / k
    x = np.zeros(2 * n)
    for i, rr in enumerate(reqs_order):
        x[rr] = (i + 0.5) / n
    x[n:] = assign
    return x, charged


def guided_abc(leg, reqs, dt, tm, k, T, budget, seed, use_seed, use_repair, N=30):
    prob = FleetProblemG(leg, reqs, dt, tm, k, T, repair=use_repair)
    pop0 = None
    if use_seed:
        x, charged = seed_keys(prob, k, T, seed)
        budget = max(budget - charged, 1000)
        if x is not None:
            pop0 = init_population(prob.D, N, seed)
            rng = np.random.default_rng(seed + 5)
            pop0[0] = x
            for j in range(1, 5):
                pop0[j] = np.clip(x + rng.normal(0, 0.05, x.shape), 0.0, 1.0)
    res = abc(prob, budget, seed, N=N, pop0=pop0)
    return res.best if res.best < PENALTY else None
