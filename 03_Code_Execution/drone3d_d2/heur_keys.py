"""Library metaheuristics (ABC, TLBO, GWO) on the fleet problem: random-key search with a fleet decoder.

The three algorithms are copied unchanged from the authors' D1 study (Drone_D1/03_Code_Execution/drone3d/heuristics.py, which ports
lines 297-337 ABC, 544-581 TLBO and 614-643 GWO of OptimizationMethods.vb to a continuous random-key space [0,1]^D). Only the decoder differs:
D1 decodes one drone; here D = 2n keys: keys 0..n-1 order the requests, keys n..2n-1 assign each request to one of k drones (floor(key*k)).
Each drone's requests, in key order, are split into consecutive trips by an exact Pareto dynamic programme over (energy, time) with the
inter-trip turnaround; the drone then takes the least-energy split whose time is <= T. Energy of the solution = sum over drones.
If some drone cannot meet T, the value is PENALTY + sum of the violations (s), as in D1's decoder (PENALTY + L).
Budget = number of decoder calls. The decoder is exact per drone, so these baselines are stronger than the penalty-only ABC/GA of heur_pop.py.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .analysis import _req_events
from .heur import _Eval

INF = float("inf")
PENALTY = 1.0e6


class FleetProblem:
    def __init__(self, leg, reqs, dt, tm, k, T):
        self.ev, self.reqs, self.k, self.T, self.tm = _Eval(leg, reqs, dt, tm), reqs, k, T, tm
        self.n = len(reqs)
        self.D = 2 * self.n
        self.evs = [_req_events(reqs, r) for r in range(self.n)]
        self._pareto = {}
        self.evals = 0

    def _front(self, seq):
        """Pareto list of (E, t) over all splits of the request sequence seq into consecutive trips (t includes turnarounds)."""
        if seq in self._pareto:
            return self._pareto[seq]
        m = len(seq)
        f = [[] for _ in range(m + 1)]
        f[0] = [(0.0, 0.0)]
        for i in range(1, m + 1):
            cand = []
            for j in range(i):
                if not f[j]:
                    continue
                evs = [e for r in seq[j:i] for e in self.evs[r]]
                res = self.ev.trip(evs)
                if res is None:
                    continue
                add = res[1] + (self.tm.tau if j > 0 else 0.0)
                cand.extend((E0 + res[0], t0 + add) for E0, t0 in f[j])
            cand.sort()
            out, best_t = [], INF
            for E, t in cand:
                if t < best_t - 1e-9:
                    out.append((E, t)); best_t = t
            f[i] = out
        self._pareto[seq] = f[m]
        return f[m]

    def decode(self, x):
        n, k = self.n, self.k
        order = np.argsort(x[:n], kind="stable")
        drone = np.minimum((np.asarray(x[n:]) * k).astype(int), k - 1)
        total, viol = 0.0, 0.0
        for d in range(k):
            seq = tuple(int(r) for r in order if drone[r] == d)
            if not seq:
                continue
            fr = self._front(seq)
            if not fr:
                return PENALTY * 10, None
            ok = [E for E, t in fr if t <= self.T + 1e-9]
            if ok:
                total += min(ok)
            else:
                viol += min(t for E, t in fr) - self.T
        if viol > 0:
            return PENALTY + viol, None
        return total, None

    def evaluate(self, x):
        self.evals += 1
        return self.decode(x)[0]


@dataclass
class Result:
    best: float
    x: np.ndarray
    history: list
    evals: int
    extra: dict = None


class Budget:
    def __init__(self, problem, limit):
        self.p, self.limit, self.hist = problem, limit, []
        self.best, self.bx = INF, None
        self.start = problem.evals

    @property
    def used(self):
        return self.p.evals - self.start

    @property
    def done(self):
        return self.used >= self.limit

    def f(self, x):
        v = self.p.evaluate(x)
        if v < self.best:
            self.best, self.bx = v, np.array(x, copy=True)
            self.hist.append((self.used, v))
        return v


def init_population(D, N, seed):
    return np.random.default_rng(np.random.SeedSequence([seed, 0])).random((N, D))


def tlbo(problem, limit, seed, N=30, pop0=None):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    B = Budget(problem, limit)
    X = (pop0 if pop0 is not None else init_population(problem.D, N, seed)).copy()
    N = len(X)
    cost = np.array([B.f(x) for x in X])
    while not B.done:
        teacher = X[int(np.argmin(cost))].copy()
        mean = X.mean(axis=0)
        for i in range(N):
            if B.done:
                break
            tf = round(1 + rng.random())
            xn = np.clip(X[i] + rng.random(problem.D) * (teacher - tf * mean), 0.0, 1.0)
            fn = B.f(xn)
            if fn < cost[i]:
                X[i], cost[i] = xn, fn
            if B.done:
                break
            j = int(rng.integers(N - 1)); j += j >= i
            better = cost[i] < cost[j]
            r = rng.random(problem.D)
            xn = np.clip(X[i] + r * ((X[i] - X[j]) if better else (X[j] - X[i])), 0.0, 1.0)
            fn = B.f(xn)
            if fn < cost[i]:
                X[i], cost[i] = xn, fn
    return Result(B.best, B.bx, B.hist, B.used)


def gwo(problem, limit, seed, N=30, pop0=None, a_start=2.0):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    B = Budget(problem, limit)
    X = (pop0 if pop0 is not None else init_population(problem.D, N, seed)).copy()
    N = len(X)
    cost = np.array([B.f(x) for x in X])
    order = np.argsort(cost)
    leaders = [(float(cost[k]), X[k].copy()) for k in order[:3]]
    while len(leaders) < 3:
        leaders.append(leaders[0])
    while not B.done:
        for i in range(N):
            if B.done:
                break
            a = a_start * (1.0 - B.used / limit)
            xn = np.zeros(problem.D)
            for _, L in leaders:
                A = 2 * a * rng.random(problem.D) - a
                C = 2 * rng.random(problem.D)
                xn += L - A * np.abs(C * L - X[i])
            xn = np.clip(xn / 3.0, 0.0, 1.0)
            fn = B.f(xn)
            X[i], cost[i] = xn, fn
            key = tuple(np.round(xn, 6))
            if any(tuple(np.round(L, 6)) == key for _, L in leaders):
                continue
            for k in range(3):
                if fn < leaders[k][0]:
                    leaders.insert(k, (fn, xn.copy()))
                    leaders.pop()
                    break
    return Result(B.best, B.bx, B.hist, B.used)


def abc(problem, limit, seed, N=30, pop0=None, limit_trials=0):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 3]))
    B = Budget(problem, limit)
    X = (pop0 if pop0 is not None else init_population(problem.D, N, seed)).copy()
    N, D = len(X), problem.D
    cost = np.array([B.f(x) for x in X])
    trials = np.zeros(N, dtype=int)
    lim = limit_trials if limit_trials > 0 else N * D
    scouts = 0

    def neighbour(i):
        k = int(rng.integers(N - 1)); k += k >= i
        v = X[i].copy()
        d = int(rng.integers(D))
        v[d] = np.clip(X[i][d] + (2 * rng.random() - 1) * (X[i][d] - X[k][d]), 0.0, 1.0)
        return v

    while not B.done:
        for i in range(N):
            if B.done:
                break
            v = neighbour(i)
            fv = B.f(v)
            if fv < cost[i]:
                X[i], cost[i], trials[i] = v, fv, 0
            else:
                trials[i] += 1
            if B.done:
                break
            fit = 1.0 / (1.0 + np.where(np.isfinite(cost), np.maximum(cost, 0.0), 1e12))
            k = int(rng.choice(N, p=fit / fit.sum()))
            v = neighbour(k)
            fv = B.f(v)
            if fv < cost[k]:
                X[k], cost[k], trials[k] = v, fv, 0
            else:
                trials[k] += 1
            if B.done:
                break
            if trials[i] > lim:
                v = rng.random(D)
                fv = B.f(v)
                X[i], cost[i], trials[i] = v, fv, 0
                scouts += 1
    return Result(B.best, B.bx, B.hist, B.used, {"scouts": scouts})


ALGORITHMS = {"ABC": abc, "TLBO": tlbo, "GWO": gwo}
