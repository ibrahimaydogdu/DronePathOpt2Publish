"""Scale-bridge heuristic (Phase B): simulated annealing for min energy s.t. makespan <= T, k identical drones.

Solution = list of drones; a drone = list of trips; a trip = list of events (r, typ, node). A trip leaves the depot loaded with
all D masses of its requests (same convention as the exact solver). Trip feasibility/energy/time come from
analysis.eval_trip_order (capacity, battery, service hover), so the heuristic and the exact solver share one cost model.
Moves: relocate a request (both events) to a random position of another trip / a new trip / another drone; swap two requests;
reverse a segment inside a trip. Objective: E + PEN * max(0, makespan - T). Reports the best FEASIBLE (makespan <= T) solution.
No claim of optimality; the gap is measured against the exact frontier for n <= 10 (run_scale_bridge.py).
"""
from __future__ import annotations
import math
import random
from .analysis import eval_trip_order, _req_events

PEN = 50.0   # Wh per second of makespan violation (large vs typical trip energy gradients)


class _Eval:
    def __init__(self, leg, reqs, dt, tm):
        self.leg, self.reqs, self.dt, self.tm = leg, reqs, dt, tm
        self.cache = {}

    def trip(self, ev):
        key = tuple(ev)
        v = self.cache.get(key, 0)
        if v == 0:
            S = sorted({e[0] for e in ev})
            v = eval_trip_order(self.leg, self.reqs, self.dt, self.tm, S, ev)
            self.cache[key] = v
        return v   # (E, T) or None


def _cost(ev_, sol, T):
    """Returns (objective, E, makespan) or None if any trip infeasible."""
    E = 0.0; mk = 0.0
    for plan in sol:
        t = 0.0
        for k, trip in enumerate(plan):
            r = ev_.trip(trip)
            if r is None:
                return None
            E += r[0]; t += r[1] + (ev_.tm.tau if k else 0.0)
        mk = max(mk, t)
    return E + PEN * max(0.0, mk - T), E, mk


def _remove(sol, r):
    out = []
    for plan in sol:
        np_ = []
        for trip in plan:
            tr = [e for e in trip if e[0] != r]
            if tr:
                np_.append(tr)
        out.append(np_)
    return out


def _insert_events(trip, evs, rng):
    """Insert request events into a trip at random positions keeping pickup before delivery."""
    t = list(trip)
    for k, e in enumerate(evs):
        lo = 0
        if k == 1:   # delivery of a B request must follow its pickup
            lo = max(i for i, x in enumerate(t) if x[0] == e[0]) + 1
        t.insert(rng.randint(lo, len(t)), e)
    return t


def initial(ev_, n, k):
    """One trip per request, longest first onto the least-loaded drone."""
    sol = [[] for _ in range(k)]
    items = []
    for r in range(n):
        evs = _req_events(ev_.reqs, r)
        res = ev_.trip(evs)
        if res is None:
            return None
        items.append((res[1], evs))
    loads = [0.0] * k
    for t, evs in sorted(items, key=lambda x: -x[0]):
        d = min(range(k), key=lambda i: loads[i])
        sol[d].append(evs); loads[d] += t + (ev_.tm.tau if sol[d][1:] else 0.0)
    return sol


def _neighbor(cur, rng, reqs, n, k):
    """Random neighbour of a solution (relocate / swap / reverse), or None if the move is not applicable."""
    mv = rng.random()
    sol = [[list(t) for t in plan] for plan in cur]
    if mv < 0.55:   # relocate
        r = rng.randrange(n)
        sol = _remove(sol, r)
        evs = _req_events(reqs, r)
        flat = [(d, i) for d in range(k) for i in range(len(sol[d]))]
        if flat and rng.random() < 0.6:
            d, i = rng.choice(flat)
            sol[d][i] = _insert_events(sol[d][i], evs, rng)
        else:
            d = rng.randrange(k)
            sol[d].insert(rng.randint(0, len(sol[d])), list(evs))
        return sol
    if mv < 0.8:    # swap two requests (reinserted at random trips)
        if n < 2:
            return None
        a, b = rng.sample(range(n), 2)
        ea, eb = _req_events(reqs, a), _req_events(reqs, b)
        sol = _remove(sol, a); sol = _remove(sol, b)
        flat = [(d, i) for d in range(k) for i in range(len(sol[d]))]
        for evs in (ea, eb):
            if flat:
                d, i = rng.choice(flat)
                sol[d][i] = _insert_events(sol[d][i], evs, rng)
            else:
                sol[0].append(list(evs)); flat = [(0, 0)]
        return sol
    flat = [(d, i) for d in range(k) for i in range(len(sol[d])) if len(sol[d][i]) > 2]
    if not flat:
        return None
    d, i = rng.choice(flat)
    tr = sol[d][i]
    a, b = sorted(rng.sample(range(len(tr)), 2))
    tr[a:b + 1] = tr[a:b + 1][::-1]
    seen = set()
    for e in tr:
        if e[1] == "d" and reqs[e[0]].kind == "B" and e[0] not in seen:
            return None
        if e[1] == "p":
            seen.add(e[0])
    return sol


def _phase1(ev_, cur, T, k, reqs, n, rng, iters):
    """Feasibility repair: minimise makespan (energy as tie-break) until makespan <= T; returns a solution or None."""
    cc = _cost(ev_, cur, 0.0)   # PEN * mk dominates; objective ~ makespan
    if cc is None:
        return None
    if cc[2] <= T + 1e-9:
        return cur
    temp = max(1.0, 0.02 * cc[2])
    for it in range(iters):
        sol = _neighbor(cur, rng, reqs, n, k)
        if sol is None:
            continue
        c = _cost(ev_, sol, 0.0)
        if c is None:
            continue
        f_new, f_cur = c[2] + 1e-6 * c[1], cc[2] + 1e-6 * cc[1]
        if f_new <= f_cur or rng.random() < math.exp(-(f_new - f_cur) / temp):
            cur, cc = sol, c
            if cc[2] <= T + 1e-9:
                return cur
        temp = max(temp * 0.9995, 1e-3)
    return None


def anneal(leg, reqs, dt, tm, k, T, iters=20000, seed=1001, t0=None, alpha=0.9995, evaluator=None, phase1_iters=None, start=None):
    """Two phases. Phase 1 (only if the start is not time-feasible): minimise makespan until <= T (repair).
    Phase 2: simulated annealing on energy with the makespan constraint HARD (violating neighbours rejected).
    Returns (E, makespan, solution) or None (no time-feasible solution found; this is NOT a proof of infeasibility)."""
    rng = random.Random(seed)
    ev_ = evaluator or _Eval(leg, reqs, dt, tm)
    n = len(reqs)
    cur = start if start is not None else initial(ev_, n, k)
    if cur is None:
        return None
    cur = _phase1(ev_, cur, T, k, reqs, n, rng, phase1_iters or iters)
    if cur is None:
        return None
    cc = _cost(ev_, cur, T)
    best, bestsol = (cc[1], cc[2]), cur
    temp = t0 if t0 is not None else max(1.0, 0.02 * cc[1])
    for it in range(iters):
        sol = _neighbor(cur, rng, reqs, n, k)
        if sol is None:
            continue
        c = _cost(ev_, sol, T)
        if c is None or c[2] > T + 1e-9:
            continue
        if c[1] <= cc[1] or rng.random() < math.exp(-(c[1] - cc[1]) / temp):
            cur, cc = sol, c
            if c[1] < best[0]:
                best, bestsol = (c[1], c[2]), sol
        temp = max(temp * alpha, 1e-3)
    return best[0], best[1], bestsol


def _pack(ev_, trips, k):
    """LPT packing of trips onto k drones (start solution for the finite-fleet search)."""
    sol = [[] for _ in range(k)]; loads = [0.0] * k
    for tr in sorted(trips, key=lambda x: -ev_.trip(x)[1]):
        d = min(range(k), key=lambda i: loads[i])
        loads[d] += ev_.trip(tr)[1] + (ev_.tm.tau if sol[d] else 0.0); sol[d].append(tr)
    return sol


def anneal_multi(leg, reqs, dt, tm, k, T, restarts=3, iters=20000, seed0=1001, alpha=0.9995):
    """Best of independent restarts. Restart 0 starts from the 'unlimited fleet' solution (every trip on its own drone,
    trip time <= T; energy-optimal by Proposition 1c when found) packed LPT onto k drones; the others start from
    one-trip-per-request. The evaluation budget is (restarts + 1) x iters (+ phase 1)."""
    ev_ = _Eval(leg, reqs, dt, tm)
    n = len(reqs)
    bst = None
    inf = anneal(leg, reqs, dt, tm, n, T, iters=iters, seed=seed0 + 777, evaluator=ev_, phase1_iters=3 * iters, alpha=alpha)
    starts = []
    if inf is not None:
        trips = [list(t) for plan in inf[2] for t in plan]
        starts.append(_pack(ev_, trips, k))
        if k >= len(trips) and inf[1] <= T + 1e-9:
            bst = (inf[0], inf[1], inf[2])
    starts += [None] * max(restarts - len(starts), 1)
    for j, st in enumerate(starts):
        r = anneal(leg, reqs, dt, tm, k, T, iters=iters, seed=seed0 + 1000 * j, evaluator=ev_, phase1_iters=3 * iters, alpha=alpha, start=st)
        if r and (bst is None or r[0] < bst[0]):
            bst = r
    return bst
