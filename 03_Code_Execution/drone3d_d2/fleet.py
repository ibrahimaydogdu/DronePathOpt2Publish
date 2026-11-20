"""Exact (energy, makespan) Pareto frontier for multi-trip, multi-drone single-depot pickup/delivery.

Requests: Req(kind, o, d, q) with kind 'D' (depot -> d), 'P' (o -> depot), 'B' (o -> d). A trip leaves the depot loaded
with all D masses of its subset and returns to the depot; battery e_use per trip. A drone's trips are sequential with
turnaround tau between them (not after the last). Makespan = last drone's return to the depot.
Method: (1) per request subset, label-correcting Pareto over (events done, last event) with (E, T) labels;
(2) one-drone table by subset DP (time adds, +tau between trips); (3) fleet table by subset combination (time = max).
"""
from __future__ import annotations
from dataclasses import dataclass
from .dtypes import DroneType, TimeModel

INF = float("inf")
EPS = 1e-9


E_TOL = 1e-7   # Wh: energies closer than this are the same value up to floating-point summation order


def pareto(items, key=lambda x: x):
    """Keep non-dominated (E, T); returns list sorted by E ascending, T strictly decreasing.
    Energies within E_TOL are treated as equal (the smaller time wins)."""
    items = sorted(items, key=lambda x: (key(x)[0], key(x)[1]))
    out = []
    for it in items:
        e, t = key(it)[0], key(it)[1]
        if out and e - key(out[-1])[0] <= E_TOL:
            if t < key(out[-1])[1] - EPS:
                out[-1] = it
            continue
        if not out or t < key(out[-1])[1] - EPS:
            out.append(it)
    return out


@dataclass(frozen=True)
class Req:
    kind: str
    o: int
    d: int
    q: float


def _events(reqs, S):
    ev = []
    for r in S:
        q = reqs[r]
        if q.kind == "D":
            ev.append((r, "d", q.d, -q.q))
        elif q.kind == "P":
            ev.append((r, "p", q.o, +q.q))
        else:
            ev.append((r, "p", q.o, +q.q))
            ev.append((r, "d", q.d, -q.q))
    return ev


def trip_pareto(leg, reqs, S, dt: DroneType, tm: TimeModel):
    """Pareto list of (E, T, order) for the single trip serving subset S (tuple of request indices)."""
    hw = dt.hw
    ev = _events(reqs, S)
    m = len(ev)
    load0 = sum(reqs[r].q for r in S if reqs[r].kind == "D")
    if load0 > hw.q_max + EPS:
        return []
    pick_of = {r: a for a, (r, t, _, _) in enumerate(ev) if t == "p" and reqs[r].kind == "B"}
    eu = hw.e_use_wh
    lab = {(0, -1): [(0.0, 0.0, ())]}
    final = []
    full = (1 << m) - 1
    for done in range(1 << m):
        load = load0 + sum(ev[a][3] for a in range(m) if done >> a & 1)
        for last in range(-1, m):
            L = lab.get((done, last))
            if not L:
                continue
            pos = 0 if last < 0 else ev[last][2]
            if done == full:
                e, t = leg.et(pos, 0, max(load, 0.0))
                for x, y, path in L:
                    if x + e <= eu + EPS:
                        final.append((x + e, y + t, path))
                continue
            for a in range(m):
                if done >> a & 1:
                    continue
                r, typ, node, dq = ev[a]
                if typ == "d" and reqs[r].kind == "B" and not done >> pick_of[r] & 1:
                    continue
                nl = load + dq
                if nl > hw.q_max + EPS or nl < -EPS:
                    continue
                e, t = leg.et(pos, node, max(load, 0.0))
                es = hw.hover_power(max(nl, 0.0)) * tm.s_service / 3600.0 if tm.service_hover else 0.0
                key = (done | 1 << a, a)
                tgt = lab.get(key, [])
                for x, y, path in L:
                    if x + e + es <= eu + EPS:
                        tgt.append((x + e + es, y + t + tm.s_service, path + ((r, typ, node),)))
                lab[key] = pareto(tgt)
    return pareto(final)


def trip_table(leg, reqs, dt, tm, max_size=None):
    """Pareto (E, T) options of every request subset as one trip. max_size: trips with more requests are not enumerated (empty list);
    the result is exact only if no optimal plan needs a larger trip (check against max_size=None on smaller instances)."""
    n = len(reqs)
    tab = {}
    for mask in range(1, 1 << n):
        S = tuple(i for i in range(n) if mask >> i & 1)
        tab[mask] = [] if (max_size is not None and len(S) > max_size) else trip_pareto(leg, reqs, S, dt, tm)
    return tab


def one_drone_table(trips, n, tm: TimeModel, max_trips=None):
    """h[mask] = Pareto list of (E, T, plan) for ONE drone serving exactly mask; plan = ((sub, E, T, order), ...).
    max_trips: battery inventory per drone (a drone may fly at most this many trips in the horizon); None = unlimited."""
    full = (1 << n) - 1
    if max_trips is None:
        h = {0: [(0.0, 0.0, ())]}
        for mask in range(1, full + 1):
            low = mask & -mask
            C, sub = [], mask
            while sub:
                if sub & low and trips[sub]:
                    rest = mask ^ sub
                    if h.get(rest):
                        gap = tm.tau if rest else 0.0
                        for e1, t1, o1 in trips[sub]:
                            for e2, t2, p2 in h[rest]:
                                C.append((e1 + e2, t1 + t2 + gap, ((sub, e1, t1, o1),) + p2))
                sub = (sub - 1) & mask
            h[mask] = pareto(C)
        return h
    layer = {0: [(0.0, 0.0, ())]}          # exactly c trips; c = 0
    agg = {0: [(0.0, 0.0, ())]}
    for m in range(1, full + 1):
        agg[m] = []
    for c in range(1, max_trips + 1):
        nxt = {}
        for mask in range(1, full + 1):
            low = mask & -mask
            C, sub = [], mask
            while sub:
                if sub & low and trips[sub]:
                    rest = mask ^ sub
                    if layer.get(rest):
                        gap = tm.tau if rest else 0.0
                        for e1, t1, o1 in trips[sub]:
                            for e2, t2, p2 in layer[rest]:
                                C.append((e1 + e2, t1 + t2 + gap, ((sub, e1, t1, o1),) + p2))
                sub = (sub - 1) & mask
            nxt[mask] = pareto(C)
            agg[mask] = agg[mask] + nxt[mask]
        nxt[0] = []
        layer = nxt
    return {m: (pareto(v) if m else v) for m, v in agg.items()}


def fleet_frontier(tables, n, fixed_wh=0.0):
    """tables: one one-drone table per drone (may differ by type; a drone may stay unused).
    Returns Pareto list of (E, makespan, assignment) for all requests; assignment[d] = plan of drone d (reverse order)."""
    full = (1 << n) - 1
    G = {m: ([(0.0, 0.0, ())] if m == 0 else []) for m in range(full + 1)}
    for h in tables:
        N = {0: [(0.0, 0.0, ((),))]}
        for mask in range(1, full + 1):
            C, sub = [], mask
            while True:                       # sub = 0 allowed: drone unused
                rest = mask ^ sub
                if h.get(sub) and G.get(rest):
                    for e1, t1, p1 in h[sub]:
                        for e2, t2, a2 in G[rest]:
                            C.append((e1 + e2 + (fixed_wh if sub else 0.0), max(t1, t2), a2 + (p1,)))
                if sub == 0:
                    break
                sub = (sub - 1) & mask
            N[mask] = pareto(C)
        G = N
    return G[full]


def frontier_homogeneous(leg, reqs, dt, tm, k, fixed_wh=0.0, max_trips=None):
    """fixed_wh: energy charged per DISPATCHED drone (unused drones cost nothing); max_trips: battery inventory per drone."""
    n = len(reqs)
    trips = trip_table(leg, reqs, dt, tm)
    h = one_drone_table(trips, n, tm, max_trips)
    return fleet_frontier([h] * k, n, fixed_wh), trips, h


def min_energy_at(frontier, T):
    """Smallest E on the frontier with makespan <= T (None if infeasible)."""
    best = None
    for e, t, _ in frontier:
        if t <= T + EPS and (best is None or e < best[0]):
            best = (e, t)
    return best


def min_time(frontier):
    return frontier[-1][1] if frontier else INF
