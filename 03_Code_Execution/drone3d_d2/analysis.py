"""Analysis helpers: plan evaluation with energy components, simple dispatch rules, lower bound on fleet size."""
from __future__ import annotations
import math
from .dtypes import DroneType, TimeModel
from .fleet import Req, one_drone_table, fleet_frontier, EPS


def trip_components(leg, reqs, dt: DroneType, tm: TimeModel, sub: int, order):
    """Energy split of one trip: (cruise Wh, service-hover Wh, time s)."""
    hw = dt.hw
    S = [i for i in range(len(reqs)) if sub >> i & 1]
    load = sum(reqs[r].q for r in S if reqs[r].kind == "D")
    pos, ec, es, t = 0, 0.0, 0.0, 0.0
    for (r, typ, node) in order:
        e, tt = leg.et(pos, node, max(load, 0.0))
        ec += e; t += tt
        load += -reqs[r].q if typ == "d" else reqs[r].q
        if tm.service_hover:
            es += hw.hover_power(max(load, 0.0)) * tm.s_service / 3600.0
        t += tm.s_service
        pos = node
    e, tt = leg.et(pos, 0, max(load, 0.0))
    return ec + e, es, t + tt


def plan_components(leg, reqs, dt, tm, assignment):
    """Totals over a fleet assignment: cruise Wh, service-hover Wh, number of trips, drones used."""
    ec = es = 0.0; trips = 0; used = 0
    for plan in assignment:
        if plan:
            used += 1
        for (sub, _, _, order) in plan:
            c, s, _ = trip_components(leg, reqs, dt, tm, sub, order)
            ec += c; es += s; trips += 1
    return ec, es, trips, used


def eval_trip_order(leg, reqs, dt, tm, S, order_events):
    """Evaluate a given event order (list of (r, typ, node)) for subset S; returns (E, T) or None if infeasible."""
    hw = dt.hw
    load = sum(reqs[r].q for r in S if reqs[r].kind == "D")
    if load > hw.q_max + EPS:
        return None
    pos, E, T = 0, 0.0, 0.0
    picked = set()
    for (r, typ, node) in order_events:
        e, t = leg.et(pos, node, max(load, 0.0)); E += e; T += t
        load += -reqs[r].q if typ == "d" else reqs[r].q
        if load > hw.q_max + EPS or load < -EPS:
            return None
        if tm.service_hover:
            E += hw.hover_power(max(load, 0.0)) * tm.s_service / 3600.0
        T += tm.s_service; pos = node
    e, t = leg.et(pos, 0, max(load, 0.0)); E += e; T += t
    if E > hw.e_use_wh + EPS:
        return None
    return E, T


def _req_events(reqs, r):
    q = reqs[r]
    if q.kind == "D":
        return [(r, "d", q.d)]
    if q.kind == "P":
        return [(r, "p", q.o)]
    return [(r, "p", q.o), (r, "d", q.d)]


def _lpt(trips_et, k, tm):
    """Assign trips (E, T) to k identical drones by longest-processing-time; returns (E, makespan)."""
    loads = [0.0] * k; cnt = [0] * k
    for e, t in sorted(trips_et, key=lambda x: -x[1]):
        d = min(range(k), key=lambda i: loads[i])
        loads[d] += t; cnt[d] += 1
    mk = max((loads[d] + tm.tau * (cnt[d] - 1)) if cnt[d] else 0.0 for d in range(k))
    return sum(e for e, _ in trips_et), mk


def rule_one_trip_per_request(leg, reqs, dt, tm, k):
    trips = []
    for r in range(len(reqs)):
        res = eval_trip_order(leg, reqs, dt, tm, (r,), _req_events(reqs, r))
        if res is None:
            return None
        trips.append(res)
    return _lpt(trips, k, tm)


def rule_nearest_neighbour_bagging(leg, reqs, dt, tm, k):
    """Greedy: grow a trip by adding the nearest remaining request (by first-event travel energy) while the
    trip stays feasible (capacity, battery); events appended in request order (pickup then delivery); LPT to drones."""
    remaining = set(range(len(reqs)))
    trips = []
    while remaining:
        cur, order, S = 0, [], []
        while True:
            cand = []
            for r in remaining:
                ev = _req_events(reqs, r)
                cand.append((leg.et(cur, ev[0][2], 0.0)[0], r))
            added = False
            for _, r in sorted(cand):
                trial = order + _req_events(reqs, r)
                if eval_trip_order(leg, reqs, dt, tm, S + [r], trial) is not None:
                    order, S = trial, S + [r]; remaining.discard(r); cur = order[-1][2]; added = True
                    break
            if not added:
                break
        if not S:
            return None
        trips.append(eval_trip_order(leg, reqs, dt, tm, S, order))
    return _lpt(trips, k, tm)


def k_lower_bound(trips_tab, n, tm, T):
    """Energy-agnostic lower bound on the number of identical drones needed for makespan <= T:
    total minimum trip time of a partition cannot be shorter than W = DP over minimum-time trips (tau ignored);
    k >= ceil(W / T). Returns (W, k_lb)."""
    full = (1 << n) - 1
    INF = float("inf")
    best_t = {m: (min(t for _, t, _ in v) if v else INF) for m, v in trips_tab.items()}
    f = [INF] * (full + 1); f[0] = 0.0
    for mask in range(1, full + 1):
        low = mask & -mask; sub = mask
        while sub:
            if sub & low and best_t[sub] < INF and f[mask ^ sub] < INF:
                f[mask] = min(f[mask], best_t[sub] + f[mask ^ sub])
            sub = (sub - 1) & mask
    W = f[full]
    return W, (math.ceil(W / T - 1e-9) if W < INF else None)


def energy_unlimited_fleet(trips_tab, n, T):
    """Proposition 4: with an unlimited homogeneous fleet every trip can fly on its own drone, so
    E*(T; inf) = min over partitions into trips with trip time <= T of the summed trip energy
    (minimum energy among a subset's Pareto trips with time <= T). Returns Wh or None."""
    full = (1 << n) - 1
    INF = float("inf")
    c = {}
    for m, lst in trips_tab.items():
        v = [e for e, t, _ in lst if t <= T + 1e-9]
        c[m] = min(v) if v else INF
    f = [INF] * (full + 1); f[0] = 0.0
    for mask in range(1, full + 1):
        low = mask & -mask; sub = mask
        while sub:
            if sub & low and c[sub] < INF and f[mask ^ sub] < INF:
                f[mask] = min(f[mask], c[sub] + f[mask ^ sub])
            sub = (sub - 1) & mask
    return None if f[full] == INF else f[full]


def k_lower_bound_tau(trips_tab, n, tm, T):
    """Tight bound including turnaround (P3'): every plan's drones satisfy sum_trips t(c) + tau*(m_d - 1) <= T, so summing over
    k used drones: sum_c (t(c) + tau) <= k (T + tau)  =>  k >= W_tau / (T + tau), W_tau = min over partitions of sum_c (t_min(c) + tau)."""
    full = (1 << n) - 1
    INF = float("inf")
    cost = {m: (min(t for _, t, _ in v) + tm.tau if v else INF) for m, v in trips_tab.items()}
    f = [INF] * (full + 1); f[0] = 0.0
    for mask in range(1, full + 1):
        low = mask & -mask; sub = mask
        while sub:
            if sub & low and cost[sub] < INF and f[mask ^ sub] < INF:
                f[mask] = min(f[mask], cost[sub] + f[mask ^ sub])
            sub = (sub - 1) & mask
    W = f[full]
    return W, (math.ceil(W / (T + tm.tau) - 1e-9) if W < INF else None)
