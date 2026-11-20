"""Validation gates (run: python drone3d_d2/tests/test_core.py from 03_Code_Execution). All instances SYNTHETIC."""
import sys, os, random, itertools, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from drone3d_d2.fleet import Req, frontier_homogeneous, min_energy_at, trip_table, one_drone_table, fleet_frontier, pareto, EPS
from drone3d_d2.dtypes import TYPE_M600, TYPE_FLYCART, TimeModel, DroneType
from drone3d_d2.legs import PlanarLeg
from drone3d_d2._core import segment_energy_wh


def inst(n, seed, kinds, span=1500, qmax=3.0):
    r = random.Random(seed); pts = [(0, 0, 20.0)]; reqs = []
    for _ in range(n):
        k = r.choice(kinds); q = round(r.uniform(0.5, qmax), 1)
        pt = lambda: (r.uniform(-span, span), r.uniform(-span, span), r.uniform(15, 120))
        if k == "B":
            pts.append(pt()); pts.append(pt()); reqs.append(Req("B", len(pts) - 2, len(pts) - 1, q))
        elif k == "D":
            pts.append(pt()); reqs.append(Req("D", 0, len(pts) - 1, q))
        else:
            pts.append(pt()); reqs.append(Req("P", len(pts) - 1, 0, q))
    return pts, reqs


def evaluate_plan(leg, reqs, dt, tm, assignment):
    """Independent re-evaluation of a fleet plan: energy, makespan, feasibility (capacity, battery, precedence, cover)."""
    hw = dt.hw
    served = []
    E = 0.0; makespan = 0.0
    for plan in assignment:
        tdrone = 0.0
        for ti, (sub, _, _, order) in enumerate(plan):
            S = [i for i in range(len(reqs)) if sub >> i & 1]
            load = sum(reqs[r].q for r in S if reqs[r].kind == "D")
            assert load <= hw.q_max + 1e-9
            pos, e_tr, t_tr, picked = 0, 0.0, 0.0, set()
            for (r, typ, node) in order:
                e, t = leg.et(pos, node, max(load, 0.0))
                e_tr += e; t_tr += t
                load += -reqs[r].q if typ == "d" else reqs[r].q
                assert -1e-9 <= load <= hw.q_max + 1e-9
                if typ == "p": picked.add(r)
                if typ == "d" and reqs[r].kind == "B": assert r in picked
                if tm.service_hover: e_tr += hw.hover_power(max(load, 0.0)) * tm.s_service / 3600.0
                t_tr += tm.s_service; pos = node; served.append((r, typ))
            e, t = leg.et(pos, 0, max(load, 0.0)); e_tr += e; t_tr += t
            assert e_tr <= hw.e_use_wh + 1e-9
            E += e_tr; tdrone += t_tr + (tm.tau if ti < len(plan) - 1 else 0.0)
        makespan = max(makespan, tdrone)
    need = sorted((i, "d") for i, r in enumerate(reqs) if r.kind in "DB") + sorted((i, "p") for i, r in enumerate(reqs) if r.kind in "PB")
    assert sorted(served) == sorted(need), "coverage"
    return E, makespan


def brute_frontier_mixed(legs, reqs, dts, tm):
    """Independent brute force for a fleet with different drone types: legs[d]/dts[d] for drone d."""
    n = len(reqs)
    full = (1 << n) - 1
    per_type = {}
    for d, (leg, dt) in enumerate(zip(legs, dts)):
        trips = {}
        hw = dt.hw
        for mask in range(1, 1 << n):
            S = [i for i in range(n) if mask >> i & 1]
            ev = []
            for r in S:
                q = reqs[r]
                if q.kind == "D": ev.append((r, "d", q.d, -q.q))
                elif q.kind == "P": ev.append((r, "p", q.o, q.q))
                else: ev += [(r, "p", q.o, q.q), (r, "d", q.d, -q.q)]
            load0 = sum(reqs[r].q for r in S if reqs[r].kind == "D")
            L = []
            if load0 <= hw.q_max + 1e-9:
                for perm in itertools.permutations(range(len(ev))):
                    seen, ok = set(), True
                    for a in perm:
                        r, typ, _, _ = ev[a]
                        if typ == "p" and reqs[r].kind == "B": seen.add(r)
                        if typ == "d" and reqs[r].kind == "B" and r not in seen: ok = False; break
                    if not ok: continue
                    load, pos, e_tr, t_tr = load0, 0, 0.0, 0.0
                    for a in perm:
                        r, typ, node, dq = ev[a]
                        e, t = leg.et(pos, node, max(load, 0.0)); e_tr += e; t_tr += t
                        load += dq
                        if load > hw.q_max + 1e-9 or load < -1e-9: ok = False; break
                        if tm.service_hover: e_tr += hw.hover_power(max(load, 0.0)) * tm.s_service / 3600.0
                        t_tr += tm.s_service; pos = node
                    if not ok: continue
                    e, t = leg.et(pos, 0, max(load, 0.0)); e_tr += e; t_tr += t
                    if e_tr <= hw.e_use_wh + 1e-9: L.append((e_tr, t_tr))
            trips[mask] = L
        per_type[d] = trips
    # each drone d gets an ordered list of trips; enumerate: assign each trip of a set partition to a drone
    def parts(rem):
        if rem == 0: yield []; return
        low = rem & -rem; sub = rem
        while sub:
            if sub & low:
                for rest in parts(rem ^ sub): yield [sub] + rest
            sub = (sub - 1) & rem
    res = []
    K = len(dts)
    for part in parts(full):
        for assign in itertools.product(range(K), repeat=len(part)):
            opts = [per_type[assign[i]][m] for i, m in enumerate(part)]
            if any(not o for o in opts): continue
            for opt in itertools.product(*opts):
                tmax = 0.0
                for d in range(K):
                    idx = [i for i, a in enumerate(assign) if a == d]
                    if idx: tmax = max(tmax, sum(opt[i][1] for i in idx) + tm.tau * (len(idx) - 1))
                res.append((sum(o[0] for o in opt), tmax))
    return pareto(res)


def brute_frontier(leg, reqs, dt, tm, k, fixed_wh=0.0, max_trips=None):
    """Independent brute force (n <= 5): all request->(drone) set partitions into ordered trips, all event orders."""
    n = len(reqs); hw = dt.hw
    # enumerate every feasible trip for every subset (all event permutations), then combine by assigning trips to drones
    trips = {}
    for mask in range(1, 1 << n):
        S = [i for i in range(n) if mask >> i & 1]
        ev = []
        for r in S:
            q = reqs[r]
            if q.kind == "D": ev.append((r, "d", q.d, -q.q))
            elif q.kind == "P": ev.append((r, "p", q.o, q.q))
            else: ev += [(r, "p", q.o, q.q), (r, "d", q.d, -q.q)]
        load0 = sum(reqs[r].q for r in S if reqs[r].kind == "D")
        L = []
        for perm in itertools.permutations(range(len(ev))):
            ok = True; seen = set()
            for a in perm:
                r, typ, _, _ = ev[a]
                if typ == "p" and reqs[r].kind == "B": seen.add(r)
                if typ == "d" and reqs[r].kind == "B" and r not in seen: ok = False; break
            if not ok: continue
            load, pos, e_tr, t_tr = load0, 0, 0.0, 0.0
            if load > hw.q_max + 1e-9: continue
            for a in perm:
                r, typ, node, dq = ev[a]
                e, t = leg.et(pos, node, max(load, 0.0)); e_tr += e; t_tr += t
                load += dq
                if load > hw.q_max + 1e-9 or load < -1e-9: ok = False; break
                if tm.service_hover: e_tr += hw.hover_power(max(load, 0.0)) * tm.s_service / 3600.0
                t_tr += tm.s_service; pos = node
            if not ok: continue
            e, t = leg.et(pos, 0, max(load, 0.0)); e_tr += e; t_tr += t
            if e_tr <= hw.e_use_wh + 1e-9: L.append((e_tr, t_tr))
        trips[mask] = L
    full = (1 << n) - 1
    # set partitions into trips, then distribute trips over k drones (ordered sequences)
    pts = []
    def parts(rem):
        if rem == 0: yield []; return
        low = rem & -rem; sub = rem
        while sub:
            if sub & low:
                for rest in parts(rem ^ sub): yield [sub] + rest
            sub = (sub - 1) & rem
    res = []
    for part in parts(full):
        if any(not trips[m] for m in part): continue
        # choose a (E,T) option per trip, assign to drones by all functions trip->drone (order within a drone irrelevant for E,T sum)
        for opt in itertools.product(*[trips[m] for m in part]):
            for assign in itertools.product(range(k), repeat=len(part)):
                tmax = 0.0; used = 0; bad = False
                for d in range(k):
                    idx = [i for i, a in enumerate(assign) if a == d]
                    if idx:
                        if max_trips is not None and len(idx) > max_trips:
                            bad = True; break
                        used += 1
                        tt = sum(opt[i][1] for i in idx) + tm.tau * (len(idx) - 1)
                        tmax = max(tmax, tt)
                if bad: continue
                res.append((sum(o[0] for o in opt) + fixed_wh * used, tmax))
    return pareto(res)


def test_fixed_energy_and_inventory():
    """Per-dispatched-drone fixed energy and per-drone battery inventory (trip cap) vs brute force."""
    tm = TimeModel(s_service=60.0, tau=180.0)
    cnt = 0
    for kinds in ("DP", "DPB"):
        for n in ((3, 4) if kinds == "DPB" else (3, 4, 5)):
            for seed in (1001, 1002):
                pts, reqs = inst(n, seed, kinds, span=1500)
                leg = PlanarLeg(pts, TYPE_M600)
                for k in (1, 2, 3):
                    for fx, b in ((20.0, None), (0.0, 1), (0.0, 2), (20.0, 2)):
                        F, _, _ = frontier_homogeneous(leg, reqs, TYPE_M600, tm, k, fixed_wh=fx, max_trips=b)
                        B = brute_frontier(leg, reqs, TYPE_M600, tm, k, fixed_wh=fx, max_trips=b)
                        a = [(round(e, 6), round(t, 6)) for e, t, _ in F]; c = [(round(e, 6), round(t, 6)) for e, t in B]
                        assert a == c, (kinds, n, seed, k, fx, b, a, c)
                        cnt += 1
    # defaults reproduce the base frontier
    pts, reqs = inst(4, 1001, "DP"); leg = PlanarLeg(pts, TYPE_M600)
    F0, _, _ = frontier_homogeneous(leg, reqs, TYPE_M600, tm, 2)
    F1, _, _ = frontier_homogeneous(leg, reqs, TYPE_M600, tm, 2, fixed_wh=0.0, max_trips=99)
    assert [(round(e, 6), round(t, 6)) for e, t, _ in F0] == [(round(e, 6), round(t, 6)) for e, t, _ in F1]
    print("gate fixed energy + battery inventory vs brute force OK (%d cases)" % cnt)


def test_slack_energy_formula():
    """Proposition 1' (closed form): with unlimited time, E_slack(k; b, c) = min over partitions into m <= k*b trips of
    sum_trip e_min + c * ceil(m / b)  (b = None: m free, one drone, cost c). Checked against the Pareto-frontier solver."""
    tm = TimeModel(s_service=60.0, tau=180.0)
    cnt = 0
    for kinds in ("DP", "DPB"):
        for n in (4, 5, 6):
            for seed in (1001, 1002, 1003, 1004):
                pts, reqs = inst(n, seed, kinds, span=1500)
                leg = PlanarLeg(pts, TYPE_M600)
                trips = trip_table(leg, reqs, TYPE_M600, tm)
                emin = {m: (min(e for e, _, _ in v) if v else math.inf) for m, v in trips.items()}
                full = (1 << n) - 1
                # f[m][mask] = min energy partitioning mask into exactly m trips
                f = [[math.inf] * (full + 1) for _ in range(n + 1)]; f[0][0] = 0.0
                for m in range(1, n + 1):
                    for mask in range(1, full + 1):
                        low = mask & -mask; sub = mask
                        while sub:
                            if sub & low and emin[sub] < math.inf and f[m - 1][mask ^ sub] < math.inf:
                                f[m][mask] = min(f[m][mask], emin[sub] + f[m - 1][mask ^ sub])
                            sub = (sub - 1) & mask
                for b in (1, 2, None):
                    for c in (0.0, 30.0):
                        for k in (1, 2, 3):
                            F, _, _ = frontier_homogeneous(leg, reqs, TYPE_M600, tm, k, fixed_wh=c, max_trips=b)
                            best = math.inf
                            for m in range(1, n + 1):
                                if b is not None and m > k * b:
                                    continue
                                drones = 1 if b is None else -(-m // b)
                                best = min(best, f[m][full] + c * drones)
                            got = F[0][0] if F else math.inf
                            assert (got == math.inf and best == math.inf) or abs(got - best) < 1e-6, (kinds, n, seed, b, c, k, got, best)
                            cnt += 1
    print("gate slack-energy closed form (Proposition 1') OK (%d cases)" % cnt)


def test_d1_identity():
    try:
        from exact import Request, solve_exact
    except ImportError:
        print("gate D1 identity SKIPPED (single-drone solver of the first paper not available)"); return
    tm0 = TimeModel(s_service=0.0, tau=0.0)
    for kinds in ("DP", "DPB"):
        for n in (4, 5, 6):
            for seed in (1001, 1002, 1003):
                pts, reqs = inst(n, seed, kinds)
                leg = PlanarLeg(pts, TYPE_M600)
                class L:
                    def energy(s, i, j, load): return leg.et(i, j, load)[0]
                E1, _ = solve_exact(L(), [Request(r.kind, r.o, r.d, r.q) for r in reqs], TYPE_M600.hw.q_max, TYPE_M600.hw.e_use_wh)
                F, _, _ = frontier_homogeneous(leg, reqs, TYPE_M600, tm0, 3)
                assert abs(E1 - F[0][0]) < 1e-9, (kinds, n, seed, E1, F[0][0])
    print("gate D1 identity OK")


def test_brute_pareto():
    tm = TimeModel(s_service=60.0, tau=180.0)
    cnt = 0; nonempty = 0
    for kinds in ("DP", "DPB"):
        for n in ((3, 4) if kinds == "DPB" else (3, 4, 5)):
            for seed in (1001, 1002, 1003):
                for span in (1500, 500, 9000):
                    pts, reqs = inst(n, seed, kinds, span=span)
                    leg = PlanarLeg(pts, TYPE_M600)
                    for k in (1, 2, 3):
                        F, _, _ = frontier_homogeneous(leg, reqs, TYPE_M600, tm, k)
                        B = brute_frontier(leg, reqs, TYPE_M600, tm, k)
                        a = [(round(e, 6), round(t, 6)) for e, t, _ in F]; b = [(round(e, 6), round(t, 6)) for e, t in B]
                        assert a == b, (kinds, n, seed, span, k, a, b)
                        nonempty += (len(a) > 0)
                        for e, t, assign in F:
                            E2, T2 = evaluate_plan(leg, reqs, TYPE_M600, tm, assign)
                            assert abs(E2 - e) < 1e-6 and abs(T2 - t) < 1e-6, ("plan mismatch", e, t, E2, T2)
                        cnt += 1
    print("gate brute-force Pareto + plan re-evaluation OK (%d cases, %d with non-empty frontier; the rest are empty==empty)" % (cnt, nonempty))


def test_k_identity_and_monotone():
    tm = TimeModel()
    for seed in (1001, 1002, 1003):
        pts, reqs = inst(6, seed, "DPB")
        leg = PlanarLeg(pts, TYPE_M600)
        F1, _, _ = frontier_homogeneous(leg, reqs, TYPE_M600, tm, 1)
        F3, _, _ = frontier_homogeneous(leg, reqs, TYPE_M600, tm, 3)
        assert abs(F1[0][0] - F3[0][0]) < 1e-9, "E(inf) must not depend on k when time is slack"
        assert F3[-1][1] <= F1[-1][1] + 1e-9, "more drones cannot raise min makespan"
        for F in (F1, F3):
            for (e1, t1, _), (e2, t2, _) in zip(F, F[1:]):
                assert e2 > e1 - 1e-12 and t2 < t1 + 1e-12
    print("gate k identity + monotone OK")


def test_identical_types_reduce():
    tm = TimeModel()
    pts, reqs = inst(5, 1001, "DP")
    same = DroneType("M600b", TYPE_M600.hw)
    legA, legB = PlanarLeg(pts, TYPE_M600), PlanarLeg(pts, same)
    hA = one_drone_table(trip_table(legA, reqs, TYPE_M600, tm), 5, tm)
    hB = one_drone_table(trip_table(legB, reqs, same, tm), 5, tm)
    FM = fleet_frontier([hA, hB], 5)
    FH, _, _ = frontier_homogeneous(legA, reqs, TYPE_M600, tm, 2)
    assert [(round(e, 6), round(t, 6)) for e, t, _ in FM] == [(round(e, 6), round(t, 6)) for e, t, _ in FH]
    print("gate identical mixed == homogeneous OK")


def test_closed_form_flat():
    # one delivery at distance L, z = 0 everywhere, load 0 after drop: E = cruise there with q + hover service + cruise back empty
    hw = TYPE_M600.hw
    tm = TimeModel(s_service=60.0, tau=0.0)
    L, q = 1000.0, 2.0
    pts = [(0, 0, 0.0), (L, 0, 0.0)]
    leg = PlanarLeg(pts, TYPE_M600)
    F, _, _ = frontier_homogeneous(leg, [Req("D", 0, 1, q)], TYPE_M600, tm, 1)
    t_leg = L / TYPE_M600.v
    e = hw.hover_power(q) * t_leg / 3600 + hw.hover_power(0.0) * 60 / 3600 + hw.hover_power(0.0) * t_leg / 3600
    assert abs(F[0][0] - e) < 1e-9 and abs(F[0][1] - (2 * t_leg + 60.0)) < 1e-9, (F[0][:2], e)
    print("gate closed form (flat, single delivery) OK")


def test_unlimited_fleet_proposition():
    from drone3d_d2.analysis import energy_unlimited_fleet
    tm = TimeModel()
    for kinds in ("DP", "DPB"):
        for n in (4, 5, 6):
            for seed in (1001, 1002, 1003):
                for span in (400, 1500):
                    pts, reqs = inst(n, seed, kinds, span=span)
                    leg = PlanarLeg(pts, TYPE_M600)
                    trips = trip_table(leg, reqs, TYPE_M600, tm)
                    F, _, _ = frontier_homogeneous(leg, reqs, TYPE_M600, tm, n)   # k = n drones = unlimited
                    for T in (300, 450, 600, 900, 1500):
                        a = min_energy_at(F, T); b = energy_unlimited_fleet(trips, n, T)
                        assert (a is None) == (b is None), (kinds, n, seed, span, T, a, b)
                        if a is not None:
                            assert abs(a[0] - b) < 1e-6, (kinds, n, seed, span, T, a, b)
    print("gate Proposition 4 (E*(T; inf) = trip-time-capped partition DP) OK")


def test_mixed_type_brute():
    from drone3d_d2.dtypes import FLYCART_P09
    tm = TimeModel(); cnt = 0
    for n in (3, 4):
        for seed in (1001, 1002, 1003):
            pts, reqs = inst(n, seed, "DPB", span=1500, qmax=3.0)
            # make one request heavy so only FlyCart can carry it
            reqs = [Req(reqs[0].kind, reqs[0].o, reqs[0].d, 12.0)] + reqs[1:]
            dts = [TYPE_M600, TYPE_FLYCART, DroneType("FC15", TYPE_FLYCART.hw, v=15.0)]
            legs = [PlanarLeg(pts, d) for d in dts]
            hs = [one_drone_table(trip_table(l, reqs, d, tm), n, tm) for l, d in zip(legs, dts)]
            F = fleet_frontier(hs, n)
            B = brute_frontier_mixed(legs, reqs, dts, tm)
            a = [(round(e, 6), round(t, 6)) for e, t, _ in F]; b = [(round(e, 6), round(t, 6)) for e, t in B]
            assert a == b and len(a) > 0, (n, seed, a, b)
            cnt += 1
    print("gate mixed-type brute-force (M600 + FlyCart10 + FlyCart15, heavy parcel) OK (%d, non-vacuous)" % cnt)


def test_flycart_power_base():
    from drone3d_d2.dtypes import FLYCART, FLYCART_P09
    assert abs(FLYCART.p0 - 3968.8 / (29.0 / 60.0)) < 1e-6           # power base E: P = E / t
    assert abs(FLYCART_P09.p0 - 0.9 * 3968.8 / (29.0 / 60.0)) < 1e-6  # variant: 0.9 E / t
    assert abs(FLYCART.e_use_wh - 0.9 * 3968.8) < 1e-9 and abs(FLYCART.alpha - FLYCART_P09.alpha) < 1e-12
    print("gate FlyCart power base OK")


def test_speed_power_model():
    from drone3d_d2.dtypes import DroneType
    a = DroneType("a", TYPE_M600.hw, v=10.0, ind_share=0.8)
    assert abs(a.kappa() - 1.0) < 1e-12 and a.kappa(15.0) > 1.2 and a.kappa(8.0) > 1.05
    b = DroneType("b", TYPE_M600.hw, v=12.0, kappa_c=1.0, ind_share=None)
    assert b.kappa() == 1.0
    pts, reqs = inst(4, 1001, "DP"); tm = TimeModel()
    F1, _, _ = frontier_homogeneous(PlanarLeg(pts, TYPE_M600), reqs, TYPE_M600, tm, 2)
    F2, _, _ = frontier_homogeneous(PlanarLeg(pts, DroneType("c", TYPE_M600.hw, v=10.0, ind_share=0.8)), reqs, DroneType("c", TYPE_M600.hw, v=10.0, ind_share=0.8), tm, 2)
    assert [(round(e, 6), round(t, 6)) for e, t, _ in F1] == [(round(e, 6), round(t, 6)) for e, t, _ in F2]  # kappa(v_ref) = 1
    print("gate speed-power model OK")


if __name__ == "__main__":
    test_closed_form_flat(); test_d1_identity(); test_k_identity_and_monotone(); test_identical_types_reduce(); test_flycart_power_base(); test_speed_power_model(); test_unlimited_fleet_proposition(); test_mixed_type_brute(); test_brute_pareto(); test_fixed_energy_and_inventory(); test_slack_energy_formula()
    print("ALL GATES PASSED")
