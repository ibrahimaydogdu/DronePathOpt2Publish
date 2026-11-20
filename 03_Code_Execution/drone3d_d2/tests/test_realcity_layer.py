"""Tests for the real-city layer, the weighted-sum hull count and the baseline planners (run: python -I -m pytest drone3d_d2/tests/test_realcity_layer.py from 03_Code_Execution).
Data-dependent tests are skipped when the downloaded building layers are absent (Evidence/ is not redistributed)."""
import sys, os, math
import pytest
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "03_Code_Execution")); sys.path.insert(0, os.path.join(ROOT, "03_Code_Execution", "core_d1"))
sys.path.insert(0, os.path.join(ROOT, "04_Results"))
try:
    from drone3d_d2.realcity import height_la, height_chicago, height_austin, BLD, make_real_instance, station_data
    from geometry import FT
    from run_scalarization import hull_supported
except ImportError:
    pytest.skip("leg-cost module of the first paper not available", allow_module_level=True)

HAVE = all(os.path.exists(BLD[c]) for c in ("chicago", "los_angeles", "austin")) if all(c in BLD for c in ("chicago", "los_angeles", "austin")) else False


def F(pts):
    return [(e, t, None) for e, t in pts]


def test_hull_known_sets():
    assert hull_supported(F([(0, 10), (1, 4), (2, 0)])) == 3          # strictly convex: all supported
    assert hull_supported(F([(0, 10), (1, 8), (2, 0)])) == 2          # middle point above the chord: unsupported
    assert hull_supported(F([(0, 10), (1, 5), (2, 0)])) == 2          # collinear middle point: not unique optimum
    assert hull_supported(F([(0, 10)])) == 1


def test_hull_affine_invariance():
    P = [(0, 20), (1, 9), (2, 8), (3, 3), (5, 0)]
    a = hull_supported(F(P))
    assert a == hull_supported(F([(3.7 * e, 0.01 * t) for e, t in P]))


def test_height_functions():
    assert height_la({"HEIGHT": 10}) == pytest.approx(10 * FT)
    assert math.isnan(height_la({"HEIGHT": 0})) and math.isnan(height_la({}))
    ch = height_chicago(3.5, 0)
    assert ch({"stories": 4}) == pytest.approx(14.0) and math.isnan(ch({"stories": 0})) and math.isnan(ch({}))
    assert height_chicago(3.5, 2)({"stories": 0}) == pytest.approx(7.0)
    au = height_austin(0.0)
    assert au({"max_height": 20}) == pytest.approx(20 * FT)
    assert math.isnan(au({"max_height": -9999999999})) and math.isnan(au({"max_height": None}))
    assert height_austin(6.0)({"max_height": -9999999999}) == pytest.approx(6.0)


@pytest.mark.skipif(not HAVE, reason="building layers not present")
@pytest.mark.parametrize("code,hf", [("DLA4", height_la), ("DCH1", height_chicago(3.5, 0)), ("DAU1", height_austin(0.0))])
def test_instance_properties(code, hf):
    city, nodes, reqs, meta = make_real_instance(code, 8, 1001, 3000.0, 100.0, hf)
    city2, nodes2, reqs2, meta2 = make_real_instance(code, 8, 1001, 3000.0, 100.0, hf)
    assert reqs == reqs2 and meta == meta2                              # same seed, same draw
    assert len(reqs) == 8 and all(r[0] == "D" and r[1] == 0 for r in reqs)
    assert all(0.1 <= r[3] <= 3.0 + 1e-9 for r in reqs)                 # carryable by the drone
    assert len({r[2] for r in reqs}) == 8                               # distinct delivery nodes
    _, _, reqs3, _ = make_real_instance(code, 8, 1002, 3000.0, 100.0, hf)
    assert reqs3 != reqs                                                # different seed, different draw


@pytest.mark.skipif(not HAVE, reason="building layers not present")
def test_baseline_plans_meet_makespan_bound_under_true_model():
    """DIST and LOADBL plans, re-evaluated under the full model, must not exceed the makespan bound T (reviewer check)."""
    import run_baselines as rb
    from drone3d_d2.fleet import Req, trip_table, one_drone_table, fleet_frontier
    from drone3d_d2.legs import CityLeg
    from drone3d_d2.dtypes import TYPE_M600
    from legs import LegCost
    from around import CityRaster
    city, nodes, reqs, meta = make_real_instance("DCH1", rb.N, 1001, rb.RADIUS, 100.0, height_chicago(3.5, 0))
    leg = CityLeg(LegCost(city, nodes, model="F", hw=TYPE_M600.hw, v=10.0, raster=CityRaster(city, res=10.0)), TYPE_M600)
    R = [Req(*r) for r in reqs]
    F_ = fleet_frontier([one_drone_table(trip_table(leg, R, TYPE_M600, rb.TM), rb.N, rb.TM)] * rb.K, rb.N)
    tmin, temin = F_[-1][1], F_[0][1]
    from drone3d_d2.analysis import eval_trip_order
    for bleg, btm in ((rb.DistLeg(leg), rb.TM_PROXY), (rb.LoadBlind(leg), rb.TM)):
        FB = fleet_frontier([one_drone_table(trip_table(bleg, R, TYPE_M600, btm), rb.N, btm)] * rb.K, rb.N)
        for th in (0.0, 0.5, 1.0):
            T = tmin + th * (temin - tmin)
            cand = [(e, t, a) for e, t, a in FB if t <= T + 1e-7]
            if not cand:
                continue
            e, t, a = min(cand, key=lambda x: x[0])
            mk = 0.0
            for plan in a:
                td = 0.0
                for i, (sub, _, _, order) in enumerate(plan):
                    S = tuple(j for j in range(len(R)) if sub >> j & 1)
                    r = eval_trip_order(leg, R, TYPE_M600, rb.TM, S, list(order))
                    if r is None:
                        td = float("nan"); break
                    td += r[1] + (rb.TM.tau if i else 0.0)
                if td == td:
                    mk = max(mk, td)
            assert mk <= T + 1e-6, (th, mk, T)


def test_baseline_regret_nonnegative_in_stored_results():
    import csv
    p = os.path.join(ROOT, "04_Results", "baselines_rows.csv")
    if not os.path.exists(p):
        pytest.skip("no stored baselines")
    rows = [r for r in csv.DictReader(open(p, encoding="utf-8")) if r["status"] == "ok"]
    assert rows and all(float(r["regret"]) >= -1e-9 for r in rows)
