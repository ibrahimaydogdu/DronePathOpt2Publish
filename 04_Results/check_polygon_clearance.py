"""Exact-polygon check of the raster leg model.
Part A (clearance): for every leg between the nodes of a real instance whose minimum-energy option at 1 kg is an 'around' path, the path is traced again on the
raster (same blocked mask as core_d1.around.around_options) and its distance to the exact footprint polygons that block at the cruise altitude h (top + delta > h,
end-node host buildings exempt when top <= h) is computed with shapely. Parts of the path within 15 m of an end node are left out (the model relaxes the margin there).
Part B (energy): the minimum energy E_min, the shortest makespan at four drones and the premium of the fastest plan are recomputed on rasters of 5 m, 10 m (base) and
20 m and on the 10 m raster with the polygon vertices shifted by half a cell (RASTER_SHIFT), for n = 8.
Usage (from 03_Code_Execution): python -I ../04_Results/check_polygon_clearance.py <draws>   Output: polygon_clearance_rows.csv, polygon_energy_rows.csv, POLYGON_CHECK_SUMMARY_D2.txt"""
import sys, os, csv, math, statistics as st
import numpy as np
from scipy import ndimage
from skimage.graph import MCP_Geometric
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution")); sys.path.insert(0, os.path.join(HERE, "..", "03_Code_Execution", "core_d1"))
from drone3d_d2.realcity import make_real_instance, height_la, height_chicago, height_austin
from drone3d_d2.fleet import Req, trip_table, one_drone_table, fleet_frontier
from drone3d_d2.dtypes import TYPE_M600, TimeModel
from drone3d_d2.legs import CityLeg
from legs import LegCost
from around import CityRaster, draw_polygon, _sh
from energy import segment_energy_wh
TM = TimeModel(s_service=60.0, tau=180.0)
ARMS = [("LA", "DLA4", height_la), ("CHI", "DCH1", height_chicago(3.5, 0)), ("AUS", "DAU1", height_austin(0.0))]
DELTA, W = 10.0, 5.0


def trace(city, raster, a, b, h):
    """Re-trace the around path at cruise altitude h; returns list of (x, y) cell centres or None."""
    res = raster.res; L = math.hypot(b.x - a.x, b.y - a.y); margin = 0.35 * L + 60.0
    ra, ca = raster.cell(a.x, a.y); rb, cb = raster.cell(b.x, b.y)
    r0 = max(min(ra, rb) - int(margin / res), 0); r1 = min(max(ra, rb) + int(margin / res) + 1, raster.shape[0])
    c0 = max(min(ca, cb) - int(margin / res), 0); c1 = min(max(ca, cb) + int(margin / res) + 1, raster.shape[1])
    win = raster.top[r0:r1, c0:c1]
    blocked = (win + DELTA) > h
    for nd in (a, b):
        if nd.host >= 0 and nd.host in range(len(city.geoms)) and city.top[nd.host] <= h:
            g = city.geoms[nd.host]
            for poly in (g.geoms if g.geom_type == "MultiPolygon" else [g]):
                cx = (np.array(poly.exterior.coords)[:, 0] - raster.x0) / res - c0 - _sh()
                cy = (np.array(poly.exterior.coords)[:, 1] - raster.y0) / res - r0 - _sh()
                rr, cc = draw_polygon(cy, cx, blocked.shape); blocked[rr, cc] = False
    raw = blocked.copy()
    dil = max(int(math.ceil(W / res)), 1)
    blocked = ndimage.binary_dilation(blocked, structure=ndimage.generate_binary_structure(2, 2), iterations=dil)
    for (r, c) in ((ra - r0, ca - c0), (rb - r0, cb - c0)):
        sq = (slice(max(r - 2, 0), r + 3), slice(max(c - 2, 0), c + 3)); blocked[sq] = raw[sq]
        blocked[min(max(r, 0), blocked.shape[0] - 1), min(max(c, 0), blocked.shape[1] - 1)] = False
    m = MCP_Geometric(np.where(blocked, np.inf, 1.0), fully_connected=True)
    cum, _ = m.find_costs([(ra - r0, ca - c0)], [(rb - r0, cb - c0)])
    if not np.isfinite(cum[rb - r0, cb - c0]): return None
    path = m.traceback((rb - r0, cb - c0))
    return [(raster.x0 + (c0 + c + 0.5) * res, raster.y0 + (r0 + r + 0.5) * res) for r, c in path]


def part_a(name, st_, hf, seed, rows):
    city, nodes, reqs, meta = make_real_instance(st_, 8, seed, 3000.0, 100.0, hf)
    ras = CityRaster(city, res=10.0); lc = LegCost(city, nodes, model="F", hw=TYPE_M600.hw, v=10.0, raster=ras)
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i == j: continue
            best = (float("inf"), None, None)
            for kind, segs in lc.options(i, j):
                e = sum(segment_energy_wh("F", ell, dz, 1.0, TYPE_M600.hw, 10.0, 1.0) for ell, dz in segs)
                if e < best[0]: best = (e, kind, segs)
            if best[1] != "around": continue
            a, b = nodes[i], nodes[j]; h = a.z + best[2][0][1]
            pts = trace(city, ras, a, b, h)
            if pts is None or len(pts) < 3: continue
            keep = [p for p in pts if math.hypot(p[0] - a.x, p[1] - a.y) > 15 and math.hypot(p[0] - b.x, p[1] - b.y) > 15]
            if len(keep) < 2: continue
            blockers = [city.geoms[k] for k in city.obs_idx if city.top[k] + DELTA > h and not ((a.host == k or b.host == k) and city.top[k] <= h)]
            near = [g for g in blockers if g.distance(Point(a.x, a.y)) < 3000]
            line = LineString(keep)
            d = min((g.distance(line) for g in near), default=float("inf"))
            rows.append(dict(arm=name, seed=seed, i=i, j=j, h=round(h, 1), path_m=round(line.length, 1), clearance_m=round(d, 2), inside=int(d <= 1e-9)))


def part_b(name, st_, hf, seed, rows):
    out = {}
    for cfg, res, shift, w in (("5m", 5.0, "0", 5.0), ("10m", 10.0, "0", 5.0), ("20m", 20.0, "0", 5.0), ("10m_shift", 10.0, "1", 5.0), ("10m_w15", 10.0, "0", 15.0)):
        os.environ["RASTER_SHIFT"] = shift
        city, nodes, reqs, meta = make_real_instance(st_, 8, seed, 3000.0, 100.0, hf)
        leg = CityLeg(LegCost(city, nodes, model="F", hw=TYPE_M600.hw, v=10.0, w=w, raster=CityRaster(city, res=res)), TYPE_M600)
        R = [Req(*r) for r in reqs]; h = one_drone_table(trip_table(leg, R, TYPE_M600, TM), 8, TM)
        F4 = fleet_frontier([h] * 4, 8)
        out[cfg] = (F4[0][0], F4[-1][1], F4[-1][0] / F4[0][0] - 1)
    os.environ["RASTER_SHIFT"] = "0"
    for cfg, (e, t, p) in out.items():
        rows.append(dict(arm=name, seed=seed, cfg=cfg, E_min=round(e, 3), Tmin_k4=round(t, 1), premium=round(p, 5),
                         dE_vs_10m=round(e / out["10m"][0] - 1, 5), dT_vs_10m=round(t / out["10m"][1] - 1, 5), dprem_pts=round(100 * (p - out["10m"][2]), 3)))


if __name__ == "__main__":
    draws = int(sys.argv[1]); A, B = [], []; sel = os.environ.get("ARM", ""); suf = ("_" + sel) if sel else ""
    for name, st_, hf in ARMS:
        if sel and name != sel: continue
        for seed in range(1001, 1001 + draws):
            part_a(name, st_, hf, seed, A); part_b(name, st_, hf, seed, B); print(name, seed, len(A), flush=True)
    for fn, rows in ((f"polygon_clearance_rows{suf}.csv", A), (f"polygon_energy_rows{suf}.csv", B)):
        with open(os.path.join(HERE, fn), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    out = []
    for name, _, _ in ARMS:
        if sel and name != sel: continue
        g = [r for r in A if r["arm"] == name]
        if not g: continue
        d = [r["clearance_m"] for r in g]
        out.append(f"{name}: around legs checked {len(g)}; path inside a blocking footprint {sum(r['inside'] for r in g)}; clearance median {st.median(d):.1f} m, 10th pct {np.percentile(d, 10):.1f} m, min {min(d):.1f} m; < 5 m: {sum(x < 5 for x in d)}")
    for cfg in ("5m", "20m", "10m_shift", "10m_w15"):
        g = [r for r in B if r["cfg"] == cfg]
        de = [100 * r["dE_vs_10m"] for r in g]; dt = [100 * r["dT_vs_10m"] for r in g]; dp = [r["dprem_pts"] for r in g]
        out.append(f"raster {cfg} vs 10 m ({len(g)} instances): E_min change median {st.median(de):+.2f} % (range {min(de):+.2f} to {max(de):+.2f}); T_min(k=4) change median {st.median(dt):+.2f} % (range {min(dt):+.2f} to {max(dt):+.2f}); premium change median {st.median(dp):+.2f} points (range {min(dp):+.2f} to {max(dp):+.2f})")
    open(os.path.join(HERE, f"POLYGON_CHECK_SUMMARY{suf}_D2.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n"); print("\n".join(out))
