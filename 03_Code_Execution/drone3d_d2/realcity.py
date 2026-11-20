"""Real-demand + real-building instances for D2 (Phase B).

Demand: Amazon Last Mile Routing Research Challenge evaluation subset (Evidence/data_realdemand/; CC BY-NC 4.0,
Merchan et al., Transportation Science 58(1), 2024). Every stop is a delivery (type 'Dropoff') from the station (depot):
all requests are kind D. Stops are pooled from all routes of one station and n of them are drawn at random within a
radius of the station (the truck routes themselves span tens of km and are not used as routes).
Buildings: Evidence/data_buildings/ (Chicago syp8-uezg; Los Angeles LARIAC4). Terrain is taken as flat (ground = 0 m, no ground
elevation in either source): this is a stated simplification.
Sensitivity parameters (UNSOURCED, reported only as sensitivity): package density rho (kg/m3) turns package volume into mass;
metres per storey and the treatment of unknown storeys (Chicago).
"""
from __future__ import annotations
import json
import math
import os
import random
import sys

import numpy as np
from shapely.geometry import MultiPolygon, Polygon, shape

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "core_d1"))
from geometry import City, Node, Projection, make_node, FT   # noqa: E402

EV = os.path.join(ROOT, "Evidence")
BLD = {"chicago": os.path.join(EV, "data_buildings", "chicago", "chicago_buildings.geojson"),
       "los_angeles": os.path.join(EV, "data_buildings", "los_angeles", "la_buildings.geojson"),
       "austin": os.path.join(EV, "data_buildings", "austin", "dau1_buildings_raw.geojson")}
STATIONS = {"DCH1": "chicago", "DCH2": "chicago", "DLA3": "los_angeles", "DLA4": "los_angeles", "DLA8": "los_angeles", "DAU1": "austin"}
_G = {}


def _demand():
    if "d" not in _G:
        _G["d"] = json.load(open(os.path.join(EV, "data_realdemand", "eval_route_data.json"), encoding="utf-8"))
        _G["p"] = json.load(open(os.path.join(EV, "data_realdemand", "eval_package_data.json"), encoding="utf-8"))
    return _G["d"], _G["p"]


def station_data(code: str):
    """Station coordinates and the pooled stops (id, lat, lng, volume_cm3, service_s) of one station."""
    d, p = _demand()
    st, stops = None, {}
    for rid, r in d.items():
        if r["station_code"] != code:
            continue
        for sid, s in r["stops"].items():
            if s["type"] == "Station":
                st = (s["lat"], s["lng"])
                continue
            pk = p[rid].get(sid, {})
            vol = sum(x["dimensions"]["depth_cm"] * x["dimensions"]["height_cm"] * x["dimensions"]["width_cm"] for x in pk.values())
            svc = sum(x["planned_service_time_seconds"] for x in pk.values())
            stops[(rid, sid)] = (s["lat"], s["lng"], vol, svc)
    return st, stops


def load_city_geojson(path: str, lat0: float, lon0: float, half: float, height_fn, bbox_pad_deg: float = 0.0):
    """City (core_d1 dataclass) from a footprint GeoJSON, keeping buildings whose centroid lies within `half` m of (lat0, lon0).
    height_fn(properties) -> height above ground in m (NaN/0 = unknown, treated as no obstacle)."""
    proj = Projection(lat0, lon0)
    key = (path, round(lat0, 4), round(lon0, 4), half, getattr(height_fn, 'tag', 'x'))
    if key in _G:
        return _G[key]
    dlat = math.degrees(half / 6371008.8) * 1.05
    dlon = dlat / math.cos(math.radians(lat0))
    gj = json.load(open(path, encoding="utf-8"))
    geoms, height, ids = [], [], []
    for f in gj["features"]:
        g = f.get("geometry")
        if not g:
            continue
        shp = shape(g)
        c = shp.centroid
        if abs(c.y - lat0) > dlat or abs(c.x - lon0) > dlon:
            continue
        polys = []
        for q in (shp.geoms if isinstance(shp, MultiPolygon) else [shp]):
            if q.geom_type != "Polygon":
                continue
            xs, ys = proj.fwd([a[0] for a in q.exterior.coords], [a[1] for a in q.exterior.coords])
            pp = Polygon(list(zip(xs, ys))).buffer(0)
            if not pp.is_empty and pp.geom_type == "Polygon":
                polys.append(pp)
        if not polys:
            continue
        geoms.append(MultiPolygon(polys) if len(polys) > 1 else polys[0])
        height.append(height_fn(f["properties"]))
        ids.append(f["properties"].get("bldg_id") or f["properties"].get("OBJECTID"))
    height = np.array(height, dtype=float)
    ground = np.zeros(len(geoms))
    obstacle = np.nan_to_num(height, nan=0.0) > 0
    top = ground + np.nan_to_num(height, nan=0.0)
    city = City(geoms, top, ground, height, obstacle, ids, [None] * len(geoms), [None] * len(geoms), proj)
    _G[key] = city
    return city


def height_la(props):
    h = props.get("HEIGHT")
    return float(h) * FT if h not in (None, "") and float(h) > 0 else float("nan")


height_la.tag = "la"


height_la_tag = None


def height_austin(fill_m: float = 0.0):
    """Austin 'max_height' in feet (ELEVATION - BASE_ELEVATION); -9999999999 = unknown -> fill_m metres (0 = no obstacle)."""
    def f(props):
        h = props.get("max_height")
        try:
            h = float(h)
        except (TypeError, ValueError):
            h = -1.0
        if h > 0:
            return h * FT
        return fill_m if fill_m > 0 else float("nan")
    f.tag = ("aus", fill_m)
    return f


def height_chicago(m_per_storey: float, fill_storeys: float):
    """stories > 0: stories * m_per_storey. stories == 0 (unknown): fill_storeys * m_per_storey (0 -> no obstacle)."""
    def f(props):
        s = props.get("stories")
        s = float(s) if s not in (None, "") else 0.0
        if s <= 0:
            s = fill_storeys
        return s * m_per_storey if s > 0 else float("nan")
    f.tag = (m_per_storey, fill_storeys)
    return f


def make_real_instance(code: str, n: int, seed: int, radius: float, rho: float, hfn, max_snap: float = 30.0, mass_fn=None):
    """Returns (city, nodes, reqs_as_tuples(kind,o,d,q), meta). Stops snap to the nearest footprint (<= max_snap m), roof node."""
    from scipy.spatial import cKDTree
    from shapely.geometry import Point
    st, stops = station_data(code)
    lat0, lon0 = st
    city = load_city_geojson(BLD[STATIONS[code]], lat0, lon0, radius + 1500.0, hfn)
    xs, ys = city.proj.fwd(lon0, lat0)
    cen = np.array([[g.centroid.x, g.centroid.y] for g in city.geoms])
    kd = cKDTree(cen)
    pool = []
    n_heavy = 0
    from drone3d_d2.dtypes import TYPE_M600
    qmax = TYPE_M600.hw.q_max
    for (rid, sid), (la, lo, vol, svc) in stops.items():
        x, y = city.proj.fwd(lo, la)
        if math.hypot(x - xs, y - ys) > radius:
            continue
        if mass_fn is None and vol * 1e-6 * rho > qmax:   # not carryable by this drone: not drone-eligible (share reported)
            n_heavy += 1
            continue
        _, idx = kd.query([x, y], k=5)
        dd = [(city.geoms[int(i)].distance(Point(x, y)), int(i)) for i in idx]
        dist, b = min(dd)
        if dist > max_snap:
            continue
        pool.append((int(b), vol, svc))
    rng = random.Random(seed)
    rng.shuffle(pool)
    chosen, used = [], set()
    for b, vol, svc in pool:
        if b in used:
            continue
        used.add(b); chosen.append((b, vol, svc))
        if len(chosen) == n:
            break
    if len(chosen) < n:
        raise ValueError("pool too small")
    # depot: roof of the host footprint if inside one, else open ground at 5 m
    pt = Point(float(xs), float(ys))
    host = [i for i in city.tree.query(pt, predicate="intersects")]
    if host:
        depot = make_node(city, int(city.obs_idx[host[0]]), lam=1.0, kind="depot")
    else:
        depot = Node(float(xs), float(ys), 5.0, -1, "depot")
    nodes = [depot]
    reqs = []
    qcap = qmax
    mrng = random.Random(seed + 50000)
    for b, vol, svc in chosen:
        if mass_fn is None:
            q = max(0.1, round(vol * 1e-6 * rho, 2))
        else:                      # mass drawn from a given distribution (calibration arm); resampled until carryable
            q = qcap + 1.0
            while q > qcap:
                q = max(0.1, round(mass_fn(mrng), 2))
        nodes.append(make_node(city, b, lam=1.0))
        reqs.append(("D", 0, len(nodes) - 1, q))
    meta = dict(station=code, n=n, seed=seed, radius=radius, rho=rho, pool=len(pool), heavy_excluded=n_heavy, mean_q=float(np.mean([r[3] for r in reqs])),
                max_q=float(max(r[3] for r in reqs)), unknown_height_hosts=int(sum(1 for b, _, _ in chosen if not city.obstacle[b])))
    return city, nodes, reqs, meta
