"""Leg models returning (energy Wh, time s) for a drone type and a carried load.

PlanarLeg: synthetic 3D points (labelled SYNTHETIC). CityLeg: adapter over D1 LegCost (real city geometry,
used in Phase B); time of a leg = time of its minimum-energy option (documented simplification).
"""
from __future__ import annotations
import math
from ._core import segment_energy_wh
from .dtypes import DroneType


def seg_time(ell: float, dz: float, hw, v: float) -> float:
    return max(ell / v, max(dz, 0.0) / hw.v_up, max(-dz, 0.0) / hw.v_down)


class PlanarLeg:
    def __init__(self, pts, dt: DroneType, model: str = "F"):
        self.pts, self.dt, self.model = pts, dt, model
        self._c = {}

    def et(self, i: int, j: int, load: float):
        if i == j:
            return 0.0, 0.0
        key = (i, j, round(load, 9))
        r = self._c.get(key)
        if r is None:
            (x1, y1, z1), (x2, y2, z2) = self.pts[i], self.pts[j]
            ell, dz = math.hypot(x2 - x1, y2 - y1), z2 - z1
            e = segment_energy_wh(self.model, ell, dz, load, self.dt.hw, self.dt.v, self.dt.kappa())
            r = (e, seg_time(ell, dz, self.dt.hw, self.dt.v))
            self._c[key] = r
        return r


class CityLeg:
    """Adapter over core_d1.legs.LegCost: energy = D1 minimum over geometric options; time of that option."""
    def __init__(self, legcost, dt: DroneType):
        self.lc, self.dt = legcost, dt
        self._c = {}

    def et(self, i: int, j: int, load: float):
        if i == j:
            return 0.0, 0.0
        key = (i, j, round(load, 9))
        r = self._c.get(key)
        if r is None:
            r = self._et(i, j, load)
            self._c[key] = r
        return r

    def _et(self, i: int, j: int, load: float):
        best, bt = float("inf"), float("inf")
        for _, segs in self.lc.options(i, j):
            e = t = 0.0
            for ell, dz in segs:
                e += segment_energy_wh(self.lc.model, ell, dz, load, self.dt.hw, self.dt.v, self.dt.kappa())
                t += seg_time(ell, dz, self.dt.hw, self.dt.v)
            if e < best:
                best, bt = e, t
        return best, bt
