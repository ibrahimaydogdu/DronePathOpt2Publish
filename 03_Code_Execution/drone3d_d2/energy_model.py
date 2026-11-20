"""Self-contained energy model of the paper (Section 3.2), written for the public release: hover power fitted to the manufacturer's hover times and the
segment energy with the potential-energy climb term (model F), the obstacle-free variant without a climb term (D) and the flat variant (2D).
Units: m, s, kg, W; energies in Wh."""
from __future__ import annotations
from dataclasses import dataclass
import math

G = 9.81
RHO = 1.225


@dataclass(frozen=True)
class Hardware:
    m0: float = 9.5            # empty mass, kg
    mtow: float = 15.5         # maximum take-off mass, kg
    e_batt_wh: float = 599.4   # 6 x 99.9 Wh
    t0_min: float = 32.0       # hover time, no payload (manufacturer)
    t1_min: float = 16.0       # hover time, 6 kg payload (manufacturer)
    reserve: float = 0.10      # hover times quoted down to 10 % battery
    v_up: float = 5.0
    v_down: float = 3.0
    rotors: int = 6
    prop_d: float = 0.533
    eta_h: float = 0.50

    @property
    def q_max(self): return self.mtow - self.m0
    @property
    def e_use_wh(self): return (1.0 - self.reserve) * self.e_batt_wh
    @property
    def p0(self): return self.e_use_wh / (self.t0_min / 60.0)
    @property
    def p1(self): return self.e_use_wh / (self.t1_min / 60.0)
    @property
    def alpha(self): return math.log(self.p1 / self.p0) / math.log(self.mtow / self.m0)
    @property
    def k(self): return self.p0 / self.m0 ** self.alpha
    @property
    def disk_area(self): return self.rotors * math.pi * (self.prop_d / 2.0) ** 2

    def hover_power(self, load_kg: float) -> float:
        return self.k * (self.m0 + load_kg) ** self.alpha


def segment_energy_wh(model: str, ell: float, dz: float, load_kg: float, hw: Hardware = Hardware(), v: float = 10.0, kappa_c: float = 1.0) -> float:
    """Energy of one straight segment with horizontal length ell (m) and altitude change dz (m)."""
    if model not in ("F", "D", "2D"):
        raise ValueError("model must be F, D or 2D in this release")
    if ell < 0 or not 0.0 <= load_kg <= hw.q_max + 1e-9:
        raise ValueError("invalid segment")
    if model == "2D":
        dz = 0.0
    up, down = max(dz, 0.0), max(-dz, 0.0)
    t = max(ell / v, up / hw.v_up, down / hw.v_down)
    e_j = kappa_c * hw.hover_power(load_kg) * t
    if model == "F":
        e_j += (hw.m0 + load_kg) * G * up / hw.eta_h
    return e_j / 3600.0
