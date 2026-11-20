"""Drone types and time-model parameters for D2 (evidence: 02_Methods/TIME_AND_RESERVE_EVIDENCE.md).

M600 Pro = D1 default Hardware() (manual p.44-45). FlyCart 30 values from the manufacturer manual
(Evidence/DJI_FlyCart_30_User_Manual_v1.1_0709.pdf): 65 kg with two DB2000, MTOW 95 kg, 2 x 1984.4 Wh,
hover 29 min empty / 18 min with 30 kg, v_up 5 m/s, v_down 3 m/s, 8 rotors of 54 in (1.3716 m).
Reserve 10 % for both (base E for the power derivation; see evidence file section 3.3).
"""
from __future__ import annotations
from dataclasses import dataclass
from ._core import Hardware

M600 = Hardware()
# Hover power base E (evidence file 3.3): manufacturer endurance assumed measured to 0 % battery, so P = E/t.
# Hardware derives P = e_use/t with e_use = 0.9 E, hence t0, t1 are scaled by 0.9; usable energy per trip stays 0.9 E.
FLYCART = Hardware(m0=65.0, mtow=95.0, e_batt_wh=3968.8, t0_min=29.0 * 0.9, t1_min=18.0 * 0.9, reserve=0.10,
                   v_up=5.0, v_down=3.0, rotors=8, prop_d=1.3716, eta_h=0.50)
# Variant (sensitivity): power base 0.9 E (as for M600, where the manual states the 10 % cut-off).
FLYCART_P09 = Hardware(m0=65.0, mtow=95.0, e_batt_wh=3968.8, t0_min=29.0, t1_min=18.0, reserve=0.10,
                       v_up=5.0, v_down=3.0, rotors=8, prop_d=1.3716, eta_h=0.50)


@dataclass(frozen=True)
class DroneType:
    name: str
    hw: Hardware
    v: float = 10.0          # horizontal cruise speed, m/s
    kappa_c: float = 1.0     # cruise power / hover power at v_ref
    ind_share: float | None = None   # None: constant kappa_c. Else speed-dependent cruise power factor
    v_ref: float = 10.0      # kappa(v) = kappa_c * (a*(v_ref/v) + (1-a)*(v/v_ref)**3), a = ind_share (induced vs parasitic split)

    def kappa(self, v: float | None = None) -> float:
        v = self.v if v is None else v
        if self.ind_share is None:
            return self.kappa_c
        a = self.ind_share
        return self.kappa_c * (a * (self.v_ref / v) + (1.0 - a) * (v / self.v_ref) ** 3)


@dataclass(frozen=True)
class TimeModel:
    s_service: float = 60.0   # s per service event, drone hovers (hover energy charged)
    tau: float = 180.0        # s depot turnaround between consecutive trips of one drone (not after the last)
    service_hover: bool = True  # False: drone lands, no service energy


TYPE_M600 = DroneType("M600", M600)
TYPE_FLYCART = DroneType("FlyCart30", FLYCART)
