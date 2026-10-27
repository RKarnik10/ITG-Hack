"""Expected Threat valuation of ball progression.

Uses the 18x12 xT surface from the sibling xT project (Karun Singh's model).
Zone lookup is the vectorized `_zone_vec` from xT/xt_model.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..sim.engine import N_X, N_Y, PITCH_X, PITCH_Y, PRIORS

XT = PRIORS["xt"]


def zone_vec(x, y, n_x: int = N_X, n_y: int = N_Y) -> np.ndarray:
    xi = np.clip((np.asarray(x) / PITCH_X * n_x).astype(int), 0, n_x - 1)
    yi = np.clip((np.asarray(y) / PITCH_Y * n_y).astype(int), 0, n_y - 1)
    return yi * n_x + xi


def add_xt(df: pd.DataFrame) -> pd.DataFrame:
    """Adds `xt_added` (end-zone xT minus start-zone xT) to completed passes and carries."""
    df = df.copy()
    move = ((df["type"] == "Pass") & (df["outcome"] == "Complete")) | (df["type"] == "Carry")
    df["xt_added"] = 0.0
    sub = df[move]
    df.loc[move, "xt_added"] = (XT[zone_vec(sub.end_x.values, sub.end_y.values)]
                                - XT[zone_vec(sub.x.values, sub.y.values)])
    return df


def xt_at(x: float, y: float) -> float:
    return float(XT[zone_vec([x], [y])[0]])
