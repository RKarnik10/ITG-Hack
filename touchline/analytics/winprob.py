"""Live win probability.

Remaining goals for each side are Poisson; the per-minute scoring rate is a
shrinkage blend of the pre-match expectation and the xG the side has actually
created so far. The joint scoreline grid (Dixon-Coles optional) is shifted by
the current score. Adapted from WC Predictor/scoreline.py (score_matrix).
"""
from __future__ import annotations

from math import exp, lgamma, log

import numpy as np

MAX_GOALS = 10
MATCH_MIN = 95.0
PRIOR_WEIGHT = 0.6  # how many "matches" of prior the live rate is shrunk toward


def _pmf(lam: float) -> np.ndarray:
    k = np.arange(MAX_GOALS + 1)
    lam = max(lam, 1e-9)
    return np.exp(k * log(lam) - lam - np.array([lgamma(i + 1) for i in k]))


def score_matrix(lam: float, mu: float, rho: float = 0.0) -> np.ndarray:
    """Joint scoreline probabilities. m[i, j] = P(home scores i, away scores j)."""
    m = np.outer(_pmf(lam), _pmf(mu))
    if rho:
        t = np.ones_like(m)
        t[0, 0] = 1.0 - lam * mu * rho
        t[0, 1] = 1.0 + lam * rho
        t[1, 0] = 1.0 + mu * rho
        t[1, 1] = 1.0 - rho
        m = np.clip(m * t, 1e-15, None)
    return m / m.sum()


def prematch_rates(q_home: float, q_away: float) -> tuple[float, float]:
    """Expected goals from club quality (home advantage folded in), fit to the engine."""
    r = (q_home * 1.05) / q_away
    return 1.55 * r ** 1.5, 1.55 / r ** 1.5


def outcome_probs(lam_rem: float, mu_rem: float, diff: int) -> tuple[float, float, float]:
    m = score_matrix(lam_rem, mu_rem)
    i, j = np.indices(m.shape)
    final = i - j + diff
    return float(m[final > 0].sum()), float(m[final == 0].sum()), float(m[final < 0].sum())


def live_rates(lam0: float, mu0: float, minute: float, xg_h: float, xg_a: float,
               red_h: int = 0, red_a: int = 0) -> tuple[float, float]:
    """Remaining expected goals for each side at `minute`."""
    played = min(minute, MATCH_MIN - 0.5)
    rem = max(MATCH_MIN - played, 0.5)
    w = PRIOR_WEIGHT * MATCH_MIN
    rate_h = (lam0 / MATCH_MIN * w + xg_h) / (w + played)
    rate_a = (mu0 / MATCH_MIN * w + xg_a) / (w + played)
    # a red card costs the short side ~35% of its rate and gifts the other ~25%
    rate_h *= 0.65 ** red_h * 1.25 ** red_a
    rate_a *= 0.65 ** red_a * 1.25 ** red_h
    return rate_h * rem, rate_a * rem


def game_state(lam: float, mu: float, diff: int) -> tuple[float, float]:
    """Trailing sides push (+18%), leaders sit deeper (-8%)."""
    if diff > 0:
        return lam * 0.92, mu * 1.18
    if diff < 0:
        return lam * 1.18, mu * 0.92
    return lam, mu


def timeline(df, home: str, lam0: float, mu0: float) -> np.ndarray:
    """Home win/draw/away probability *after* each event, shape (n_events, 3)."""
    out = np.empty((len(df), 3))
    xg_h = xg_a = 0.0
    red_h = red_a = 0
    shot_xg = df["xg"].fillna(0.0).to_numpy() if "xg" in df else np.zeros(len(df))
    team = df["team"].to_numpy()
    typ = df["type"].to_numpy()
    outcome = df["outcome"].to_numpy()
    tmin = df["elapsed"].to_numpy() / 60.0  # continuous clock across both halves
    sh, sa = df["score_home"].to_numpy(), df["score_away"].to_numpy()
    for k in range(len(df)):
        if typ[k] == "Shot":
            if team[k] == home:
                xg_h += shot_xg[k]
            else:
                xg_a += shot_xg[k]
        elif typ[k] == "Card" and outcome[k] in ("Red Card", "Second Yellow"):
            if team[k] == home:
                red_h += 1
            else:
                red_a += 1
        lam, mu = live_rates(lam0, mu0, tmin[k], xg_h, xg_a, red_h, red_a)
        diff = int(sh[k] - sa[k])
        out[k] = outcome_probs(*game_state(lam, mu, diff), diff)
    return out
