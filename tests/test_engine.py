"""Calibration tests: the synthetic engine must stay football-realistic.

Reference ranges are typical top-flight per-match values (both teams unless noted).
"""
import itertools

import numpy as np
import pandas as pd
import pytest

from touchline.sim.engine import simulate_match, to_display, to_statsbomb
from touchline.sim.teams import CLUBS


@pytest.fixture(scope="module")
def season():
    pairs = list(itertools.permutations(CLUBS, 2))
    rows, frames = [], []
    for i in range(120):
        home, away = pairs[i % len(pairs)]
        df, meta = simulate_match(home, away, seed=500 + i)
        frames.append(df)
        shots, passes = df[df.type == "Shot"], df[df.type == "Pass"]
        rows.append({
            "goals": sum(meta["score"]), "home_goals": meta["score"][0], "away_goals": meta["score"][1],
            "shots_per_team": len(shots) / 2, "xg": shots.xg.sum(), "events": len(df),
            "completion": (passes.outcome == "Complete").mean(),
            "fouls": (df.type == "Foul Committed").sum(), "corners": (df.set_piece == "Corner").sum(),
            "xg_per_shot": shots.xg.mean(),
        })
    return pd.DataFrame(rows), frames


@pytest.mark.parametrize("col,lo,hi", [
    ("goals", 2.4, 3.6),
    ("shots_per_team", 10, 16),
    ("completion", 0.75, 0.88),
    ("events", 1800, 2600),
    ("fouls", 15, 30),
    ("corners", 6, 13),
    ("xg_per_shot", 0.08, 0.14),
])
def test_per_match_means_are_realistic(season, col, lo, hi):
    stats, _ = season
    assert lo <= stats[col].mean() <= hi, f"{col} mean {stats[col].mean():.3f} outside [{lo}, {hi}]"


def test_xg_tracks_goals(season):
    stats, _ = season
    assert abs(stats.xg.sum() - stats.goals.sum()) / stats.goals.sum() < 0.15


def test_home_advantage_exists(season):
    stats, _ = season
    assert stats.home_goals.mean() > stats.away_goals.mean()


def test_same_seed_same_match():
    a, _ = simulate_match("KBY", "IRV", seed=7)
    b, _ = simulate_match("KBY", "IRV", seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_coordinates_on_pitch(season):
    _, frames = season
    df = pd.concat(frames)
    assert df.x.between(0, 120).all() and df.y.between(0, 80).all()
    assert df.end_x.dropna().between(0, 120).all() and df.end_y.dropna().between(0, 80).all()


def test_score_columns_are_monotonic(season):
    _, frames = season
    for df in frames[:10]:
        assert df.score_home.is_monotonic_increasing and df.score_away.is_monotonic_increasing


def test_views():
    df, meta = simulate_match("SCR", "EMP", seed=3)
    disp = to_display(df, meta["home"])
    away = df.team == meta["away"]
    assert np.allclose(disp.loc[away, "x"], 120 - df.loc[away, "x"])
    sb = to_statsbomb(df)
    assert {"location", "pass_end_location", "pass_outcome", "shot_outcome"} <= set(sb.columns)


def test_store_verifier_catches_bad_numbers():
    from touchline.analytics.store import MatchStore
    df, meta = simulate_match("KBY", "NMA", seed=2031)
    store = MatchStore(df, meta)
    fid = store.f_team["KBY"]["xg"]
    good = f"Kestrel Bay created {store.facts[fid].show()} xG [{fid}]."
    bad = f"Kestrel Bay created 9.9 xG [{fid}] and won 7-0."
    assert store.verify(good)["ok"]
    report = store.verify(bad)
    assert not report["ok"]
    assert any("never occurred" in i["problem"] for i in report["issues"])


def test_real_time_cut_only_sees_events_so_far():
    from touchline.analytics.store import MatchStore, cut_match
    df, meta = simulate_match("SCR", "DSK", seed=3079)
    d, m = cut_match(df, meta, "Half-time")
    assert (d.period == 1).all() and m["live"] == "Half-time" and m["match_id"].endswith("@Halftime")
    store = MatchStore(d, m)
    probs = [store.facts[store.f_live[k]].value for k in ("wp_home", "wp_draw", "wp_away")]
    assert abs(sum(probs) - 100) < 1.5
    assert "STATUS: LIVE" in store.summary_text()
    full = MatchStore(df, meta)
    assert not full.live and "STATUS: full time" in full.summary_text()


def test_fan_template_localised_and_verified():
    from touchline.agents import fallback
    from touchline.analytics.store import MatchStore, cut_match
    df, meta = simulate_match("SCR", "DSK", seed=3079)
    for moment in ("Full time", "Half-time"):
        store = MatchStore(*cut_match(df, meta, moment))
        for lang in fallback.FAN_TEXT:
            text = fallback.fan(store, {"team": "DSK", "expertise": "beginner", "language": lang})
            assert store.verify(text)["ok"], (moment, lang)
    store = MatchStore(df, meta)
    assert "beat" not in fallback.fan(store, {"team": "DSK", "expertise": "expert", "language": "Spanish"})
