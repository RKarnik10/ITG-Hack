"""Match analytics: possessions, momentum, patterns, player ratings, turning points.

Everything here is deterministic and explainable: each output carries the event
ids it was computed from so the agents (and the UI) can show their evidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .winprob import game_state, live_rates, outcome_probs, prematch_rates, timeline
from .xt import add_xt

DEF_ACTIONS = ("Ball Recovery", "Interception", "Tackle", "Clearance")


def clock(period: int, t: float) -> str:
    """Broadcast clock: 45+2', 90+4'."""
    m = int(t // 60) + 1
    if period == 1 and m > 45:
        return f"45+{m - 45}'"
    if period == 2 and m > 90:
        return f"90+{m - 90}'"
    return f"{m}'"


def prepare(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Adds elapsed clock, broadcast clock, xT added and shared-pitch coordinates."""
    df = add_xt(df)
    stoppage1 = int(meta["stoppage"][1] if 1 in meta["stoppage"] else meta["stoppage"]["1"])
    df["elapsed"] = df["t"] + np.where(df["period"] == 2, stoppage1 * 60, 0)
    df["clock"] = [clock(p, t) for p, t in zip(df["period"], df["t"])]
    away = df["team"] != meta["home"]
    for a, b in (("x", "y"), ("end_x", "end_y")):
        df["d" + a] = np.where(away, 120 - df[a], df[a])
        df["d" + b] = np.where(away, 80 - df[b], df[b])
    return df


def momentum(df: pd.DataFrame, home: str, span: int = 6) -> pd.DataFrame:
    """Per-minute threat (positive xT added + xG) for each side and a smoothed differential."""
    minute = (df["elapsed"] // 60).astype(int)
    threat = df["xt_added"].clip(lower=0) + df["xg"].fillna(0)
    g = pd.DataFrame({"minute": minute, "home": np.where(df.team == home, threat, 0.0),
                      "away": np.where(df.team != home, threat, 0.0)}).groupby("minute").sum()
    g = g.reindex(range(int(minute.max()) + 1), fill_value=0.0)
    g["diff"] = g["home"] - g["away"]
    g["smooth"] = g["diff"].ewm(span=span).mean()
    return g


def possessions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, p in df[df.possession_id > 0].groupby("possession_id", sort=True):
        p = p[p.team == p.possession_team.iloc[0]]
        if p.empty:
            continue
        shots = p[p.type == "Shot"]
        first = p.iloc[0]
        rows.append({
            "possession_id": pid, "team": first.possession_team, "pattern": first.play_pattern,
            "start_x": first.x, "start_type": first.type, "start_t": first.elapsed,
            "first_shot_t": shots.elapsed.iloc[0] if len(shots) else np.nan,
            "n": len(p), "passes": int((p.type == "Pass").sum()), "xt": p.xt_added.clip(lower=0).sum(),
            "xg": shots.xg.sum(), "goal": bool((shots.outcome == "Goal").any()),
            "event_ids": p.event_id.tolist(),
        })
    return pd.DataFrame(rows)


def team_patterns(df: pd.DataFrame, poss: pd.DataFrame, team: str) -> dict:
    mine, theirs = df[df.team == team], df[df.team != team]
    passes = mine[mine.type == "Pass"]
    open_passes = passes[passes.set_piece.isna()] if "set_piece" in passes else passes
    shots = mine[mine.type == "Shot"]
    touches = df[df.type.isin(["Pass", "Carry"])]
    my_poss = poss[poss.team == team]
    counters = my_poss[(my_poss.pattern == "Regular Play") & (my_poss.start_x < 60)
                       & ((my_poss.first_shot_t - my_poss.start_t) <= 15)]
    high = mine[mine.type.isin(DEF_ACTIONS[:3]) & (mine.x >= 76)]  # within ~40m of goal
    # PPDA: opponent passes in their own 60% / our defensive actions + fouls in that zone
    opp_build = theirs[(theirs.type == "Pass") & (theirs.x < 72)]
    my_press = mine[(mine.type.isin(list(DEF_ACTIONS[:3]) + ["Foul Committed"])) & (mine.x > 48)]
    # final-third entries (completed pass/carry crossing x=80) by channel
    ft = mine[(mine.type.isin(["Pass", "Carry"])) & (mine.outcome.fillna("Complete") == "Complete")
              & (mine.x < 80) & (mine.end_x >= 80)]
    lanes = pd.cut(ft.end_y, [-1, 26.7, 53.3, 81], labels=["left", "centre", "right"])
    lane_n = ft.groupby(lanes, observed=False).size()
    lane_share = (lane_n / lane_n.sum()).to_dict() if lane_n.sum() > 0 else {}
    sp_xg = shots[shots.play_pattern.isin(["From Corner", "From Free Kick"]) | shots.set_piece.notna()].xg.sum()
    fwd = (open_passes.end_x - open_passes.x) > 10
    return {
        "possession_pct": 100 * (touches.team == team).mean(),
        "passes": len(passes),
        "pass_completion_pct": 100 * (passes.outcome == "Complete").mean(),
        "avg_pass_length": float(open_passes.length.mean()) if "length" in open_passes else np.nan,
        "forward_pass_pct": 100 * fwd.mean(),
        "shots": len(shots),
        "shots_on_target": int(shots.outcome.isin(["Goal", "Saved"]).sum()),
        "xg": float(shots.xg.sum()),
        "goals": int((shots.outcome == "Goal").sum()),
        "big_chances": int((shots.xg >= 0.25).sum()),
        "xt": float(mine.xt_added.clip(lower=0).sum()),
        "counters_to_shot": len(counters),
        "counter_ids": counters.event_ids.sum() if len(counters) else [],
        "final_third_entries": int(lane_n.sum()),
        "high_regains": len(high),
        "high_regain_ids": high.event_id.tolist(),
        "ppda": len(opp_build) / max(len(my_press), 1),
        "crosses": int(mine.get("cross", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "set_piece_xg": float(sp_xg),
        "fouls": int((mine.type == "Foul Committed").sum()),
        "yellows": int((mine.outcome == "Yellow Card").sum()),
        "reds": int(mine.outcome.isin(["Red Card", "Second Yellow"]).sum()),
        "corners": int((mine.set_piece == "Corner").sum()),
        "lane_share": {k: 100 * v for k, v in lane_share.items()},
    }


def player_table(df: pd.DataFrame, poss: pd.DataFrame) -> pd.DataFrame:
    on = df[df.player.notna()]
    # key pass = last completed pass before a shot in the same possession
    key_ids, assist_ids = set(), set()
    for _, p in df[df.type.isin(["Pass", "Shot"])].groupby("possession_id"):
        p = p.reset_index(drop=True)
        for i in range(1, len(p)):
            if p.type[i] == "Shot" and p.type[i - 1] == "Pass" and p.outcome[i - 1] == "Complete" \
                    and p.team[i] == p.team[i - 1]:
                key_ids.add(p.event_id[i - 1])
                if p.outcome[i] == "Goal":
                    assist_ids.add(p.event_id[i - 1])
    g = on.groupby(["team", "player"])
    t = pd.DataFrame({
        "position": g.position.first(),
        "touches": g.size(),
        "passes": g.apply(lambda d: (d.type == "Pass").sum(), include_groups=False),
        "pass_pct": g.apply(lambda d: 100 * (d[d.type == "Pass"].outcome == "Complete").mean()
                            if (d.type == "Pass").any() else np.nan, include_groups=False),
        "xt": g.xt_added.apply(lambda s: s.clip(lower=0).sum()),
        "shots": g.apply(lambda d: (d.type == "Shot").sum(), include_groups=False),
        "xg": g.xg.sum(),
        "goals": g.apply(lambda d: ((d.type == "Shot") & (d.outcome == "Goal")).sum(), include_groups=False),
        "key_passes": g.event_id.apply(lambda s: len(set(s) & key_ids)),
        "assists": g.event_id.apply(lambda s: len(set(s) & assist_ids)),
        "def_actions": g.apply(lambda d: d.type.isin(DEF_ACTIONS).sum(), include_groups=False),
        "fouls": g.apply(lambda d: (d.type == "Foul Committed").sum(), include_groups=False),
        "cards": g.apply(lambda d: (d.type == "Card").sum(), include_groups=False),
    }).reset_index()
    t["rating"] = (6.0 + 1.0 * t.goals + 0.6 * t.assists + 0.8 * t.xt + 1.0 * t.xg + 0.1 * t.key_passes
                   + 0.02 * t.def_actions - 0.1 * t.fouls - 0.6 * t.cards
                   + 0.01 * (t.pass_pct.fillna(75) - 75)).clip(4.0, 10.0).round(1)
    return t.sort_values("rating", ascending=False)


def turning_points(df: pd.DataFrame, meta: dict, wp: np.ndarray, lam0: float, mu0: float,
                   top: int = 8) -> list[dict]:
    """Events that moved the match, ranked by win-probability impact (incl. counterfactuals)."""
    home = meta["home"]
    names = {meta["home"]: meta["home_name"], meta["away"]: meta["away_name"]}
    out = []
    reds = {home: 0, meta["away"]: 0}
    xg = {home: 0.0, meta["away"]: 0.0}
    rows = df.reset_index(drop=True)
    for k, ev in rows.iterrows():
        if ev.type == "Shot":
            xg[ev.team] += ev.xg
        is_red = ev.type == "Card" and ev.outcome in ("Red Card", "Second Yellow")
        if is_red:
            reds[ev.team] += 1
        is_shot = ev.type == "Shot"
        if not (is_red or (is_shot and (ev.outcome == "Goal" or ev.xg >= 0.25))):
            continue
        side = 0 if ev.team == home else 2   # index of this team's win prob
        before = wp[k - 1] if k > 0 else wp[0]
        after = wp[k]
        dwp = float(after[side] - before[side])
        penalty = ev.get("set_piece") == "Penalty"
        if is_red:
            kind = "Red card"
        elif ev.outcome == "Goal":
            kind = "Penalty" if penalty else "Goal"
        else:
            kind = "Penalty missed" if penalty else "Big chance missed"
        counterfactual = None
        if is_shot and ev.outcome != "Goal":
            lam, mu = live_rates(lam0, mu0, ev.elapsed / 60, xg[home], xg[meta["away"]],
                                 reds[home], reds[meta["away"]])
            diff = int(ev.score_home - ev.score_away) + (1 if ev.team == home else -1)
            if_scored = outcome_probs(*game_state(lam, mu, diff), diff)
            counterfactual = float(if_scored[side] - after[side])
        impact = abs(dwp) if counterfactual is None else 0.6 * abs(counterfactual)
        chain = df[(df.possession_id == ev.possession_id) & (df.event_id <= ev.event_id)].tail(8)
        out.append({
            "event_id": int(ev.event_id), "clock": ev.clock, "minute": int(ev.elapsed // 60) + 1,
            "kind": kind, "team": ev.team, "team_name": names[ev.team], "player": ev.player,
            "xg": None if pd.isna(ev.get("xg")) else round(float(ev.xg), 2),
            "outcome": ev.outcome,
            "score_after": f"{int(ev.score_home)}-{int(ev.score_away)}",
            "wp_before": round(100 * float(before[side])), "wp_after": round(100 * float(after[side])),
            "dwp": round(100 * dwp), "counterfactual_dwp": None if counterfactual is None else round(100 * counterfactual),
            "impact": impact,
            "buildup_xt": round(float(chain.xt_added.clip(lower=0).sum()), 3),
            "pattern": ev.play_pattern,
            "chain": [{"id": int(c.event_id), "clock": c.clock, "type": c.type, "player": c.player,
                       "x": c.dx, "y": c.dy, "end_x": c.dend_x, "end_y": c.dend_y}
                      for c in chain.itertuples()],
            "evidence": chain.event_id.astype(int).tolist(),
        })
    goals_reds = [t for t in out if t["kind"] in ("Goal", "Penalty", "Red card")]
    misses = sorted([t for t in out if t not in goals_reds], key=lambda t: -t["impact"])
    keep = goals_reds + misses[: max(top - len(goals_reds), 2)]
    return sorted(keep, key=lambda t: t["event_id"])


def momentum_spells(mom: pd.DataFrame, home: str, away: str, min_len: int = 7,
                    thresh: float = 0.012) -> list[dict]:
    """Sustained periods where one side's smoothed threat dominated."""
    spells, cur = [], None
    for minute, v in mom["smooth"].items():
        side = home if v > thresh else away if v < -thresh else None
        if side != (cur or {}).get("team"):
            if cur and cur["end"] - cur["start"] + 1 >= min_len:
                spells.append(cur)
            cur = {"team": side, "start": minute, "end": minute} if side else None
        elif cur:
            cur["end"] = minute
    if cur and cur["end"] - cur["start"] + 1 >= min_len:
        spells.append(cur)
    for s in spells:
        w = mom.loc[s["start"]:s["end"]]
        s["threat_for"] = float(w["home" if s["team"] == home else "away"].sum())
        s["threat_against"] = float(w["away" if s["team"] == home else "home"].sum())
        s["start"] += 1
        s["end"] += 1
    return spells


def storyline(meta: dict, wp: np.ndarray, tps: list[dict], df: pd.DataFrame) -> dict:
    """Tags that make a match worth showing (used by the demo storyline finder)."""
    h, a = meta["score"]
    winner = 0 if h > a else 2 if a > h else None
    tags, drama = [], 0.0
    goals = df[(df.type == "Shot") & (df.outcome == "Goal")]
    leads = np.sign(goals.score_home - goals.score_away).to_numpy()
    lead_changes = int(np.sum((leads[1:] != leads[:-1]) & (leads[1:] != 0) & (leads[:-1] != 0))) if len(leads) > 1 else 0
    diffs = (goals.score_home - goals.score_away).to_numpy()
    if winner is not None and len(diffs):
        trailed = (diffs < 0).any() if winner == 0 else (diffs > 0).any()
        if trailed:
            tags.append("Comeback")
            drama += 2 + (0.3 - min(float(wp[:, winner].min()), 0.3)) * 10
        last = goals.iloc[-1]
        before = diffs[-2] if len(diffs) > 1 else 0
        if last.elapsed >= 84 * 60 and before == 0 and abs(h - a) == 1:
            tags.append("Late winner")
            drama += 2.5
    elif len(diffs) and goals.iloc[-1].elapsed >= 84 * 60:
        tags.append("Late equaliser")
        drama += 1.5
    if any(t["kind"] == "Red card" for t in tps):
        tags.append("Red-card swing")
        drama += 1
    xg_h = df[(df.type == "Shot") & (df.team == meta["home"])].xg.sum()
    xg_a = df[(df.type == "Shot") & (df.team == meta["away"])].xg.sum()
    if winner is not None and ((winner == 0 and xg_h < xg_a - 0.6) or (winner == 2 and xg_a < xg_h - 0.6)):
        tags.append("Smash and grab")
        drama += 1.5
    if h + a >= 5:
        tags.append("Goal-fest")
        drama += 1
    drama += 0.8 * lead_changes + 0.15 * (h + a) + float(np.abs(np.diff(wp[:, 0])).max()) * 2
    return {"tags": tags, "drama": round(drama, 2), "lead_changes": lead_changes}


def analyse(df: pd.DataFrame, meta: dict) -> dict:
    """One-call analysis bundle used by MatchStore."""
    from ..sim.teams import get_club
    df = prepare(df, meta)
    lam0, mu0 = prematch_rates(get_club(meta["home"]).quality, get_club(meta["away"]).quality)
    wp = timeline(df, meta["home"], lam0, mu0)
    poss = possessions(df)
    mom = momentum(df, meta["home"])
    tps = turning_points(df, meta, wp, lam0, mu0)
    return {
        "df": df, "wp": wp, "lam0": lam0, "mu0": mu0, "poss": poss, "momentum": mom,
        "patterns": {t: team_patterns(df, poss, t) for t in (meta["home"], meta["away"])},
        "players": player_table(df, poss),
        "turning_points": tps,
        "spells": momentum_spells(mom, meta["home"], meta["away"]),
        "storyline": storyline(meta, wp, tps, df),
    }
