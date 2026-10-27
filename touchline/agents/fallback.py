"""Deterministic template writers: the last link of the provider chain.

They produce every artefact the crew produces, built straight from the fact
store, so they are grounded by construction and the product never goes blank.
"""
from __future__ import annotations

from ..analytics.store import MatchStore


def _lead(store: MatchStore):
    h, a = store.meta["score"]
    if h == a:
        return None, None
    return (store.home, store.away) if h > a else (store.away, store.home)


def analyst(store: MatchStore) -> str:
    n, c = store.names, store.c
    fh, fa = store.f_team[store.home], store.f_team[store.away]
    xg_h, xg_a = store.facts[fh["xg"]].value, store.facts[fa["xg"]].value
    better = store.home if xg_h >= xg_a else store.away
    out = [f"- **{n[better]} made the better chances**: xG {c(store.f_team[better]['xg'])} vs "
           f"{c(store.f_team[store.away if better == store.home else store.home]['xg'])}."]
    pos = store.home if store.facts[fh["possession_pct"]].value >= 50 else store.away
    out.append(f"- **{n[pos]} had the ball**: {c(store.f_team[pos]['possession_pct'])} possession at "
               f"{c(store.f_team[pos]['pass_completion_pct'])} completion.")
    if store.turning_points:
        tp = max(store.turning_points, key=lambda t: abs(t["dwp"]))
        out.append(f"- **The decisive moment came at {tp['clock']}**: {tp['player']}'s {tp['kind'].lower()} swung "
                   f"{tp['team_name']}'s win probability by {c(tp['facts']['dwp'])}.")
        misses = [t for t in store.turning_points if "cf" in t["facts"]]
        if misses:
            m = max(misses, key=lambda t: t["counterfactual_dwp"])
            out.append(f"- **The one that got away**: {m['player']}'s chance at {m['clock']} (xG {c(m['facts']['xg'])}) "
                       f"would have been worth {c(m['facts']['cf'])} of win probability.")
    if store.f_player:
        name, f = max(store.f_player.items(), key=lambda kv: store.facts[kv[1]["rating"]].value)
        out.append(f"- **Player of the match: {name}**: rated {c(f['rating'])} with xT {c(f['xt'])}.")
    return "\n".join(out)


def tactician(store: MatchStore) -> str:
    n, c = store.names, store.c
    parts = []
    for team in (store.home, store.away):
        f = store.f_team[team]
        lanes = {k.split("_")[1]: v for k, v in f.items() if k.startswith("lane_")}
        main = max(lanes, key=lambda k: store.facts[lanes[k]].value) if lanes else "centre"
        parts.append(f"### {n[team]}\nPossession {c(f['possession_pct'])} with average pass length "
                     f"{c(f['avg_pass_length'])} yards. Pressed at PPDA {c(f['ppda'])} and won the ball back "
                     f"within 40m of goal {c(f['high_regains'])} times. Most final-third entries came down the "
                     f"{main} ({c(lanes[main]) if lanes else 'n/a'}).")
    if store.spells:
        s = max(store.spells, key=lambda s: s["threat_for"])
        why = (f"### Why momentum shifted\n{n[s['team']]}'s strongest spell ran from minute {s['start']} to {s['end']} "
               f"[{s['facts']['window']}], generating threat {c(s['facts']['for'])} against {c(s['facts']['against'])}.")
    else:
        why = "### Why momentum shifted\nNeither side sustained a long spell of dominance; the game turned on moments."
    parts.append(why)
    return "\n\n".join(parts)


def commentary(store: MatchStore) -> str:
    c = store.c
    lines = []
    for tp in store.turning_points:
        f = tp["facts"]
        if tp["kind"] in ("Goal", "Penalty"):
            txt = (f"GOAL! {tp['player']} scores for {tp['team_name']} - {tp['score_after']}. Win probability "
                   f"{c(f['wp_before'])} to {c(f['wp_after'])}.")
        elif tp["kind"] == "Red card":
            txt = f"Red card! {tp['player']} is off and {tp['team_name']}'s win probability drops to {c(f['wp_after'])}."
        else:
            txt = (f"Big chance for {tp['player']} (xG {c(f['xg'])}) but it stays out - a goal would have added "
                   f"{c(f['cf'])} to {tp['team_name']}'s win probability.")
        lines.append(f"{tp['clock']} | {txt}")
    return "LIVE:\n" + "\n".join(lines) + "\nRECAP:\n" + recap(store)


def recap(store: MatchStore) -> str:
    n, c = store.names, store.c
    win, lose = _lead(store)
    h, a = store.meta["score"]
    head = (f"{n[win]} beat {n[lose]} {max(h, a)}-{min(h, a)} [{store.f_score}]." if win
            else f"{n[store.home]} and {n[store.away]} drew {h}-{a} [{store.f_score}].")
    fh, fa = store.f_team[store.home], store.f_team[store.away]
    body = (f" {n[store.home]} produced xG {c(fh['xg'])} from {c(fh['shots'])} shots; {n[store.away]} {c(fa['xg'])} "
            f"from {c(fa['shots'])}.")
    if store.turning_points:
        tp = max(store.turning_points, key=lambda t: abs(t["dwp"]))
        body += (f" The biggest swing came at {tp['clock']} when {tp['player']}'s {tp['kind'].lower()} moved "
                 f"{tp['team_name']}'s win probability from {c(tp['facts']['wp_before'])} to {c(tp['facts']['wp_after'])}.")
    return head + body


def broadcast(store: MatchStore) -> str:
    return "**Headline**\n" + recap(store).split(".")[0] + ".\n\n" + analyst(store) + \
        "\n\n**On-screen graphics**\n- Win-probability timeline\n- Shot map sized by xG\n- Momentum (xT) chart"


def fan(store: MatchStore, persona: dict) -> str:
    return recap(store)


def social(store: MatchStore) -> str:
    n, c = store.names, store.c
    h, a = store.meta["score"]
    posts = [f"1) FT: {n[store.home]} {h}-{a} {n[store.away]} [{store.f_score}] ⚽ #Touchline"]
    fh = store.f_team[store.home]
    posts.append(f"2) {n[store.home]} created xG {c(fh['xg'])} 📊 #Touchline")
    if store.turning_points:
        tp = max(store.turning_points, key=lambda t: abs(t["dwp"]))
        posts.append(f"3) {tp['clock']}: {tp['player']} swings it - win probability {c(tp['facts']['dwp'])} 🔄 #Touchline")
    return "\n".join(posts)


def answer(store: MatchStore, question: str) -> str:
    return ("Offline mode: here is the verified match summary.\n" + store.summary_text())
