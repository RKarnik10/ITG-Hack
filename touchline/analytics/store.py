"""MatchStore: the single source of truth the agents read from and are checked against.

Every number the analytics produce is registered as a *fact* with an id (F1, F2,
...), a label, a unit and the event ids it was computed from. Agents must cite
facts as [F#]; `verify()` deterministically checks every cited number and flags
numbers that don't correspond to anything in the store.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .moments import analyse
from .winprob import game_state, live_rates

# "Analyse at" moments: (period, game-clock minute) - None minute = end of that period
MOMENTS = {"15'": (1, 15), "30'": (1, 30), "Half-time": (1, None), "60'": (2, 60), "75'": (2, 75), "Full time": None}


def cut_match(df: pd.DataFrame, meta: dict, moment: str = "Full time") -> tuple[pd.DataFrame, dict]:
    """The match as the studio sees it at `moment`: events so far, score so far."""
    spec = MOMENTS[moment]
    if spec is None:
        return df, meta
    period, minute = spec
    keep = (df.period <= period) if minute is None else \
        (df.period < period) | ((df.period == period) & (df.t < minute * 60))
    d = df[keep].reset_index(drop=True)
    m = dict(meta)
    m["score"] = [int(d.score_home.iloc[-1]), int(d.score_away.iloc[-1])]
    m["live"] = moment
    m["match_id"] = f"{meta['match_id']}@{re.sub(r'[^A-Za-z0-9]', '', moment)}"
    return d, m

# re.A: ASCII \w/\b, so numbers written straight after CJK/Devanagari text still count
CITE = re.compile(r"\[(F\d+)\]")
CITES = re.compile(r"\[(F\d+(?:\s*,\s*F\d+)*)\]")
SCORE = re.compile(r"\b(\d{1,2})\s*[-–]\s*(\d{1,2})\b", re.A)
MINUTE = re.compile(r"\b(\d{1,3})(?:\+(\d{1,2}))?(?:'|’|\s*(?:st|nd|rd|th)?\s*min(?:ute)?s?\b)", re.A)
NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w])", re.A)
PAIRED = re.compile(r"(?<![\w.])[+-]?(\d+(?:\.\d+)?)\s*(?:%|pts?|points|/10|yards?|yds)?\s*\[(F\d+(?:\s*,\s*F\d+)*)\]", re.A)
SENT = re.compile(r"(?<=[.!?])\s+|(?<=[。！？।])|\n+")


@dataclass
class Fact:
    id: str
    label: str
    value: float | int | str
    unit: str = ""
    team: str | None = None
    evidence: list[int] = field(default_factory=list)
    kind: str = "stat"

    def show(self) -> str:
        v = self.value
        if isinstance(v, float):
            v = f"{v:.1f}" if self.unit == "/10" else f"{v:.2f}" if abs(v) < 10 and not self.unit else f"{v:.0f}"
        return f"{v}{self.unit}"


def _p(name: str) -> str:
    return name + ("'" if str(name).endswith("s") else "'s")


def _num(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


class MatchStore:
    def __init__(self, df: pd.DataFrame, meta: dict):
        self.meta = meta
        self.home, self.away = meta["home"], meta["away"]
        self.names = {self.home: meta["home_name"], self.away: meta["away_name"]}
        a = analyse(df, meta)
        self.df: pd.DataFrame = a["df"]
        self.wp: np.ndarray = a["wp"]
        self.lam0, self.mu0 = a["lam0"], a["mu0"]
        self.momentum: pd.DataFrame = a["momentum"]
        self.patterns: dict = a["patterns"]
        self.players: pd.DataFrame = a["players"]
        self.turning_points: list[dict] = a["turning_points"]
        self.spells: list[dict] = a["spells"]
        self.storyline: dict = a["storyline"]
        self.facts: dict[str, Fact] = {}
        self._register()

    # ----------------------------------------------------------- registration
    def add(self, label: str, value, unit: str = "", team: str | None = None,
            evidence=(), kind: str = "stat") -> str:
        fid = f"F{len(self.facts) + 1}"
        value = _num(value)
        if isinstance(value, float):
            value = round(value, 2) if abs(value) < 10 else round(value, 1)
        self.facts[fid] = Fact(fid, label, value, unit, team, [int(e) for e in evidence][:40], kind)
        return fid

    def _register(self) -> None:
        h, a = self.meta["score"]
        n = self.names
        self.live = self.meta.get("live")
        self.f_score = self.add(f"{'Score at ' + self.live + ' (in progress)' if self.live else 'Final score'} "
                                f"{n[self.home]} v {n[self.away]}", f"{h}-{a}", kind="score")
        self.f_live = {}
        if self.live:
            now = self.wp[-1]
            last = self.df.iloc[-1]
            shots = self.df[self.df.type == "Shot"]
            reds = self.df[(self.df.type == "Card") & self.df.outcome.isin(["Red Card", "Second Yellow"])]
            lam, mu = live_rates(self.lam0, self.mu0, last.elapsed / 60,
                                 shots[shots.team == self.home].xg.sum(), shots[shots.team == self.away].xg.sum(),
                                 int((reds.team == self.home).sum()), int((reds.team == self.away).sum()))
            lam, mu = game_state(lam, mu, h - a)
            p_more = 1 - np.exp(-(lam + mu))
            self.f_live = {
                "wp_home": self.add(f"Live win probability {n[self.home]} at {self.live}", 100 * now[0], "%", self.home, kind="wp"),
                "wp_draw": self.add(f"Live draw probability at {self.live}", 100 * now[1], "%", kind="wp"),
                "wp_away": self.add(f"Live win probability {n[self.away]} at {self.live}", 100 * now[2], "%", self.away, kind="wp"),
                "more": self.add(f"Probability of at least one more goal after {self.live}", 100 * p_more, "%", kind="wp"),
                "next_home": self.add(f"Probability {n[self.home]} score the next goal", 100 * p_more * lam / (lam + mu), "%", self.home, kind="wp"),
                "next_away": self.add(f"Probability {n[self.away]} score the next goal", 100 * p_more * mu / (lam + mu), "%", self.away, kind="wp"),
            }
        pre = self.wp[0]
        self.f_pre = {
            self.home: self.add(f"Pre-match win probability {n[self.home]}", 100 * pre[0], "%", self.home, kind="wp"),
            "draw": self.add("Pre-match draw probability", 100 * pre[1], "%", kind="wp"),
            self.away: self.add(f"Pre-match win probability {n[self.away]}", 100 * pre[2], "%", self.away, kind="wp"),
        }
        self.f_team: dict[str, dict[str, str]] = {}
        spec = [("goals", "Goals", ""), ("xg", "Expected goals (xG)", ""), ("shots", "Shots", ""),
                ("shots_on_target", "Shots on target", ""), ("possession_pct", "Possession", "%"),
                ("passes", "Passes", ""), ("pass_completion_pct", "Pass completion", "%"),
                ("xt", "Expected threat (xT) generated", ""), ("big_chances", "Big chances (xG>=0.25)", ""),
                ("counters_to_shot", "Counter-attacks ending in a shot within 15s", ""),
                ("high_regains", "High regains (ball won back within 40m of the OPPONENT's goal)", ""), ("ppda", "PPDA (lower = more intense press)", ""),
                ("crosses", "Crosses", ""), ("set_piece_xg", "Set-piece xG", ""), ("corners", "Corners", ""),
                ("fouls", "Fouls committed", ""), ("yellows", "Yellow cards", ""), ("reds", "Red cards", ""),
                ("avg_pass_length", "Average open-play pass length (yards)", ""),
                ("forward_pass_pct", "Forward passes (>10 yards)", "%"),
                ("final_third_entries", "Final-third entries", "")]
        for team in (self.home, self.away):
            p = self.patterns[team]
            ids = {}
            for key, label, unit in spec:
                ev = p.get("counter_ids", []) if key == "counters_to_shot" else \
                    p.get("high_regain_ids", []) if key == "high_regains" else ()
                ids[key] = self.add(f"{n[team]} {label}", p[key], unit, team, ev)
            for lane, share in p["lane_share"].items():
                ids[f"lane_{lane}"] = self.add(f"{n[team]} share of final-third entries down the {lane}",
                                               share, "%", team)
            self.f_team[team] = ids
        for tp in self.turning_points:
            team = tp["team"]
            f = {"clock": self.add(f"{tp['kind']} by {tp['player']} ({n[team]}) - match clock", tp["clock"],
                                   team=team, evidence=tp["evidence"], kind="clock")}
            f["wp_before"] = self.add(f"{n[team]} win probability before {_p(tp['player'])} {tp['kind'].lower()} at {tp['clock']}",
                                      tp["wp_before"], "%", team, tp["evidence"], "wp")
            f["wp_after"] = self.add(f"{n[team]} win probability after {_p(tp['player'])} {tp['kind'].lower()} at {tp['clock']}",
                                     tp["wp_after"], "%", team, tp["evidence"], "wp")
            f["dwp"] = self.add(f"Win-probability swing for {n[team]} from {_p(tp['player'])} {tp['kind'].lower()} at {tp['clock']}",
                                tp["dwp"], " pts", team, tp["evidence"], "wp")
            if tp["xg"] is not None:
                f["xg"] = self.add(f"xG of {_p(tp['player'])} chance at {tp['clock']}", tp["xg"], "", team, tp["evidence"])
            if tp["counterfactual_dwp"] is not None:
                f["cf"] = self.add(f"Win-probability {n[team]} would have gained had {tp['player']} scored at {tp['clock']}",
                                   tp["counterfactual_dwp"], " pts", team, tp["evidence"], "wp")
            f["buildup_xt"] = self.add(f"xT built up in the move leading to {_p(tp['player'])} {tp['kind'].lower()} at {tp['clock']}",
                                       tp["buildup_xt"], "", team, tp["evidence"])
            tp["facts"] = f
        for s in self.spells:
            s["facts"] = {
                "window": self.add(f"{n[s['team']]} spell of pressure (minutes)", f"{s['start']}-{s['end']}",
                                   team=s["team"], kind="window"),
                "for": self.add(f"{n[s['team']]} threat (xT+xG) during {s['start']}'-{s['end']}' spell",
                                s["threat_for"], "", s["team"]),
                "against": self.add(f"Opponent threat (xT+xG) during {_p(n[s['team']])} {s['start']}'-{s['end']}' spell",
                                    s["threat_against"], "", s["team"]),
            }
        self.f_player: dict[str, dict[str, str]] = {}
        scorers = set(self.df[(self.df.type == "Shot") & (self.df.outcome == "Goal")].player)
        for team in (self.home, self.away):
            t = self.players[self.players.team == team]
            keep = pd.concat([t.head(3), t[t.player.isin(scorers)]]).drop_duplicates("player")
            for r in keep.itertuples():
                self.f_player[r.player] = {
                    "rating": self.add(f"{r.player} ({n[team]}, {r.position}) match rating", r.rating, "/10", team),
                    "goals": self.add(f"{r.player} goals", r.goals, "", team),
                    "xg": self.add(f"{r.player} xG", r.xg, "", team),
                    "xt": self.add(f"{r.player} xT generated", r.xt, "", team),
                    "key_passes": self.add(f"{r.player} key passes", r.key_passes, "", team),
                    "def_actions": self.add(f"{r.player} defensive actions", r.def_actions, "", team),
                }

    # ------------------------------------------------------------------ views
    def c(self, fid: str) -> str:
        """Value + citation, e.g. '1.94 [F7]'."""
        return f"{self.facts[fid].show()} [{fid}]"

    def summary_text(self) -> str:
        m, n = self.meta, self.names
        status = (f"STATUS: LIVE at {self.live}. The match is still in progress; this is NOT a final result." if self.live
                  else "STATUS: full time.")
        lines = [status,
                 f"{n[self.home]} {m['score'][0]}-{m['score'][1]} {n[self.away]} [{self.f_score}] (synthetic match, seed {m['seed']})",
                 f"Pre-match win probability: {n[self.home]} {self.c(self.f_pre[self.home])}, draw {self.c(self.f_pre['draw'])}, "
                 f"{n[self.away]} {self.c(self.f_pre[self.away])}",
                 f"Storyline tags: {', '.join(self.storyline['tags']) or 'none'}",
                 "Goals in order (running score, home-away): " + self.goals_text()]
        if self.live:
            f = self.f_live
            lines.append(f"Live win probability now: {n[self.home]} {self.c(f['wp_home'])}, draw {self.c(f['wp_draw'])}, "
                         f"{n[self.away]} {self.c(f['wp_away'])}. Chance of another goal {self.c(f['more'])}: "
                         f"next goal {n[self.home]} {self.c(f['next_home'])} vs {n[self.away]} {self.c(f['next_away'])}.")
        for team in (self.home, self.away):
            f = self.f_team[team]
            lines.append(f"{n[team]}: xG {self.c(f['xg'])}, shots {self.c(f['shots'])} ({self.c(f['shots_on_target'])} on target), "
                         f"possession {self.c(f['possession_pct'])}, pass completion {self.c(f['pass_completion_pct'])}, "
                         f"xT {self.c(f['xt'])}, big chances {self.c(f['big_chances'])}")
        return "\n".join(lines)

    def goals_text(self) -> str:
        g = self.df[(self.df.type == "Shot") & (self.df.outcome == "Goal")]
        return "; ".join(f"{r.clock} {r.player}{' (pen)' if r.set_piece == 'Penalty' else ''}, {self.names[r.team]} "
                         f"({r.score_home}-{r.score_away})" for r in g.itertuples()) or "none"

    def team_text(self, team: str) -> str:
        f, n = self.f_team[team], self.names[team]
        club = __import__("touchline.sim.teams", fromlist=["get_club"]).get_club(team)
        lanes = ", ".join(f"{k.split('_')[1]} {self.c(v)}" for k, v in f.items() if k.startswith("lane_"))
        return "\n".join([
            f"{n} (designed style: {club.style})",
            f"Possession {self.c(f['possession_pct'])}; passes {self.c(f['passes'])} at {self.c(f['pass_completion_pct'])}; "
            f"avg pass length {self.c(f['avg_pass_length'])} yds; forward passes {self.c(f['forward_pass_pct'])}",
            f"Pressing: PPDA {self.c(f['ppda'])}; high regains (won back within 40m of the opponent's goal) {self.c(f['high_regains'])}; fouls {self.c(f['fouls'])}",
            f"Attack: final-third entries {self.c(f['final_third_entries'])} ({lanes}); crosses {self.c(f['crosses'])}; "
            f"counters to shot {self.c(f['counters_to_shot'])}",
            f"Chances: shots {self.c(f['shots'])}, xG {self.c(f['xg'])}, big chances {self.c(f['big_chances'])}, "
            f"set-piece xG {self.c(f['set_piece_xg'])}, corners {self.c(f['corners'])}",
            f"Discipline: yellows {self.c(f['yellows'])}, reds {self.c(f['reds'])}",
        ])

    def turning_points_text(self) -> str:
        out = []
        for tp in self.turning_points:
            f = tp["facts"]
            chain = " -> ".join(f"{c['type']}({c['player']})" for c in tp["chain"][-5:])
            line = (f"{tp['clock']} [{f['clock']}] {tp['kind']}: {tp['player']} ({tp['team_name']}), "
                    f"score after {tp['score_after']}. {tp['team_name']} win prob {self.c(f['wp_before'])} -> "
                    f"{self.c(f['wp_after'])} (swing {self.c(f['dwp'])})")
            if "xg" in f:
                line += f"; chance xG {self.c(f['xg'])}"
            if "cf" in f:
                line += f"; had it gone in, +{self.c(f['cf'])} win prob"
            line += f"; build-up xT {self.c(f['buildup_xt'])}; pattern {tp['pattern']}; move: {chain}"
            out.append(line)
        return "\n".join(out) or "No major turning points."

    def momentum_text(self) -> str:
        if not self.spells:
            return "No sustained spells of dominance; the match was evenly contested throughout."
        return "\n".join(
            f"{self.names[s['team']]} dominated minutes {s['start']}-{s['end']} [{s['facts']['window']}]: threat "
            f"{self.c(s['facts']['for'])} vs {self.c(s['facts']['against'])}" for s in self.spells)

    def players_text(self, team: str | None = None) -> str:
        out = []
        for name, f in self.f_player.items():
            if team and self.facts[f["rating"]].team != team:
                continue
            out.append(f"{self.facts[f['rating']].label}: {self.c(f['rating'])}; goals {self.c(f['goals'])}, xG {self.c(f['xg'])}, "
                       f"xT {self.c(f['xt'])}, key passes {self.c(f['key_passes'])}, defensive actions {self.c(f['def_actions'])}")
        return "\n".join(out)

    def events_text(self, start_minute: int, end_minute: int, limit: int = 40) -> str:
        d = self.df[(self.df.elapsed >= (start_minute - 1) * 60) & (self.df.elapsed < end_minute * 60)]
        d = d[d.type.isin(["Shot", "Card", "Substitution", "Foul Committed", "Save"])
              | ((d.type == "Pass") & (d.xt_added > 0.03)) | ((d.type == "Carry") & (d.xt_added > 0.03))]
        rows = [f"#{r.event_id} {r.clock} {self.names[r.team]} {r.type} {r.player or ''}"
                f"{' -> ' + str(r.outcome) if isinstance(r.outcome, str) else ''}"
                f"{f' xG {r.xg:.2f}' if r.type == 'Shot' else ''}{f' xT +{r.xt_added:.2f}' if r.xt_added > 0 else ''}"
                for r in d.head(limit).itertuples()]
        return "\n".join(rows) or "No notable events in that window."

    def fact_sheet(self, ids=None) -> str:
        facts = self.facts.values() if ids is None else [self.facts[i] for i in ids if i in self.facts]
        return "\n".join(f"{f.id} | {f.label} | {f.show()}" for f in facts)

    def core_fact_ids(self) -> list[str]:
        ids = [self.f_score]
        for team in (self.home, self.away):
            ids += [self.f_team[team][k] for k in ("goals", "xg", "shots", "possession_pct", "big_chances")]
        return ids

    # ---------------------------------------------------------------- checking
    def _known_numbers(self) -> set[float]:
        vals = set()
        for f in self.facts.values():
            if isinstance(f.value, (int, float)):
                vals |= {round(float(f.value), 2), round(float(f.value), 1), float(round(float(f.value)))}
            else:  # windows like "88-96" and clocks like "90+4'"
                vals |= {float(x) for x in re.findall(r"\d+", str(f.value))}
        return vals

    @staticmethod
    def _sentences(text: str) -> list[str]:
        """Split into sentences, re-attaching citations written after the full stop ('... 61%. [F9]')."""
        out = []
        for s in (s.strip() for s in SENT.split(text or "")):
            if not s:
                continue
            m = re.match(r"^((?:\[F\d+(?:\s*,\s*F\d+)*\]\s*)+)(.*)$", s)
            if m and out:
                out[-1] += " " + m.group(1).strip()
                s = m.group(2).strip()
                if not s:
                    continue
            out.append(s)
        return out

    @staticmethod
    def _close(x: float, v: float) -> bool:
        x, v = abs(x), abs(v)
        return abs(x - v) <= max(0.051, 0.02 * v) or round(v) == x

    def verify(self, text: str) -> dict:
        """Deterministic fact check.

        - a number written directly before a citation ("1.68 [F6]") must equal that fact;
        - other citations need some matching number in their sentence (if it has numbers);
        - scorelines must have actually occurred; uncited numbers must exist somewhere in the store.
        """
        known = self._known_numbers()
        score_states = {(int(h), int(a)) for h, a in zip(self.df.score_home, self.df.score_away)}
        score_states |= {(a, h) for h, a in score_states}
        issues, checked = [], 0
        for sent in self._sentences(text):
            paired = set()
            for m in PAIRED.finditer(sent):
                if m.start() > 0 and sent[m.start() - 1] in "-–":   # "2-1 [F1]": a scoreline, checked below
                    continue
                x = float(m.group(1))
                for fid in [i.strip() for i in m.group(2).split(",")]:
                    paired.add(fid)
                    checked += 1
                    fact = self.facts.get(fid)
                    if fact is None:
                        issues.append({"sentence": sent, "problem": f"{fid} does not exist", "severity": "error"})
                    elif isinstance(fact.value, (int, float)) and not self._close(x, float(fact.value)):
                        issues.append({"sentence": sent, "fact": fid, "severity": "error",
                                       "problem": f"wrote {x:g} but {fid} = {fact.show()} ({fact.label})"})
            ids = [i.strip() for grp in CITES.findall(sent) for i in grp.split(",")]
            body = CITES.sub(" ", sent)
            for h, a in SCORE.findall(body):
                if (int(h), int(a)) not in score_states:
                    issues.append({"sentence": sent, "problem": f"scoreline {h}-{a} never occurred", "severity": "error"})
            body = MINUTE.sub(" ", SCORE.sub(" ", body))
            nums = [float(x) for x in NUM.findall(body)]
            for fid in ids:
                if fid in paired:
                    continue
                checked += 1
                fact = self.facts.get(fid)
                if fact is None:
                    issues.append({"sentence": sent, "problem": f"{fid} does not exist", "severity": "error"})
                elif isinstance(fact.value, (int, float)) and nums and \
                        not any(self._close(x, float(fact.value)) for x in nums):
                    issues.append({"sentence": sent, "fact": fid, "severity": "error",
                                   "problem": f"no number in the sentence matches {fid} = {fact.show()} ({fact.label})"})
            for x in nums:
                if x in known or round(x, 1) in known or x <= 3:
                    continue
                issues.append({"sentence": sent, "problem": f"number {x:g} is not backed by any fact",
                               "severity": "warning"})
        errors = [i for i in issues if i["severity"] == "error"]
        return {"ok": not errors and len(issues) <= 1, "citations_checked": checked, "issues": issues}

    # ------------------------------------------------------------ persistence
    def facts_json(self) -> list[dict]:
        return [asdict(f) for f in self.facts.values()]
