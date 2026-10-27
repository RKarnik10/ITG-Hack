"""Synthetic match engine: a seeded, possession-chain simulator.

Every event is generated here; no real match is replayed. Realism comes from
league-wide *zone priors* on an 18x12 grid: for an action starting in a zone,
the probability that it is a shot (s), a completed move (m) or a turnover (t),
and the chance a shot from there scores (g). The priors are aggregate rates
estimated from public StatsBomb Open Data by the sibling xT project; the engine
layers club style (quality, press, directness, tempo, aggression) on top.

Coordinates follow the StatsBomb convention: 120x80 pitch, and each event is
recorded in the acting team's frame (that team attacks toward x=120).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .teams import SLOTS, Club, Player, get_club

PITCH_X, PITCH_Y = 120.0, 80.0
N_X, N_Y = 18, 12
PRIORS_PATH = Path(__file__).resolve().parent.parent / "data" / "xt_surface.npz"
SLOT_HOME = {pos: (x, y) for pos, x, y in SLOTS}
SLOT_HOME.update({"GK": (6, 40)})


def load_priors() -> dict[str, np.ndarray]:
    d = np.load(PRIORS_PATH)
    return {k: np.asarray(d[k], dtype=float) for k in ("s", "m", "t", "g", "xt")}


PRIORS = load_priors()
SHOT_SCALE = 0.34   # calibrates shot volume to ~12-13 per team (see tests/test_engine.py)
DEEP_X = 112.0      # open-play receptions/carries rarely end inside the six-yard box


def zone(x: float, y: float) -> int:
    xi = min(max(int(x / PITCH_X * N_X), 0), N_X - 1)
    yi = min(max(int(y / PITCH_Y * N_Y), 0), N_Y - 1)
    return yi * N_X + xi


def mirror(x: float, y: float) -> tuple[float, float]:
    """Flip a location into the other team's attacking frame."""
    return PITCH_X - x, PITCH_Y - y


def shot_xg(x: float, y: float, header: bool = False) -> float:
    """Distance/angle logistic blended with the zone's empirical conversion rate."""
    dx, dy = max(PITCH_X - x, 0.5), abs(40.0 - y)
    dist = math.hypot(dx, dy)
    angle = math.atan2(8.0 * dx, dx * dx + dy * dy - 16.0)  # angle subtended by the posts
    if angle < 0:
        angle += math.pi
    p = 1.0 / (1.0 + math.exp(-(-0.45 + 1.25 * angle - 0.15 * dist)))
    xg = 0.5 * p + 0.5 * PRIORS["g"][zone(x, y)]
    if header:
        xg *= 0.55
    return float(min(max(xg, 0.01), 0.9))


def in_box(x: float, y: float) -> bool:
    return x > 102 and 18 < y < 62


@dataclass
class Side:
    club: Club
    on_pitch: list[Player]
    bench: list[Player]
    q: float
    goals: int = 0
    subs_left: int = 3
    sub_minutes: list[int] = field(default_factory=list)
    yellows: set = field(default_factory=set)

    @property
    def code(self) -> str:
        return self.club.code


class MatchSim:
    def __init__(self, home: str, away: str, seed: int, match_id: str | None = None):
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.match_id = match_id or f"{home}-{away}-{seed}"
        self.sides = {}
        for code, ha in ((home, 1.05), (away, 1.0)):
            club = get_club(code)
            squad = list(club.squad)
            side = Side(club, squad[:11], squad[11:], q=club.quality * ha)
            side.sub_minutes = sorted(int(m) for m in self.rng.uniform(56, 86, 3))
            self.sides[code] = side
        self.home, self.away = home, away
        self.events: list[dict] = []
        self.t = 0.0
        self.period = 1
        self.poss = home
        self.holder: Player | None = None
        self.ball = (60.0, 40.0)
        self.possession_id = 0
        self.play_pattern = "Kick Off"

    # ------------------------------------------------------------------ utils
    def other(self, code: str) -> str:
        return self.away if code == self.home else self.home

    def _emit(self, team: str, player: Player | None, type_: str, x: float, y: float,
              end: tuple[float, float] | None = None, outcome: str | None = None,
              **extra) -> dict:
        ev = {
            "event_id": len(self.events) + 1,
            "match_id": self.match_id,
            "period": self.period,
            "t": round(self.t, 1),
            "minute": int(self.t // 60),
            "second": int(self.t % 60),
            "team": team,
            "player": player.name if player else None,
            "player_id": player.id if player else None,
            "position": player.position if player else None,
            "type": type_,
            "x": round(x, 1), "y": round(y, 1),
            "end_x": round(end[0], 1) if end else None,
            "end_y": round(end[1], 1) if end else None,
            "outcome": outcome,
            "possession_id": self.possession_id,
            "possession_team": self.poss,
            "play_pattern": self.play_pattern,
            "score_home": self.sides[self.home].goals,
            "score_away": self.sides[self.away].goals,
        }
        ev.update(extra)
        self.events.append(ev)
        return ev

    def _pick(self, side: Side, loc: tuple[float, float], exclude: Player | None = None,
              weight_attr: str | None = None) -> Player:
        """Player most plausibly at `loc` given slot homes shifted with the ball."""
        bx = loc[0]
        shift = (bx - 50.0) * 0.45
        w = []
        for p in side.on_pitch:
            hx, hy = SLOT_HOME.get(p.position, (60, 40))
            hx = min(max(hx + (shift if p.position != "GK" else 0.0), 4.0), 116.0)
            d2 = (hx - loc[0]) ** 2 + (hy - loc[1]) ** 2
            wi = math.exp(-d2 / (2 * 16.0 ** 2))
            if p.position == "GK" and loc[0] > 25:
                wi *= 0.02
            if weight_attr:
                wi *= getattr(p, weight_attr) ** 2
            if exclude is not None and p.id == exclude.id:
                wi = 0.0
            w.append(wi + 1e-9)
        w = np.asarray(w)
        return side.on_pitch[int(self.rng.choice(len(w), p=w / w.sum()))]

    def _advance(self, seconds: float) -> None:
        self.t += max(seconds, 0.3)

    def _new_possession(self, team: str, loc: tuple[float, float], holder: Player | None,
                        pattern: str) -> None:
        self.possession_id += 1
        self.poss = team
        self.ball = loc
        self.holder = holder or self._pick(self.sides[team], loc)
        self.play_pattern = pattern

    # -------------------------------------------------------------- set pieces
    def kickoff(self, team: str) -> None:
        self._new_possession(team, (60.0, 40.0), None, "Kick Off")
        side = self.sides[team]
        taker = next(p for p in side.on_pitch if p.position in ("ST", "LCM", "RCM"))
        end = (float(self.rng.uniform(42, 52)), float(self.rng.uniform(25, 55)))
        receiver = self._pick(side, end, exclude=taker)
        self._emit(team, taker, "Pass", 60.0, 40.0, end, "Complete",
                   recipient=receiver.name, set_piece="Kick Off")
        self.ball, self.holder = end, receiver
        self._advance(3)

    def goal_kick(self, team: str) -> None:
        self._new_possession(team, (6.0, 40.0), None, "From Goal Kick")
        side = self.sides[team]
        gk = next((p for p in side.on_pitch if p.position == "GK"), side.on_pitch[0])
        self._advance(self.rng.uniform(15, 30))
        long_ = self.rng.random() < 0.35 * side.club.directness
        if long_:
            end = (float(self.rng.uniform(55, 80)), float(self.rng.uniform(15, 65)))
            won = self.rng.random() < 0.5
        else:
            end = (float(self.rng.uniform(12, 30)), float(self.rng.choice([12, 68]) + self.rng.normal(0, 6)))
            won = self.rng.random() < 0.93
        self._pass_to(team, gk, (6.0, 40.0), end, won, set_piece="Goal Kick")

    def throw_in(self, team: str, x: float, top: bool) -> None:
        y = 0.5 if top else 79.5
        self._new_possession(team, (x, y), None, "From Throw In")
        self._advance(self.rng.uniform(8, 20))
        side = self.sides[team]
        taker = self._pick(side, (x, y))
        end = (min(max(x + self.rng.normal(4, 7), 2), 118), y + (1 if top else -1) * self.rng.uniform(6, 18))
        self._pass_to(team, taker, (x, y), end, self.rng.random() < 0.8, set_piece="Throw-in")

    def corner(self, team: str) -> None:
        top = self.rng.random() < 0.5
        loc = (119.5, 0.5 if top else 79.5)
        self._new_possession(team, loc, None, "From Corner")
        self._advance(self.rng.uniform(20, 40))
        side, opp = self.sides[team], self.sides[self.other(team)]
        taker = next((p for p in side.on_pitch if p.position in ("LW", "RW", "LCM", "RCM")), side.on_pitch[-1])
        target = (float(self.rng.uniform(106, 116)), float(self.rng.uniform(30, 50)))
        if self.rng.random() < 0.30:
            header = self._pick(side, target, exclude=taker, weight_attr="aerial")
            self._emit(team, taker, "Pass", *loc, target, "Complete", recipient=header.name, set_piece="Corner")
            self.ball, self.holder = target, header
            self._advance(1.5)
            self.shot(header_=True)
        else:
            self._emit(team, taker, "Pass", *loc, target, "Incomplete", set_piece="Corner")
            self._advance(1.5)
            clearer = self._pick(opp, mirror(*target), weight_attr="aerial")
            cx, cy = mirror(*target)
            self._emit(opp.code, clearer, "Clearance", cx, cy)
            if self.rng.random() < 0.45:  # second ball won back by the attackers
                loc2 = (float(self.rng.uniform(82, 98)), float(self.rng.uniform(15, 65)))
                self._new_possession(team, loc2, None, "From Corner")
                self._emit(team, self.holder, "Ball Recovery", *loc2)
            else:
                loc2 = (float(self.rng.uniform(22, 40)), float(self.rng.uniform(15, 65)))
                self._new_possession(opp.code, loc2, None, "Regular Play")
                self._emit(opp.code, self.holder, "Ball Recovery", *loc2)
            self._advance(2)

    def free_kick(self, team: str, loc: tuple[float, float]) -> None:
        self._new_possession(team, loc, None, "From Free Kick")
        self._advance(self.rng.uniform(15, 35))
        x, y = loc
        if in_box(x, y):
            side = self.sides[team]
            taker = max(side.on_pitch, key=lambda p: p.finishing)
            self.holder, self.ball = taker, (108.0, 40.0)
            self.shot(penalty=True)
        elif x > 88 and 22 < y < 58 and self.rng.random() < 0.3:
            side = self.sides[team]
            self.holder = max(side.on_pitch, key=lambda p: p.finishing)
            self.shot(direct_fk=True)
        # otherwise play continues from the spot via the normal step loop

    # ------------------------------------------------------------- open play
    def _pass_to(self, team: str, passer: Player, start, end, complete: bool, set_piece=None,
                 cross=False) -> None:
        side = self.sides[team]
        ex = min(max(end[0], 0.5), 119.5)
        if complete and not cross and ex > DEEP_X:
            ex = DEEP_X - abs(float(self.rng.normal(0, 5)))
        ey = end[1]
        if complete:
            ey = min(max(ey, 0.5), 79.5)
            receiver = self._pick(side, (ex, ey), exclude=passer)
            self._emit(team, passer, "Pass", *start, (ex, ey), "Complete", recipient=receiver.name,
                       set_piece=set_piece, cross=cross, length=round(math.dist(start, (ex, ey)), 1))
            self.ball, self.holder = (ex, ey), receiver
            self._advance(0.9 + math.dist(start, (ex, ey)) / 20 + self.rng.exponential(0.6) / side.club.tempo)
            return
        opp_code = self.other(team)
        opp = self.sides[opp_code]
        if ey < 0 or ey > 80:
            self._emit(team, passer, "Pass", *start, (ex, min(max(ey, 0), 80)), "Out",
                       set_piece=set_piece, cross=cross, length=round(math.dist(start, (ex, ey)), 1))
            self._advance(2)
            self.throw_in(opp_code, PITCH_X - ex, top=ey > 80)
            return
        ey = min(max(ey, 0.5), 79.5)
        self._emit(team, passer, "Pass", *start, (ex, ey), "Incomplete", set_piece=set_piece,
                   cross=cross, length=round(math.dist(start, (ex, ey)), 1))
        self._advance(1.2 + math.dist(start, (ex, ey)) / 20)
        if cross and self.rng.random() < 0.35:  # blocked/headed behind
            rx, ry = mirror(ex, ey)
            self._emit(opp_code, self._pick(opp, (rx, ry), weight_attr="aerial"), "Clearance", rx, ry)
            self.corner(team)
            return
        rx, ry = mirror(ex, ey)
        winner = self._pick(opp, (rx, ry))
        kind = "Interception" if math.dist(start, (ex, ey)) < 25 and self.rng.random() < 0.5 else "Ball Recovery"
        self._new_possession(opp_code, (rx, ry), winner, "Regular Play")
        self._emit(opp_code, winner, kind, rx, ry)

    def _pass_end(self, x: float, y: float, club: Club, longer: bool) -> tuple[float, float, bool]:
        deep = (2.0 if longer else 1.35) if x < 35 else 1.0  # clearances from deep go long
        mean_len = 14.0 * club.directness * (1.45 if longer else 1.0) * deep
        length = float(self.rng.lognormal(math.log(mean_len) - 0.15, 0.55))
        # Aim at a point ahead of the ball that saturates at the edge of the box, so
        # attacks circulate around the area (sideways/backward) instead of piling in.
        aim_x = min(x + 30.0, 101.0)
        mu = math.atan2((40.0 - y) * (0.6 if x > 70 else 0.15), aim_x - x) if aim_x > x + 1 else math.pi * 0.85 * (1 if y < 40 else -1)
        kappa = 0.8 * club.directness * (1.0 if x < 70 else 0.45)
        cross = False
        if x > 80:
            if (y < 18 or y > 62) and x > 95 and self.rng.random() < 0.35:
                cross = True
                mu = math.atan2(40.0 - y, 112.0 - x)
                length = math.hypot(112.0 - x, 40.0 - y) + self.rng.normal(0, 4)
        theta = float(self.rng.vonmises(mu, kappa + (2.5 if cross else 0.0)))
        return x + length * math.cos(theta), y + length * math.sin(theta), cross

    def step(self) -> None:
        team = self.poss
        att, dfn = self.sides[team], self.sides[self.other(team)]
        x, y = self.ball
        if self.holder is None or self.holder not in att.on_pitch:
            self.holder = self._pick(att, (x, y))
        z = zone(x, y)
        ratio = att.q / dfn.q
        # game state: trailing sides push in the second half, leaders sit deeper
        if self.period == 2 and att.goals != dfn.goals:
            ratio *= 1.12 if att.goals < dfn.goals else 0.92
        s = PRIORS["s"][z] * ratio ** 0.8 * SHOT_SCALE
        t = PRIORS["t"][z] / ratio ** 1.1
        t *= 1.0 + (dfn.club.press - 1.0) * (1.3 if x < 60 else 0.4)
        m = max(PRIORS["m"][z], 0.05)
        if self.holder.position == "GK":
            s = 0.0
        total = s + t + m
        s, t = s / total, t / total

        p_foul = 0.0105 * dfn.club.aggression * (1.25 if 30 < x < 95 else 1.0) * (0.5 if in_box(x, y) else 1.0)
        u = self.rng.random()
        if u < p_foul and self.holder.position != "GK":
            self.foul()
            return
        u = self.rng.random()
        if u < s:
            self.shot()
        elif u < s + t:
            self.turnover()
        else:
            self.move()

    def move(self) -> None:
        team = self.poss
        side = self.sides[team]
        x, y = self.ball
        carry_share = 0.42 / side.club.directness
        if self.rng.random() < carry_share and self.holder.position != "GK":
            length = float(self.rng.gamma(2.0, 3.2))
            aim_x = min(x + 15.0, 106.0)
            mu = math.atan2((40 - y) * 0.4, aim_x - x) if aim_x > x + 1 else math.pi * (0.6 if y < 40 else -0.6)
            theta = float(self.rng.vonmises(mu, 1.6 if x < 80 else 0.8))
            ex = min(max(x + length * math.cos(theta), 0.5), DEEP_X + float(self.rng.uniform(0, 4)))
            ey = min(max(y + length * math.sin(theta), 0.5), 79.5)
            self._emit(team, self.holder, "Carry", x, y, (ex, ey))
            self.ball = (ex, ey)
            self._advance(0.4 + length / 5.5)
            return
        ex, ey, cross = self._pass_end(x, y, side.club, longer=False)
        if cross:  # crosses are hard: most are cleared
            if self.rng.random() < 0.3:
                self._pass_to(team, self.holder, (x, y), (ex, ey), True, cross=True)
                if self.rng.random() < 0.55:
                    self.shot(header_=self.rng.random() < 0.6)
            else:
                self._pass_to(team, self.holder, (x, y), (ex, ey), False, cross=True)
            return
        self._pass_to(team, self.holder, (x, y), (ex, ey), True)

    def turnover(self) -> None:
        team = self.poss
        opp_code = self.other(team)
        opp = self.sides[opp_code]
        x, y = self.ball
        kind = self.rng.choice(["pass", "tackle", "miscontrol"], p=[0.70, 0.19, 0.11])
        if kind == "pass" or self.holder.position == "GK":
            ex, ey, cross = self._pass_end(x, y, self.sides[team].club, longer=True)
            if ex > 120:  # overhit: goal kick
                self._emit(team, self.holder, "Pass", x, y, (120.0, min(max(ey, 0), 80)), "Out", cross=cross)
                self._advance(2)
                self.goal_kick(opp_code)
                return
            self._pass_to(team, self.holder, (x, y), (ex, ey), False, cross=cross)
            return
        rx, ry = mirror(x, y)
        winner = self._pick(opp, (rx, ry))
        if kind == "tackle":
            self._emit(team, self.holder, "Dispossessed", x, y)
            self._emit(opp_code, winner, "Tackle", rx, ry, outcome="Won")
        else:
            self._emit(team, self.holder, "Miscontrol", x, y)
            self._emit(opp_code, winner, "Ball Recovery", rx, ry)
        self._advance(1.5)
        self._new_possession(opp_code, (rx, ry), winner, "Regular Play")

    def foul(self) -> None:
        team = self.poss
        opp_code = self.other(team)
        opp = self.sides[opp_code]
        x, y = self.ball
        rx, ry = mirror(x, y)
        fouler = self._pick(opp, (rx, ry))
        self._emit(opp_code, fouler, "Foul Committed", rx, ry)
        self._emit(team, self.holder, "Foul Won", x, y)
        self._advance(2)
        card = None
        if self.rng.random() < 0.002 * opp.club.aggression:
            card = "Red Card"
        elif self.rng.random() < 0.135 * opp.club.aggression * (0.35 if fouler.id in opp.yellows else 1.0):
            card = "Second Yellow" if fouler.id in opp.yellows else "Yellow Card"
            opp.yellows.add(fouler.id)
        if card:
            self._emit(opp_code, fouler, "Card", rx, ry, outcome=card)
            self._advance(30)
            if card in ("Red Card", "Second Yellow"):
                opp.on_pitch = [p for p in opp.on_pitch if p.id != fouler.id]
                opp.q *= 0.86
        self.free_kick(team, (x, y))

    def shot(self, header_: bool = False, penalty: bool = False, direct_fk: bool = False) -> None:
        team = self.poss
        att = self.sides[team]
        opp_code = self.other(team)
        x, y = self.ball
        shooter = self.holder
        if penalty:
            xg = 0.76
        else:
            xg = shot_xg(x, y, header=header_)
            if direct_fk:
                xg *= 0.7
        p_goal = min(xg * shooter.finishing ** 0.5 * (att.q / self.sides[opp_code].q) ** 0.3, 0.97)
        goal = self.rng.random() < p_goal
        body = "Head" if header_ else "Foot"
        if goal:
            outcome = "Goal"
        else:
            outcome = self.rng.choice(["Saved", "Off T", "Blocked"],
                                      p=[0.5, 0.5, 0.0] if penalty else [0.36, 0.36, 0.28])
        sp = "Penalty" if penalty else ("Free Kick" if direct_fk else None)
        self._emit(team, shooter, "Shot", x, y, (120.0, float(np.clip(40 + self.rng.normal(0, 3.5), 34, 46))),
                   outcome, xg=round(xg, 3), body_part=body, set_piece=sp)
        self._advance(1.5)
        if goal:
            att.goals += 1
            self.events[-1]["score_home"] = self.sides[self.home].goals
            self.events[-1]["score_away"] = self.sides[self.away].goals
            self._advance(self.rng.uniform(45, 75))
            self.kickoff(opp_code)
        elif outcome == "Saved":
            opp = self.sides[opp_code]
            gk = next((p for p in opp.on_pitch if p.position == "GK"), opp.on_pitch[0])
            self._emit(opp_code, gk, "Save", *mirror(119, 40))
            if self.rng.random() < 0.3:
                self._advance(3)
                self.corner(team)
            else:
                self._new_possession(opp_code, (7.0, 40.0), gk, "Regular Play")
                self._advance(self.rng.uniform(4, 10))
        elif outcome == "Off T":
            self._advance(4)
            self.goal_kick(opp_code)
        else:
            r = self.rng.random()
            if r < 0.35:
                self.corner(team)
            elif r < 0.6:  # rebound stays with the attackers
                self.ball = (max(x - self.rng.uniform(4, 12), 70.0), y)
                self.holder = self._pick(att, self.ball)
                self._emit(team, self.holder, "Ball Recovery", *self.ball)
            else:
                rx, ry = mirror(x, y)
                opp = self.sides[opp_code]
                winner = self._pick(opp, (rx, ry))
                self._new_possession(opp_code, (rx, ry), winner, "Regular Play")
                self._emit(opp_code, winner, "Clearance", rx, ry)

    def maybe_subs(self) -> None:
        minute = self.t / 60
        for side in self.sides.values():
            while side.sub_minutes and minute >= side.sub_minutes[0] and side.subs_left > 0:
                side.sub_minutes.pop(0)
                outfield = [p for p in side.on_pitch if p.position != "GK"]
                cands = [p for p in side.bench if p.position != "GK"]
                if not cands or not outfield:
                    break
                incoming = cands[int(self.rng.integers(len(cands)))]
                prefer = [p for p in outfield if p.line in ("FWD", "MID")] or outfield
                off = prefer[int(self.rng.integers(len(prefer)))]
                incoming = Player(incoming.id, incoming.name, incoming.number, off.position,
                                  incoming.finishing, incoming.passing, incoming.aerial)
                side.on_pitch = [incoming if p.id == off.id else p for p in side.on_pitch]
                side.bench = [p for p in side.bench if p.id != incoming.id]
                side.subs_left -= 1
                self._emit(side.code, off, "Substitution", *SLOT_HOME.get(off.position, (60, 40)),
                           outcome=incoming.name, recipient=incoming.name)
                self._advance(20)

    # ------------------------------------------------------------------- run
    def run(self) -> tuple[pd.DataFrame, dict]:
        stoppage = {1: int(self.rng.integers(1, 5)), 2: int(self.rng.integers(3, 8))}
        for period, start, kick in ((1, 0, self.home), (2, 45 * 60, self.away)):
            self.period, self.t = period, float(start)
            self._emit(kick, None, "Half Start", 60, 40)
            self.kickoff(kick)
            end = start + (45 + stoppage[period]) * 60
            while self.t < end:
                self.step()
                if period == 2:
                    self.maybe_subs()
            self._emit(kick, None, "Half End", 60, 40)
        df = pd.DataFrame(self.events)
        h, a = self.sides[self.home], self.sides[self.away]
        meta = {
            "match_id": self.match_id, "seed": self.seed,
            "home": self.home, "away": self.away,
            "home_name": h.club.name, "away_name": a.club.name,
            "home_colour": h.club.colour, "away_colour": a.club.colour,
            "score": [h.goals, a.goals],
            "stoppage": stoppage,
            "lineups": {code: [p.__dict__ for p in self.sides[code].club.squad[:11]]
                        for code in (self.home, self.away)},
        }
        return df, meta


def simulate_match(home: str, away: str, seed: int, match_id: str | None = None):
    """Simulate one match. Returns (events DataFrame, metadata dict)."""
    return MatchSim(home, away, seed, match_id).run()


def to_display(df: pd.DataFrame, home: str) -> pd.DataFrame:
    """Shared-pitch coordinates: home attacks right, away attacks left."""
    out = df.copy()
    away = out["team"] != home
    for a, b in (("x", "y"), ("end_x", "end_y")):
        out.loc[away, a] = PITCH_X - out.loc[away, a]
        out.loc[away, b] = PITCH_Y - out.loc[away, b]
    return out


def to_statsbomb(df: pd.DataFrame) -> pd.DataFrame:
    """StatsBomb-shaped view (list locations, pass/shot outcomes) for xT tooling."""
    out = df.copy()
    out["location"] = [[x, y] for x, y in zip(out["x"], out["y"])]
    end = [[x, y] if pd.notna(x) else None for x, y in zip(out["end_x"], out["end_y"])]
    out["pass_end_location"] = [e if t == "Pass" else None for e, t in zip(end, out["type"])]
    out["carry_end_location"] = [e if t == "Carry" else None for e, t in zip(end, out["type"])]
    out["pass_outcome"] = np.where((out["type"] == "Pass") & (out["outcome"] != "Complete"), out["outcome"], None)
    out["shot_outcome"] = np.where(out["type"] == "Shot", out["outcome"], None)
    return out
