"""Fictional clubs and procedurally generated squads.

Every club, player and name here is invented. Style parameters are multipliers
around 1.0 that the match engine applies on top of the league-wide zone priors.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Starting XI slots (4-3-3) with a "home" location in the team's own attacking
# frame (attacking toward x=120 on a 120x80 pitch). The engine shifts these up
# and down the pitch with the ball to pick receivers, tacklers and shooters.
SLOTS = [
    ("GK", 6, 40), ("LB", 32, 10), ("LCB", 24, 30), ("RCB", 24, 50), ("RB", 32, 70),
    ("DM", 45, 40), ("LCM", 60, 28), ("RCM", 60, 52),
    ("LW", 88, 12), ("ST", 100, 40), ("RW", 88, 68),
]
BENCH = ["GK", "RCB", "RCM", "RW", "ST"]
LINE = {"GK": "GK", "LB": "DEF", "LCB": "DEF", "RCB": "DEF", "RB": "DEF",
        "DM": "MID", "LCM": "MID", "RCM": "MID", "LW": "FWD", "ST": "FWD", "RW": "FWD"}

FIRST = ["Theo", "Malik", "Joao", "Kenji", "Luca", "Ade", "Rafa", "Ilya", "Sami", "Owen",
         "Mateo", "Kofi", "Noah", "Elias", "Dario", "Tomas", "Yusuf", "Bram", "Callum", "Rui",
         "Felix", "Aaron", "Idris", "Nico", "Jonas", "Emre", "Marek", "Sol", "Ruben", "Tariq",
         "Leon", "Abel", "Caio", "Finn", "Hugo", "Ivo", "Jude", "Kai", "Lior", "Milo"]
LAST = ["Arden", "Okafor", "Varga", "Lindqvist", "Moreau", "Castell", "Brennan", "Haddad",
        "Kowal", "Ferreira", "Nakamura", "Osei", "Duval", "Petrak", "Quinlan", "Rask",
        "Santoro", "Talbot", "Ueda", "Valdez", "Whitlow", "Yilmaz", "Zoric", "Abara",
        "Bexley", "Corran", "Dumont", "Eskel", "Farrow", "Galvan", "Hollis", "Ibarra",
        "Jansen", "Kerrigan", "Lomax", "Marsh", "Novak", "Orrin", "Penhale", "Rourke"]


@dataclass(frozen=True)
class Player:
    id: str
    name: str
    number: int
    position: str
    finishing: float
    passing: float
    aerial: float

    @property
    def line(self) -> str:
        return LINE.get(self.position, "MID")


@dataclass(frozen=True)
class Club:
    code: str
    name: str
    colour: str
    quality: float      # overall strength
    press: float        # raises opponent turnovers, mostly in their own half
    directness: float   # longer, more vertical passing
    tempo: float        # faster actions
    aggression: float   # foul and card rate
    style: str
    squad: tuple = field(default=(), compare=False, repr=False)


CLUBS_SPEC = [
    ("KBY", "Kestrel Bay FC", "#2563eb", 1.12, 1.25, 0.85, 1.10, 0.95,
     "possession side with an aggressive high press"),
    ("IRV", "Ironvale United", "#b91c1c", 1.02, 0.95, 1.30, 1.00, 1.15,
     "direct, physical, early balls into the channels"),
    ("NMA", "Northmere Athletic", "#059669", 1.06, 0.85, 1.15, 1.25, 1.00,
     "compact mid-block that breaks fast on the counter"),
    ("SCR", "Saltcombe Rovers", "#d97706", 0.97, 1.00, 1.00, 1.00, 1.00,
     "balanced, well-drilled, few risks"),
    ("EMP", "Ember Park", "#7c3aed", 0.99, 1.35, 1.05, 1.15, 1.25,
     "chaotic gegenpress, high turnover, high foul count"),
    ("DSK", "Duskvale Wanderers", "#475569", 0.92, 0.75, 1.10, 0.90, 1.05,
     "deep low block, plays for set pieces"),
]


def _make_squad(code: str) -> tuple[Player, ...]:
    rng = np.random.default_rng(sum(map(ord, code)) * 7919)  # stable per club, not hash()-salted
    names = set()
    players = []
    positions = [s[0] for s in SLOTS] + BENCH
    for i, pos in enumerate(positions):
        while True:
            name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
            if name not in names:
                names.add(name)
                break
        line = LINE[pos]
        fin_mu = {"FWD": 1.06, "MID": 0.98, "DEF": 0.88, "GK": 0.3}[line]
        aer_mu = 1.15 if pos in ("LCB", "RCB", "ST") else 0.95
        players.append(Player(
            id=f"{code}{i + 1:02d}", name=name,
            number=[1, 3, 4, 5, 2, 6, 8, 10, 11, 9, 7, 13, 15, 16, 17, 19][i],
            position=pos,
            finishing=float(np.clip(rng.normal(fin_mu, 0.08), 0.2, 1.4)),
            passing=float(np.clip(rng.normal(1.0, 0.07), 0.7, 1.3)),
            aerial=float(np.clip(rng.normal(aer_mu, 0.1), 0.6, 1.5)),
        ))
    return tuple(players)


CLUBS: dict[str, Club] = {
    spec[0]: Club(*spec, squad=_make_squad(spec[0])) for spec in CLUBS_SPEC
}


def get_club(code: str) -> Club:
    return CLUBS[code]
