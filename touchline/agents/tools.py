"""Agent Framework function tools over the MatchStore.

Each agent gets its own tool closures so every call is traced to the agent
that made it.
"""
from __future__ import annotations

import functools
import time
from typing import Annotated, Callable

from ..analytics.store import MatchStore


def _team_code(store: MatchStore, team: str) -> str:
    t = (team or "").strip().lower()
    for code, name in store.names.items():
        if t in (code.lower(), name.lower()) or (t and t in name.lower()):
            return code
    return store.home


def make_tools(store: MatchStore, trace: Callable[..., None], agent: str) -> dict[str, Callable]:
    def logged(fn):
        @functools.wraps(fn)  # keeps the signature so Agent Framework can build the tool schema
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            trace(agent=agent, step="tool", detail=f"{fn.__name__}({', '.join(map(str, list(args) + list(kwargs.values())))})",
                  ms=(time.perf_counter() - t0) * 1000)
            return out
        return wrapper

    def get_match_summary() -> str:
        """Final score, pre-match win probabilities and headline stats for both teams, with fact ids."""
        return store.summary_text()

    def get_turning_points() -> str:
        """Goals, red cards and big chances ranked by win-probability impact, with build-up and fact ids."""
        return store.turning_points_text()

    def get_team_report(team: Annotated[str, "Team name or 3-letter code"]) -> str:
        """Style and tactical profile of one team: possession, passing, pressing, channels, chances."""
        return store.team_text(_team_code(store, team))

    def get_momentum() -> str:
        """Sustained spells where one team dominated threat (xT + xG), with minutes and fact ids."""
        return store.momentum_text()

    def get_player_ratings(team: Annotated[str, "Team name/code, or 'all'"] = "all") -> str:
        """Top-rated players and goalscorers with rating, goals, xG, xT, key passes, defensive actions."""
        return store.players_text(None if team.lower() == "all" else _team_code(store, team))

    def get_events(start_minute: Annotated[int, "First minute of the window"],
                   end_minute: Annotated[int, "Last minute of the window"]) -> str:
        """Notable raw events (shots, cards, subs, dangerous passes/carries) in a minute window."""
        return store.events_text(int(start_minute), int(end_minute))

    def lookup_fact(fact_id: Annotated[str, "Fact id such as F12"]) -> str:
        """Look up one fact by id: its label and exact value."""
        f = store.facts.get(fact_id.strip().strip("[]"))
        return f"{f.id}: {f.label} = {f.show()}" if f else f"{fact_id} not found"

    raw = [get_match_summary, get_turning_points, get_team_report, get_momentum,
           get_player_ratings, get_events, lookup_fact]
    return {f.__name__: logged(f) for f in raw}
