"""Find the most dramatic synthetic matches and precompute crew runs for the demo.

    python scripts/build_demo.py              # storyline finder + crew runs (uses the provider chain)
    python scripts/build_demo.py --no-crew    # only pick the demo matches
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from touchline.agents.crew import Persona, run_crew_sync  # noqa: E402
from touchline.analytics.store import MatchStore  # noqa: E402
from touchline.sim.engine import simulate_match  # noqa: E402
from touchline.sim.teams import CLUBS  # noqa: E402

CACHE = ROOT / "cache"


def find_storylines(n: int) -> list[dict]:
    pairs = list(itertools.permutations(CLUBS, 2))
    found = []
    for i in range(n):
        home, away = pairs[i % len(pairs)]
        seed = 3000 + i
        df, meta = simulate_match(home, away, seed)
        store = MatchStore(df, meta)
        found.append({"home": home, "away": away, "seed": seed, "match_id": meta["match_id"],
                      "label": f"{meta['home_name']} {meta['score'][0]}-{meta['score'][1]} {meta['away_name']}",
                      **store.storyline})
    found.sort(key=lambda m: -m["drama"])
    picks, used_tags = [], set()
    for m in found:  # most dramatic first, but prefer a different headline tag each time
        head = m["tags"][0] if m["tags"] else ""
        if head in used_tags and len(found) > 10:
            continue
        picks.append(m)
        used_tags.add(head)
        if len(picks) == 3:
            break
    return picks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--no-crew", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    picks = find_storylines(args.n)
    CACHE.mkdir(exist_ok=True)
    (CACHE / "demo_matches.json").write_text(json.dumps(picks, indent=1))
    print(f"picked in {time.time() - t0:.0f}s:", *[f"  {p['label']}  {p['tags']}  drama={p['drama']}" for p in picks], sep="\n")
    if args.no_crew:
        return
    runs = CACHE / "runs"
    runs.mkdir(exist_ok=True)
    jobs = [(p, Persona()) for p in picks]
    first = picks[0]
    jobs += [(first, Persona(first["home"], "expert", "English")), (first, Persona(first["away"], "beginner", "Spanish"))]
    for p, persona in jobs:
        t = time.time()
        df, meta = simulate_match(p["home"], p["away"], p["seed"])
        mc = run_crew_sync(MatchStore(df, meta), persona)
        (runs / f"{meta['match_id']}__{persona.key()}.json").write_text(json.dumps(mc.to_dict(), indent=1, default=str))
        print(f"crew {meta['match_id']} {persona.key()}: {time.time() - t:.0f}s, revisions={mc.revisions}, "
              f"providers={sorted(set(mc.used.values()))}")
    # the fact-check demo: a run with a deliberately corrupted number
    df, meta = simulate_match(first["home"], first["away"], first["seed"])
    mc = run_crew_sync(MatchStore(df, meta), Persona(), inject_error=True)
    (runs / f"{meta['match_id']}__{Persona().key()}__inject.json").write_text(json.dumps(mc.to_dict(), indent=1, default=str))
    print("inject run: revisions", mc.revisions)


if __name__ == "__main__":
    main()
