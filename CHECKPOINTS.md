# Build checkpoints

## 1. Synthetic match engine ✅
- **Works:** `touchline/sim/engine.py` simulates a full 90+ min match (~2,200 events: passes, carries, shots w/ xG, tackles, fouls, cards, corners, subs) for 6 fictional clubs in ~30 ms. Seeded and reproducible.
- **See it:** `.venv/bin/python -m pytest -q` (13 calibration tests: goals, shots, completion, xG≈goals, fouls, corners all in real-world ranges).
- **Next:** analytics (xT, win probability, turning points, fact store).

## 2. Explainable analytics + fact store ✅
- **Works:** xT per action, live win probability (Poisson w/ xG-updated rates, game state, red cards), turning points with *counterfactual* WP ("had it gone in: +23 pts"), momentum spells, pressing (PPDA, high regains), counters, channels, player ratings, storyline finder (comeback / late winner / red-card swing / smash-and-grab).
- **Fact store:** every number is a citable fact `[F#]` with evidence event ids; `MatchStore.verify(text)` flags wrong numbers, impossible scorelines and unbacked numbers.
- **Next:** Agent Framework crew (Analyst ‖ Tactician → Commentator ⇄ Fact-Checker → Producer).
