# Build checkpoints

## 1. Synthetic match engine ✅
- **Works:** `touchline/sim/engine.py` simulates a full 90+ min match (~2,200 events: passes, carries, shots w/ xG, tackles, fouls, cards, corners, subs) for 6 fictional clubs in ~30 ms. Seeded and reproducible.
- **See it:** `.venv/bin/python -m pytest -q` (13 calibration tests: goals, shots, completion, xG≈goals, fouls, corners all in real-world ranges).
- **Next:** analytics (xT, win probability, turning points, fact store).

## 2. Explainable analytics + fact store ✅
- **Works:** xT per action, live win probability (Poisson w/ xG-updated rates, game state, red cards), turning points with *counterfactual* WP ("had it gone in: +23 pts"), momentum spells, pressing (PPDA, high regains), counters, channels, player ratings, storyline finder (comeback / late winner / red-card swing / smash-and-grab).
- **Fact store:** every number is a citable fact `[F#]` with evidence event ids; `MatchStore.verify(text)` flags wrong numbers, impossible scorelines and unbacked numbers.
- **Next:** Agent Framework crew (Analyst ‖ Tactician → Commentator ⇄ Fact-Checker → Producer).

## 3. Multi-agent crew (Microsoft Agent Framework) ✅
- **Works:** `touchline/agents/crew.py` — AF `WorkflowBuilder` graph: Desk → (Stats Analyst ‖ Tactician) → fan-in → Commentator ⇄ Fact-Checker (conditional revise loop, max 2) → Producer hub → (Broadcast ‖ Fan ‖ Social producers) → fan-in → Editor.
- **Recovery:** per-call timeout + retry, provider failover GitHub Models → Foundry → Ollama → offline templates, simulated-outage switch, guarded outputs (strip bad sentences or fall back to grounded template), disk cache.
- **Verified live** on local Ollama: 7 tool calls, 2 fact-check rejections → revisions → approval, 3 parallel producers (~150 s with llama3.2).
- **Next:** Streamlit UI.
