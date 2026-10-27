# Build checkpoints

## 1. Synthetic match engine ✅
- **Works:** `touchline/sim/engine.py` simulates a full 90+ min match (~2,200 events: passes, carries, shots w/ xG, tackles, fouls, cards, corners, subs) for 6 fictional clubs in ~30 ms. Seeded and reproducible.
- **See it:** `.venv/bin/python -m pytest -q` (13 calibration tests: goals, shots, completion, xG≈goals, fouls, corners all in real-world ranges).
- **Next:** analytics (xT, win probability, turning points, fact store).
