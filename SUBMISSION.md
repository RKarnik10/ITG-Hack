# Submission kit

**Challenge:** The Synthetic Match Insights Engine for Premier League Studio

## Elevator pitch (paste into the form)
**Touchline is a Synthetic Match Insights Engine for the studio. It turns a synthetic football event stream into real-time insights, key storylines and recaps that are explainable and fact-checked, for every audience.**
A seeded match engine generates a realistic ~2,200-event match (passes, shots with xG, presses, cards, subs), calibrated by tests against real-world ranges. An analytics layer computes Expected Threat, a live win-probability model and turning points ranked by impact, including counterfactuals ("had it gone in: +23 pts"). Every number is stored as a citable fact with evidence event ids. It runs at full time *or live at any moment* (half-time, 60'...), with live win probability and next-goal odds. Studios also get a timecoded clip-and-graphics cue sheet ranked by impact.

A crew of 8 agents built on **Microsoft Agent Framework** takes it from there:
- A Stats Analyst and a Tactician run in parallel with function tools over the fact store.
- A Commentator writes live lines and a match report.
- A **Fact-Checker** verifies every cited number and sends wrong drafts back with corrections.
- Broadcast, Fan and Social producers fan out the approved package. The fan recap is personalised by club, expertise and language.

Models come from **GitHub Models**, with an **Azure AI Foundry** drop-in, local Ollama and deterministic templates as automatic failover, so the desk never goes blank. Every agent step, tool call, retry and hand-off is traced and visualised.

*Problem solved:* AI can write match content at broadcast speed, but it makes up numbers. Touchline makes it explainable, verifiable and personal.

**Tech:** Microsoft Agent Framework (agents, tools, WorkflowBuilder fan-out/fan-in + conditional revise loop, WorkflowViz), GitHub Models, Azure AI Foundry-ready, Azure Container Apps-ready Dockerfile, Python, Streamlit, Plotly.

## 2-minute video script (about 15s per beat)
1. **Hook.** "AI can write match content at broadcast speed. It also makes up numbers. Touchline fixes that." Show the scoreboard: Saltcombe 2-3 Duskvale, tags Comeback / Late winner.
2. **Synthetic data.** Synthetic data tab: calibration table vs. top-flight ranges, events table. "Every event is synthetic, 2,000+ per match, calibrated by tests."
3. **Live.** Live tab, press **Replay**: the win-probability line and momentum bars build, and the commentary feed fills with agent-written lines.
4. **Real-time + explainable.** Slide *Analyse at* to Half-time: live win probability, next-goal odds, the in-play brief. Back at full time, the Intelligence tab shows key storylines and turning-point cards (win-prob swing, counterfactual, the move on a pitch, event ids).
5. **The crew.** Agent trace tab: workflow graph (exported from Agent Framework), timeline, tool calls. "Analyst and Tactician run in parallel, Commentator writes, Fact-Checker verifies."
6. **Trust.** Toggle *Inject a wrong number*, run the crew: the Fact-Checker rejects the draft ("wrote 12.04 but F6 = 1.68") and the Commentator revises.
7. **Resilience.** Toggle *Simulate outage*, run: the trace shows failover to the next provider, and output still publishes.
8. **Personal.** Studio tab: switch to *Fan of Duskvale · Expert · Spanish*. Broadcaster brief, fan recap and social pack, each with ✓ fact-checked and hoverable citations. Close on the architecture diagram in the README.

## Your 15-minute finish checklist
1. **(Optional, better text)** Create a fine-grained GitHub PAT with **Models: read** (github.com/settings/personal-access-tokens). Then `cp .env.example .env`, paste the token as `GITHUB_TOKEN`, and run `.venv/bin/python scripts/build_demo.py` (~5 min).
2. **Repo:** create an empty **public** repo on github.com, then:
   `git remote add origin https://github.com/<you>/touchline.git && git push -u origin main`
3. **Live URL (free):** share.streamlit.io → New app → pick the repo, main file `app.py` → Deploy. (Optional: add `GITHUB_TOKEN` under app Secrets.)
4. **Video:** upload `demo.mp4` (or record your own over the script above) to YouTube, unlisted or public.
5. **Submit:** paste the pitch above, the repo URL and the YouTube URL.
