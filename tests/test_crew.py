"""Integration tests for the Agent Framework crew, run fully offline (template provider)."""
from touchline.agents.crew import Persona, parse_commentary, run_crew_sync
from touchline.agents.llm import Provider
from touchline.analytics.store import MatchStore
from touchline.sim.engine import simulate_match

OFFLINE = [Provider("offline", "templates")]


def _store(seed=2031):
    df, meta = simulate_match("KBY", "NMA", seed)
    return MatchStore(df, meta)


def test_crew_publishes_verified_outputs():
    mc = run_crew_sync(_store(), Persona(team="NMA"), providers=OFFLINE)
    assert set(mc.outputs) == {"broadcast", "fan", "social"}
    assert all(mc.checks[k]["ok"] for k in ("insights", "tactics", "recap", "broadcast", "fan", "social"))
    agents = {t["agent"] for t in mc.trace}
    assert {"Stats Analyst", "Tactician", "Commentator", "Fact-Checker", "Editor"} <= agents


def test_fact_checker_catches_injected_error_and_revises():
    mc = run_crew_sync(_store(), providers=OFFLINE, inject_error=True)
    assert mc.revisions >= 1
    first = mc.verdicts[0]
    assert not first["ok"] and any(i["severity"] == "error" for i in first["issues"])
    assert mc.verdicts[-1]["ok"] and mc.checks["recap"]["ok"]


def test_outage_falls_through_the_chain():
    chain = [Provider("ollama", "missing-model", "http://127.0.0.1:9/v1", "x", timeout=2), *OFFLINE]
    mc = run_crew_sync(_store(), providers=chain, outage={"ollama"})
    assert any(t["status"] == "error" and "outage" in t["detail"] for t in mc.trace)
    assert mc.outputs["social"]


def test_live_line_parser_handles_model_formats():
    store = _store()
    text = "LIVE:\n11' [F53] | Goal!\n**86'** - Penalty\nRECAP:\nA match."
    live, recap = parse_commentary(text, store)
    assert [ln["clock"] for ln in live][:2] == ["11'", "86'"] and recap == "A match."
