"""The Touchline crew: specialist agents orchestrated as a Microsoft Agent Framework workflow.

    Desk ─┬─> Stats Analyst ─┐
          └─> Tactician ─────┴─(fan-in)─> Commentator <──(reject)── Fact-Checker
                                              └──> Fact-Checker ──(approve)──> Producer hub
    Producer hub ─┬─> Broadcast Producer ─┐
                  ├─> Fan Producer ───────┼─(fan-in)─> Editor ──> published package
                  └─> Social Producer ────┘

Shared state lives on a MatchContext blackboard. Every agent call goes through
`run_agent`, which handles timeouts, retries, provider fallback
(GitHub Models -> Foundry -> Ollama -> offline templates), caching and tracing.
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import asdict, dataclass, field

from agent_framework import Agent, Executor, WorkflowBuilder, WorkflowContext, WorkflowViz, handler

from ..analytics.store import MatchStore
from . import fallback, prompts
from .llm import Provider, ProviderOutage, available_providers, cache_get, cache_key, cache_put, make_client
from .tools import make_tools

LIVE_LINE = re.compile(r"^\s*\**\s*(\d{1,3}(?:\+\d{1,2})?)\s*['’]?\s*\**\s*(?:\[F\d+\]\s*)*[|:\-–]\s*(.+)$")
MAX_REVISIONS = 2


@dataclass
class Persona:
    team: str | None = None          # club code the fan supports, None = neutral
    expertise: str = "beginner"      # beginner | expert
    language: str = "English"

    def key(self) -> str:
        return f"{self.team or 'neutral'}-{self.expertise}-{self.language}"


@dataclass
class MatchContext:
    store: MatchStore
    persona: Persona
    providers: list[Provider]
    outage: set[str] = field(default_factory=set)
    inject_error: bool = False
    t0: float = field(default_factory=time.perf_counter)
    trace: list[dict] = field(default_factory=list)
    insights: str = ""
    tactics: str = ""
    draft: str = ""
    revisions: int = 0
    verdicts: list[dict] = field(default_factory=list)
    live: list[dict] = field(default_factory=list)
    recap: str = ""
    outputs: dict[str, str] = field(default_factory=dict)
    checks: dict[str, dict] = field(default_factory=dict)
    used: dict[str, str] = field(default_factory=dict)

    def log(self, agent: str, step: str, detail: str = "", provider: str | None = None,
            status: str = "ok", ms: float | None = None, tokens: int | None = None) -> None:
        now = (time.perf_counter() - self.t0) * 1000
        self.trace.append({"agent": agent, "step": step, "detail": detail, "provider": provider,
                           "status": status, "start_ms": round(now - (ms or 0)), "end_ms": round(now),
                           "tokens": tokens})

    def to_dict(self) -> dict:
        return {"match_id": self.store.meta["match_id"], "persona": asdict(self.persona),
                "insights": self.insights, "tactics": self.tactics, "live": self.live, "recap": self.recap,
                "revisions": self.revisions, "verdicts": self.verdicts, "outputs": self.outputs,
                "checks": self.checks, "used": self.used, "trace": self.trace,
                "providers": [p.label for p in self.providers], "outage": sorted(self.outage)}


# ------------------------------------------------------------------ agent runner
async def run_agent(mc: MatchContext, name: str, instructions: str, prompt: str,
                    tool_names: tuple[str, ...] = ()) -> tuple[str | None, Provider]:
    """Run one agent with timeout, retry, provider fallback and caching. None => use templates."""
    tools_all = make_tools(mc.store, lambda **kw: mc.log(**kw), name)
    tools = [tools_all[t] for t in tool_names]
    for p in mc.providers:
        if p.name == "offline":
            mc.log(name, "fallback", "using deterministic templates", provider=p.label, status="fallback")
            mc.used[name] = p.label
            return None, p
        key = cache_key(p.name, p.model, name, instructions, prompt, ",".join(tool_names))
        hit = cache_get(key)
        if hit and p.name not in mc.outage:
            for tc in hit.get("tools", []):
                mc.log(name, "tool", f"{tc} (replayed from cache)")
            mc.log(name, "run", "cache hit", provider=p.label, status="cached", tokens=hit.get("tokens"))
            mc.used[name] = p.label
            return hit["text"], p
        for attempt in range(2):
            t0 = time.perf_counter()
            try:
                if p.name in mc.outage:
                    raise ProviderOutage(f"{p.label} unreachable (simulated outage)")
                agent = Agent(client=make_client(p), name=name.replace(" ", "_"), instructions=instructions,
                              tools=tools or None)
                resp = await asyncio.wait_for(agent.run(prompt), timeout=p.timeout)
                text = (resp.text or "").strip()
                if not text:
                    raise RuntimeError("empty response")
                calls = [getattr(c, "name", "tool") for m in resp.messages for c in m.contents
                         if getattr(c, "type", "") == "function_call"]
                tokens = (resp.usage_details or {}).get("total_token_count")
                cache_put(key, {"text": text, "tools": calls, "tokens": tokens, "provider": p.label})
                mc.log(name, "run", f"{len(calls)} tool call(s)", provider=p.label,
                       ms=(time.perf_counter() - t0) * 1000, tokens=tokens)
                mc.used[name] = p.label
                return text, p
            except ProviderOutage as e:
                mc.log(name, "error", str(e), provider=p.label, status="error", ms=(time.perf_counter() - t0) * 1000)
                break
            except Exception as e:  # rate limit, timeout, malformed tool call ...
                msg = f"{type(e).__name__}: {str(e)[:140]}"
                mc.log(name, "retry" if attempt == 0 else "error", msg, provider=p.label, status="error",
                       ms=(time.perf_counter() - t0) * 1000)
                if attempt == 0:
                    await asyncio.sleep(1.5)
        mc.log(name, "handoff", f"failing over from {p.label}", status="fallback")
    return None, Provider("offline", "templates")


# --------------------------------------------------------------------- helpers
def parse_commentary(text: str, store: MatchStore) -> tuple[list[dict], str]:
    live, recap = [], ""
    part = text.split("RECAP:", 1)
    head = part[0].replace("LIVE:", "")
    recap = part[1].strip() if len(part) > 1 else ""
    for line in head.splitlines():
        m = LIVE_LINE.match(line.strip().lstrip("-* "))
        if m:
            live.append({"clock": m.group(1) + "'", "text": m.group(2).strip()})
    # map lines onto turning points (by clock, else by order) so each line has its evidence
    tps = store.turning_points
    by_clock = {tp["clock"]: tp for tp in tps}
    for i, ln in enumerate(live):
        tp = by_clock.get(ln["clock"]) or (tps[i] if i < len(tps) else None)
        ln["event_id"] = tp["event_id"] if tp else None
        if tp:
            ln["clock"] = tp["clock"]
    return live, recap


def sanitize(text: str, report: dict) -> str:
    """Remove sentences the verifier rejected (citations that trail a sentence go with it)."""
    for issue in report["issues"]:
        if issue["severity"] == "error" and issue.get("sentence"):
            sent = issue["sentence"]
            if sent in text:
                text = text.replace(sent, "")
            else:  # sentence was re-joined with trailing citations: drop its parts
                for part in re.split(r"\s(?=\[F\d+)", sent, maxsplit=1):
                    text = text.replace(part, "")
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]{2,}", " ", text)).strip()


def guarded(mc: "MatchContext", agent: str, text: str, template: str) -> tuple[str, dict]:
    """Verify an agent's output; strip bad sentences, or fall back to the template if too little survives."""
    store = mc.store
    rep = store.verify(text)
    if rep["ok"]:
        return text, rep
    clean = sanitize(text, rep)
    if len(clean) < 0.5 * len(text) or len(clean) < 160:
        mc.log(agent, "repair", "too much failed verification: replaced with grounded template", status="fallback")
        return template, store.verify(template)
    n = sum(i["severity"] == "error" for i in rep["issues"])
    mc.log(agent, "repair", f"removed {n} unsupported sentence(s)", status="warn")
    return clean, store.verify(clean)


def inject_wrong_number(text: str) -> str:
    """Debug: corrupt the first cited number so the Fact-Checker has something to catch."""
    m = re.search(r"(?<![-\d.])(\d+(?:\.\d+)?)(%?)(\s*\[F\d+\])", text.split("RECAP:", 1)[-1])
    if not m:
        return text
    wrong = f"{float(m.group(1)) * 3 + 7:g}"
    return text.replace(m.group(0), wrong + m.group(2) + m.group(3), 1)


# ----------------------------------------------------------------- messages
@dataclass
class Kickoff:
    match_id: str


@dataclass
class Insight:
    agent: str
    text: str


@dataclass
class Draft:
    text: str
    revision: int


@dataclass
class Verdict:
    ok: bool
    revision: int
    issues: list


@dataclass
class Approved:
    revision: int


@dataclass
class Output:
    audience: str
    text: str


# ---------------------------------------------------------------- executors
class _Step(Executor):
    def __init__(self, id: str, mc: MatchContext):
        super().__init__(id=id)
        self.mc = mc


class Desk(_Step):
    @handler
    async def start(self, msg: Kickoff, ctx: WorkflowContext[Kickoff]) -> None:
        self.mc.log("Desk", "start", f"match {msg.match_id}; providers: " + " -> ".join(p.label for p in self.mc.providers))
        await ctx.send_message(msg)


class AnalystStep(_Step):
    @handler
    async def run(self, msg: Kickoff, ctx: WorkflowContext[Insight]) -> None:
        s = self.mc.store
        prompt = f"MATCH SUMMARY:\n{s.summary_text()}\n\nTURNING POINTS:\n{s.turning_points_text()}\n\nWrite the 5 insights."
        text, _ = await run_agent(self.mc, "Stats Analyst", prompts.ANALYST, prompt, ("get_player_ratings", "get_events"))
        self.mc.insights = text or fallback.analyst(s)
        await ctx.send_message(Insight("analyst", self.mc.insights))


class TacticianStep(_Step):
    @handler
    async def run(self, msg: Kickoff, ctx: WorkflowContext[Insight]) -> None:
        s = self.mc.store
        prompt = (f"MATCH SUMMARY:\n{s.summary_text()}\n\nHome team: {s.names[s.home]}. Away team: {s.names[s.away]}.\n"
                  f"MOMENTUM:\n{s.momentum_text()}\n\nInvestigate both teams with your tools, then write the 3 sections.")
        text, _ = await run_agent(self.mc, "Tactician", prompts.TACTICIAN, prompt, ("get_team_report", "get_momentum"))
        self.mc.tactics = text or fallback.tactician(s)
        await ctx.send_message(Insight("tactician", self.mc.tactics))


class CommentatorStep(_Step):
    @handler
    async def write(self, insights: list[Insight], ctx: WorkflowContext[Draft]) -> None:
        s = self.mc.store
        prompt = (f"MATCH SUMMARY:\n{s.summary_text()}\n\nTURNING POINTS (one LIVE line each):\n{s.turning_points_text()}\n\n"
                  + "\n\n".join(f"{i.agent.upper()} NOTES:\n{i.text}" for i in insights))
        text, _ = await run_agent(self.mc, "Commentator", prompts.COMMENTATOR, prompt)
        draft = text or fallback.commentary(s)
        if self.mc.inject_error:
            draft = inject_wrong_number(draft)
            self.mc.log("Commentator", "debug", "injected a wrong number to exercise the fact-check loop", status="warn")
        self.mc.draft = draft
        await ctx.send_message(Draft(draft, 0))

    @handler
    async def revise(self, verdict: Verdict, ctx: WorkflowContext[Draft]) -> None:
        self.mc.revisions = verdict.revision + 1
        issues = "\n".join(f"- {i['problem']} (in: \"{i['sentence'][:160]}\")" for i in verdict.issues[:8])
        self.mc.log("Commentator", "handoff", f"revision {self.mc.revisions} requested by Fact-Checker", status="warn")
        prompt = prompts.REVISE.format(issues=issues, draft=self.mc.draft)
        text, _ = await run_agent(self.mc, "Commentator", prompts.COMMENTATOR + "\n" + prompts.GROUNDING, prompt)
        if not text or "RECAP:" not in text:
            text = sanitize(self.mc.draft, {"issues": verdict.issues})
            self.mc.log("Commentator", "repair", "removed rejected sentences (no LLM revision available)", status="fallback")
        self.mc.draft = text
        await ctx.send_message(Draft(text, self.mc.revisions))


class FactCheckerStep(_Step):
    @handler
    async def check(self, draft: Draft, ctx: WorkflowContext[Verdict]) -> None:
        s, mc = self.mc.store, self.mc
        t0 = time.perf_counter()
        report = s.verify(draft.text)
        issues = list(report["issues"])
        mc.log("Fact-Checker", "verify", f"rev {draft.revision}: {report['citations_checked']} citations, "
               f"{sum(i['severity'] == 'error' for i in issues)} errors, {sum(i['severity'] == 'warning' for i in issues)} warnings",
               status="ok" if report["ok"] else "warn", ms=(time.perf_counter() - t0) * 1000)
        if draft.revision == 0:  # one LLM review pass for qualitative claims
            recap = draft.text.split("RECAP:", 1)[-1]
            cited = [i.strip() for grp in re.findall(r"\[(F\d+(?:\s*,\s*F\d+)*)\]", recap) for i in grp.split(",")]
            ids = list(dict.fromkeys(s.core_fact_ids() + cited))
            prompt = f"FACTS:\n{s.fact_sheet(ids)}\n\nDRAFT:\n{recap}"
            text, _ = await run_agent(mc, "Fact-Checker", prompts.FACT_CHECKER, prompt, ("lookup_fact",))
            if text and "NONE" not in text.upper()[:12]:
                for line in [ln[2:].strip() for ln in text.splitlines() if ln.strip().startswith("- ")][:3]:
                    issues.append({"sentence": "", "problem": f"reviewer: {line}", "severity": "review"})
        blocking = [i for i in issues if i["severity"] == "error"] or \
                   ([i for i in issues if i["severity"] == "review"] if draft.revision == 0 else [])
        ok = not blocking
        mc.verdicts.append({"revision": draft.revision, "ok": ok, "issues": issues})
        if not ok and draft.revision >= MAX_REVISIONS:
            mc.draft = sanitize(draft.text, report)
            mc.log("Fact-Checker", "repair", "revision budget spent: stripped unsupported sentences", status="fallback")
            ok = True
        if ok:
            mc.live, mc.recap = parse_commentary(mc.draft, s)
            if not mc.live or not mc.recap:
                fb_live, fb_recap = parse_commentary(fallback.commentary(s), s)
                mc.live, mc.recap = mc.live or fb_live, mc.recap or fb_recap
                mc.log("Fact-Checker", "repair", "filled missing LIVE/RECAP section from templates", status="fallback")
            # analyst / tactician notes are verified too; bad sentences are removed, not rewritten
            mc.insights, mc.checks["insights"] = guarded(mc, "Stats Analyst", mc.insights, fallback.analyst(s))
            mc.tactics, mc.checks["tactics"] = guarded(mc, "Tactician", mc.tactics, fallback.tactician(s))
            mc.recap, mc.checks["recap"] = guarded(mc, "Commentator", mc.recap, fallback.recap(s))
            mc.log("Fact-Checker", "approve", f"approved after {draft.revision} revision(s)")
        await ctx.send_message(Verdict(ok, draft.revision, issues))


class ProducerHub(_Step):
    @handler
    async def fan_out(self, verdict: Verdict, ctx: WorkflowContext[Approved]) -> None:
        self.mc.log("Producer hub", "handoff", "approved package sent to 3 audience producers")
        await ctx.send_message(Approved(verdict.revision))


class AudienceStep(_Step):
    def __init__(self, id: str, mc: MatchContext, audience: str):
        super().__init__(id, mc)
        self.audience = audience

    @handler
    async def run(self, msg: Approved, ctx: WorkflowContext[Output]) -> None:
        s, mc, pr = self.mc.store, self.mc, self.mc.persona
        base = f"APPROVED RECAP:\n{mc.recap}\n\nKEY INSIGHTS:\n{mc.insights}\n\nTURNING POINTS:\n{s.turning_points_text()}"
        if self.audience == "broadcast":
            name, instr = "Broadcast Producer", prompts.BROADCAST
        elif self.audience == "social":
            name, instr = "Social Producer", prompts.SOCIAL
        else:
            name = "Fan Producer"
            who = f"a passionate {s.names[pr.team]} supporter (be partisan but honest)" if pr.team else "a neutral football fan"
            instr = prompts.FAN.format(audience=who, expertise=pr.expertise, language=pr.language,
                                       expertise_rule=prompts.EXPERTISE_RULE.get(pr.expertise, ""))
        text, _ = await run_agent(mc, name, instr, base)
        if not text:
            text = {"broadcast": fallback.broadcast, "social": fallback.social}.get(
                self.audience, lambda st: fallback.fan(st, asdict(pr)))(s)
        text = re.sub(r"^(here (are|is)[^\n]*:\s*)", "", text, flags=re.I).strip()  # drop chatty preambles
        template = {"broadcast": fallback.broadcast, "social": fallback.social}.get(
            self.audience, lambda st: fallback.fan(st, asdict(pr)))(s)
        text, mc.checks[self.audience] = guarded(mc, name, text, template)
        await ctx.send_message(Output(self.audience, text))


class Editor(_Step):
    @handler
    async def publish(self, outputs: list[Output], ctx: WorkflowContext[None, dict]) -> None:
        for o in outputs:
            self.mc.outputs[o.audience] = o.text
        n_ok = sum(1 for k in ("broadcast", "fan", "social") if self.mc.checks.get(k, {}).get("ok"))
        self.mc.log("Editor", "publish", f"{len(outputs)} outputs, {n_ok} fully verified")
        await ctx.yield_output(self.mc.to_dict())


def build_workflow(mc: MatchContext):
    desk, analyst, tactician = Desk("desk", mc), AnalystStep("stats_analyst", mc), TacticianStep("tactician", mc)
    commentator, checker = CommentatorStep("commentator", mc), FactCheckerStep("fact_checker", mc)
    hub = ProducerHub("producer_hub", mc)
    audiences = [AudienceStep("broadcast_producer", mc, "broadcast"), AudienceStep("fan_producer", mc, "fan"),
                 AudienceStep("social_producer", mc, "social")]
    editor = Editor("editor", mc)
    return (WorkflowBuilder(start_executor=desk, name="touchline-crew", max_iterations=40)
            .add_fan_out_edges(desk, [analyst, tactician])
            .add_fan_in_edges([analyst, tactician], commentator)
            .add_edge(commentator, checker)
            .add_edge(checker, commentator, condition=lambda v: not v.ok)
            .add_edge(checker, hub, condition=lambda v: v.ok)
            .add_fan_out_edges(hub, audiences)
            .add_fan_in_edges(audiences, editor)
            .build())


async def run_crew(store: MatchStore, persona: Persona | None = None, outage: set[str] | None = None,
                   inject_error: bool = False, providers: list[Provider] | None = None) -> MatchContext:
    mc = MatchContext(store, persona or Persona(), providers or available_providers(), set(outage or ()), inject_error)
    wf = build_workflow(mc)
    await wf.run(Kickoff(store.meta["match_id"]))
    mc.log("Desk", "done", f"{len(mc.trace)} trace events")
    return mc


def run_crew_sync(*args, **kwargs) -> MatchContext:
    return asyncio.run(run_crew(*args, **kwargs))


async def ask(store: MatchStore, question: str, outage: set[str] | None = None,
              providers: list[Provider] | None = None) -> dict:
    mc = MatchContext(store, Persona(), providers or available_providers(), set(outage or ()))
    text, p = await run_agent(mc, "Match Q&A", prompts.QA, f"QUESTION: {question}\n\nMATCH SUMMARY:\n{store.summary_text()}",
                              ("get_turning_points", "get_team_report", "get_momentum", "get_player_ratings",
                               "get_events", "lookup_fact"))
    text = text or fallback.answer(store, question)
    return {"answer": text, "check": store.verify(text), "provider": p.label, "trace": mc.trace}


def crew_mermaid() -> str:
    from ..sim.engine import simulate_match
    df, meta = simulate_match("KBY", "IRV", 1)
    mc = MatchContext(MatchStore(df, meta), Persona(), [Provider("offline", "templates")])
    return WorkflowViz(build_workflow(mc)).to_mermaid()


def start_crew(store: MatchStore, persona: Persona | None = None, outage: set[str] | None = None,
               inject_error: bool = False, providers: list[Provider] | None = None):
    """Run the crew on a background thread so a UI can stream the trace while it works.

    Returns (MatchContext, thread); the context fills in as agents finish.
    """
    import threading
    mc = MatchContext(store, persona or Persona(), providers or available_providers(), set(outage or ()), inject_error)

    def _run():
        try:
            asyncio.run(build_workflow(mc).run(Kickoff(store.meta["match_id"])))
            mc.log("Desk", "done", f"{len(mc.trace)} trace events")
        except Exception as e:  # never leave the UI hanging
            mc.log("Desk", "error", f"workflow failed: {type(e).__name__}: {e}", status="error")

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    return mc, th


def workflow_dot() -> str:
    """Graphviz DOT of the real crew workflow, styled for a dark UI."""
    from ..sim.engine import simulate_match
    df, meta = simulate_match("KBY", "IRV", 1)
    mc = MatchContext(MatchStore(df, meta), Persona(), [Provider("offline", "templates")])
    dot = WorkflowViz(build_workflow(mc)).to_digraph()
    dot = re.sub(r",?\s*fillcolor=\"?[#\w]+\"?", "", dot)          # use our dark node style
    dot = re.sub(r",?\s*style=\"?filled\"?", "", dot).replace("[, ", "[").replace(" []", "")
    style = ('bgcolor="transparent"; rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#1f2937", '
             'fontcolor="#f0efec", color="#3987e5", fontname="Helvetica"]; '
             'edge [color="#9ca3af", fontcolor="#c3c2b7", fontname="Helvetica"];')
    return dot.replace("{", "{ " + style, 1)
