"""Touchline: multi-agent match intelligence on synthetic football data.

    streamlit run app.py
"""
from __future__ import annotations

import asyncio
import html
import json
import re
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from touchline import viz
from touchline.agents.crew import Persona, ask, start_crew, workflow_dot
from touchline.agents.llm import available_providers
from touchline.analytics.store import MOMENTS, MatchStore, cut_match
from touchline.sim.engine import simulate_match
from touchline.sim.teams import CLUBS

ROOT = Path(__file__).parent
RUNS = ROOT / "cache" / "runs"
DEMO_INDEX = ROOT / "cache" / "demo_matches.json"
LANGS = ["English", "Spanish", "French", "Portuguese", "German", "Hindi", "Japanese"]

st.set_page_config(page_title="Touchline · Synthetic Match Insights Engine", page_icon="⚽", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; max-width: 1400px;}
.cite {font: 600 0.68rem ui-monospace, Menlo, monospace; background: rgba(57,135,229,.18); color: #9cc3f5;
       border-radius: 4px; padding: 0 4px; margin-left: 2px; cursor: help; white-space: nowrap;}
.cite.bad {background: rgba(208,59,59,.25); color: #f3a5a5;}
.score {font-size: 2.6rem; font-weight: 800; text-align: center; letter-spacing: .05em; color: #f0efec;}
.team {font-size: 1.35rem; font-weight: 700; padding-top: .9rem;}
.chip {display: inline-block; background: #2a2a28; color: #c3c2b7; border-radius: 999px; padding: 2px 10px;
       margin: 2px 4px 2px 0; font-size: .78rem;}
.feed {border-left: 3px solid #3a3a37; padding: 4px 0 6px 10px; margin-bottom: 8px;}
.feed.home {border-color: #3987e5;} .feed.away {border-color: #d95926;}
.feed .clk {font-weight: 700; color: #f0efec; margin-right: 6px;}
.muted {color: #9a998f; font-size: .85rem;}
.live {display: inline-block; background: #d03b3b; color: #fff; border-radius: 6px; padding: 1px 10px;
        font-weight: 700; font-size: .8rem; letter-spacing: .06em;}
.badge-ok {color: #0ca30c; font-weight: 600;} .badge-warn {color: #ec835a; font-weight: 600;}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data
@st.cache_resource(show_spinner=False)
def get_store(home: str, away: str, seed: int, moment: str = "Full time") -> MatchStore:
    df, meta = simulate_match(home, away, seed)
    return MatchStore(*cut_match(df, meta, moment))


@st.cache_data(show_spinner=False)
def demo_matches() -> list[dict]:
    if DEMO_INDEX.exists():
        return json.loads(DEMO_INDEX.read_text())
    return [{"home": "KBY", "away": "NMA", "seed": 2031, "label": "Kestrel Bay FC 2-1 Northmere Athletic"}]


@st.cache_data(show_spinner=False)
def cached_dot() -> str:
    return workflow_dot()


def run_path(match_id: str, persona: Persona, flags: str = "") -> Path:
    return RUNS / f"{match_id}__{persona.key()}{('__' + flags) if flags else ''}.json"


def load_run(match_id: str, persona: Persona, flags: str = "") -> dict | None:
    p = run_path(match_id, persona, flags)
    if p.exists():
        return json.loads(p.read_text())
    # fall back to any run of this match (different persona) so the page is never empty
    if not flags:
        for q in sorted(RUNS.glob(f"{match_id}__*.json")):
            if q.name.count("__") == 1:
                return json.loads(q.read_text())
    return None


def cite(text: str, store: MatchStore) -> str:
    """Escape agent text and turn [F#] citations into hoverable chips (label = value)."""
    def chip(m):
        out = []
        for fid in [x.strip() for x in m.group(1).split(",")]:
            f = store.facts.get(fid)
            title = html.escape(f"{fid}: {f.label} = {f.show()}") if f else f"{fid}: unknown fact"
            out.append(f'<span class="cite{"" if f else " bad"}" title="{title}">{fid}</span>')
        return "".join(out)
    text = re.sub(r"\s*\[(?!F\d)[A-Z]{1,3}\d+\]", "", text or "")       # pseudo-citations from the model
    safe = html.escape(text).replace("&#x27;", "'")
    safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe).replace("**", "")
    safe = re.sub(r"^#{1,4}\s*(.+)$", r"<b>\1</b>", safe, flags=re.M)
    safe = re.sub(r"\[(F\d+(?:\s*,\s*F\d+)*)\]", chip, safe)
    safe = re.sub(r"^\s*[-*] ", "• ", safe, flags=re.M)
    return safe.replace("\n", "<br>")


def poss(name: str) -> str:
    return name + ("'" if name.endswith("s") else "'s")


def badge(check: dict | None) -> str:
    if not check:
        return ""
    errs = [i for i in check["issues"] if i["severity"] == "error"]
    if check["ok"]:
        return f'<span class="badge-ok">✓ Fact-checked · {check["citations_checked"]} citations verified</span>'
    return f'<span class="badge-warn">! {len(errs)} error(s), {len(check["issues"]) - len(errs)} note(s)</span>'


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## ⚽ Touchline")
    st.caption("Synthetic Match Insights Engine for the studio · Microsoft Agent Framework")
    demos = demo_matches()
    labels = [d["label"] + (f"  ·  {', '.join(d.get('tags', []))}" if d.get("tags") else "") for d in demos]
    choice = st.selectbox("Match", range(len(demos) + 1), format_func=lambda i: labels[i] if i < len(demos) else "🎲 Simulate a new match…")
    if choice < len(demos):
        home, away, seed = demos[choice]["home"], demos[choice]["away"], int(demos[choice]["seed"])
    else:
        codes = list(CLUBS)
        c1, c2 = st.columns(2)
        home = c1.selectbox("Home", codes, format_func=lambda c: CLUBS[c].name)
        away = c2.selectbox("Away", [c for c in codes if c != home], format_func=lambda c: CLUBS[c].name)
        seed = int(st.number_input("Seed", 1, 99999, 42))
    moment = st.select_slider("Analyse the match at", options=list(MOMENTS), value="Full time",
                              help="Real-time mode: the engine and the crew only see events up to this moment.")
    store = get_store(home, away, seed, moment)
    meta = store.meta

    st.markdown("#### Personalise")
    fan_of = st.radio("Watching as", ["neutral", home, away], horizontal=True,
                      format_func=lambda c: "Neutral" if c == "neutral" else f"{CLUBS[c].name} fan")
    expertise = st.radio("Level", ["beginner", "expert"], horizontal=True, format_func=str.title)
    language = st.selectbox("Language", LANGS)
    persona = Persona(None if fan_of == "neutral" else fan_of, expertise, language)

    st.markdown("#### AI crew")
    providers = available_providers()
    st.caption("Provider chain: " + " → ".join(p.label for p in providers))
    outage = {p.name for p in providers if p.name != "offline"
              and st.toggle(f"Simulate {p.label.split(' · ')[0]} outage", False, key=f"out-{p.name}")}
    inject = st.toggle("Inject a wrong number (fact-check demo)", False)
    flags = "-".join(sorted(outage) + (["inject"] if inject else []))
    run_now = st.button("▶ Run the AI crew", type="primary", use_container_width=True)

# ------------------------------------------------------------------ crew run
if run_now:
    mc, th = start_crew(store, persona, outage=outage, inject_error=inject, providers=providers)
    with st.status("The crew is on it…", expanded=True) as status:
        feed = st.empty()
        while th.is_alive():
            rows = [f"`{t['end_ms'] / 1000:6.1f}s` **{t['agent']}** · {t['step']} · {t['detail'][:110]}"
                    for t in mc.trace[-12:]]
            feed.markdown("\n\n".join(rows))
            time.sleep(0.4)
        feed.markdown("\n\n".join(f"`{t['end_ms'] / 1000:6.1f}s` **{t['agent']}** · {t['step']} · {t['detail'][:110]}"
                                  for t in mc.trace[-12:]))
        status.update(label=f"Published · {len(mc.trace)} agent events · {mc.revisions} fact-check revision(s)",
                      state="complete", expanded=False)
    RUNS.mkdir(parents=True, exist_ok=True)
    run_path(meta["match_id"], persona, flags).write_text(json.dumps(mc.to_dict(), indent=1, default=str))

run = load_run(meta["match_id"], persona, flags)

# ------------------------------------------------------------------ header
h, a = meta["score"]
c1, c2, c3 = st.columns([5, 2, 5])
c1.markdown(f'<div class="team" style="text-align:right;color:{viz.HOME}">{meta["home_name"]}</div>', unsafe_allow_html=True)
c2.markdown(f'<div class="score">{h} – {a}</div>' + (f'<div style="text-align:center"><span class="live">● LIVE · {moment.upper()}</span></div>'
            if store.live else '<div style="text-align:center" class="muted">FULL TIME</div>'), unsafe_allow_html=True)
c3.markdown(f'<div class="team" style="color:{viz.AWAY}">{meta["away_name"]}</div>', unsafe_allow_html=True)
chips = "".join(f'<span class="chip">{t}</span>' for t in store.storyline["tags"])
st.markdown(f'<div style="text-align:center">{chips}<span class="chip">synthetic match · seed {seed}</span>'
            f'<span class="chip">{len(store.df):,} events</span><span class="chip">{len(store.facts)} verified facts</span></div>',
            unsafe_allow_html=True)
if run is None:
    st.info("No AI run for this match yet. Press **▶ Run the AI crew** in the sidebar. The analytics below are live already.")
elif run["persona"] != persona.__dict__ or run.get("outage") != sorted(outage):
    rp = run["persona"]
    shown = f"{CLUBS[rp['team']].name + ' fan' if rp['team'] else 'neutral'} · {rp['expertise']} · {rp['language']}"
    st.info(f"Showing the saved run for **{shown}**. Press **▶ Run the AI crew** to produce it for your settings.")

tab_live, tab_intel, tab_studio, tab_trace, tab_ask, tab_data = st.tabs(
    ["📺 Live", "🧠 Intelligence", "🎙️ Studio", "🤖 Agent trace", "💬 Ask the match", "🧪 Synthetic data"])

# ------------------------------------------------------------------ live
with tab_live:
    end_min = int(store.df.elapsed.max() // 60) + 1
    if "minute" not in st.session_state or st.session_state.get("minute_match") != meta["match_id"]:
        st.session_state.minute, st.session_state.minute_match = end_min, meta["match_id"]
    cc1, cc2 = st.columns([6, 1])
    minute = cc1.slider("Match clock (minutes elapsed)", 1, end_min, st.session_state.minute)
    play = cc2.button("▶ Replay", use_container_width=True)
    left, right = st.columns([3, 2])
    ph_wp, ph_mom, ph_feed, ph_pitch = left.empty(), left.empty(), right.empty(), right.empty()

    live_lines = {ln.get("event_id"): ln["text"] for ln in (run or {}).get("live", [])}
    feed_events = store.df[(store.df.type == "Shot") & store.df.outcome.isin(["Goal", "Saved"])
                           | (store.df.type == "Card") | (store.df.type == "Substitution")]
    tp_ids = {tp["event_id"]: tp for tp in store.turning_points}

    def render(mn: int) -> None:
        upto = mn * 60
        ph_wp.plotly_chart(viz.win_prob(store, upto), use_container_width=True, key=f"wp{mn}")
        ph_mom.plotly_chart(viz.momentum(store, upto), use_container_width=True, key=f"mom{mn}")
        items = []
        for r in feed_events[feed_events.elapsed <= upto].iloc[::-1].head(14).itertuples():
            side = "home" if r.team == meta["home"] else "away"
            tp = tp_ids.get(r.event_id)
            if tp and r.event_id in live_lines:
                body = cite(live_lines[r.event_id], store)
            elif tp:
                body = f"<b>{tp['kind']}</b> · {r.player} · win prob {tp['wp_before']}% → {tp['wp_after']}%"
            elif r.type == "Shot":
                body = f"{'⚽ GOAL' if r.outcome == 'Goal' else 'Save'} · {r.player} (xG {r.xg:.2f})"
            elif r.type == "Card":
                body = f"{'🟨' if r.outcome == 'Yellow Card' else '🟥'} {r.player}"
            else:
                body = f"🔁 {r.outcome} on for {r.player}"
            items.append(f'<div class="feed {side}"><span class="clk">{r.clock}</span>{body}</div>')
        for tp in store.turning_points:  # big misses aren't in the event filter above
            if tp["kind"].endswith("missed") and tp["minute"] * 60 <= upto and tp["event_id"] in live_lines:
                side = "home" if tp["team"] == meta["home"] else "away"
                items.insert(0, f'<div class="feed {side}"><span class="clk">{tp["clock"]}</span>{cite(live_lines[tp["event_id"]], store)}</div>')
        ph_feed.markdown("<h4>Live commentary</h4>" + ("".join(items) or '<span class="muted">Kick-off…</span>'),
                         unsafe_allow_html=True)
        ph_pitch.plotly_chart(viz.shot_map(store.df, meta, upto, height=300), use_container_width=True, key=f"pitch{mn}")

    if play:
        for mn in range(max(1, minute if minute < end_min else 1), end_min + 1, 2):
            render(mn)
            time.sleep(0.12)
        st.session_state.minute = end_min
    else:
        render(minute)

# ------------------------------------------------------------------ intelligence
with tab_intel:
    if store.live:
        f = store.f_live
        st.markdown(f"### Right now, at {moment}")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"{meta['home_name']} win", store.facts[f['wp_home']].show())
        k2.metric("Draw", store.facts[f['wp_draw']].show())
        k3.metric(f"{meta['away_name']} win", store.facts[f['wp_away']].show())
        k4.metric("Another goal?", store.facts[f['more']].show(),
                  f"next: {meta['home'] if store.facts[f['next_home']].value >= store.facts[f['next_away']].value else meta['away']}",
                  delta_color="off")
    st.markdown("### Key storylines")
    lines = []
    goals = store.df[(store.df.type == "Shot") & (store.df.outcome == "Goal")]
    fx = {t: store.facts[store.f_team[t]["xg"]].value for t in (meta["home"], meta["away"])}
    for tag in store.storyline["tags"]:
        if tag == "Comeback":
            lines.append(f"**Comeback**: {poss(store.names[meta['home'] if h > a else meta['away']])} win probability fell as low as "
                         f"{max(100 * store.wp[:, 0 if h > a else 2].min(), 0.1):.1f}% before they turned it round.")
        elif tag in ("Late winner", "Late equaliser") and len(goals):
            g = goals.iloc[-1]
            lines.append(f"**{tag}**: {g.player} at {g.clock} for {store.names[g.team]}.")
        elif tag == "Smash and grab":
            w = meta["home"] if h > a else meta["away"]
            lines.append(f"**Smash and grab**: {store.names[w]} won despite xG {fx[w]:.2f} vs "
                         f"{fx[meta['away'] if w == meta['home'] else meta['home']]:.2f}.")
        elif tag == "Red-card swing":
            r = next(t for t in store.turning_points if t["kind"] == "Red card")
            lines.append(f"**Red-card swing**: {r['player']} ({r['team_name']}) sent off at {r['clock']}.")
        elif tag == "Goal-fest":
            lines.append(f"**Goal-fest**: {h + a} goals.")
    for sp in store.spells[:3]:
        lines.append(f"**Pressure spell**: {store.names[sp['team']]} dominated {sp['start']}'–{sp['end']}' "
                     f"(threat {sp['threat_for']:.2f} vs {sp['threat_against']:.2f}).")
    st.markdown("\n".join(f"- {ln}" for ln in lines) or "_No standout storyline yet._")
    st.markdown("### Turning points: what happened, why it mattered, and the evidence")
    for i in range(0, len(store.turning_points), 2):
        cols = st.columns(2)
        for col, tp in zip(cols, store.turning_points[i:i + 2]):
            colour = viz.HOME if tp["team"] == meta["home"] else viz.AWAY
            with col.container(border=True):
                st.markdown(f"**{tp['clock']} · {tp['kind']}** · {tp['player']} "
                            f"<span style='color:{colour}'>({tp['team_name']})</span> · score {tp['score_after']}",
                            unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("Win prob", f"{tp['wp_after']}%", f"{tp['dwp']:+d} pts")
                m2.metric("Chance xG", "-" if tp["xg"] is None else f"{tp['xg']:.2f}")
                m3.metric("If it had gone in" if tp["counterfactual_dwp"] is not None else "Build-up xT",
                          f"+{tp['counterfactual_dwp']} pts" if tp["counterfactual_dwp"] is not None else f"{tp['buildup_xt']:.2f}")
                why = (f"{poss(tp['team_name'])} chance of winning moved from {tp['wp_before']}% to {tp['wp_after']}%."
                       if tp["counterfactual_dwp"] is None else
                       f"A goal here was worth {tp['counterfactual_dwp']} points of win probability. It stayed out, so the match stayed at {tp['wp_after']}%.")
                st.caption(f"Why it mattered: {why} The move ({tp['pattern'].lower()}) built {tp['buildup_xt']:.2f} xT over "
                           f"{len(tp['chain'])} actions. Evidence: events {', '.join('#' + str(e) for e in tp['evidence'][-6:])}.")
                st.plotly_chart(viz.chain_map(tp["chain"], colour, height=300), use_container_width=True,
                                key=f"chain{tp['event_id']}")
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📊 Stats Analyst")
        if run:
            st.markdown(badge(run["checks"].get("insights")) + "<br>" + cite(run["insights"], store), unsafe_allow_html=True)
        else:
            st.caption("Run the AI crew to generate insights.")
    with c2:
        st.markdown("### 🧩 Tactician")
        if run:
            st.markdown(badge(run["checks"].get("tactics")) + "<br>" + cite(run["tactics"], store), unsafe_allow_html=True)
        else:
            st.caption("Run the AI crew to generate the tactical report.")
    st.divider()
    st.markdown("### Team comparison (every number is a citable fact)")
    rows = []
    for key, label in [("goals", "Goals"), ("xg", "xG"), ("shots", "Shots"), ("shots_on_target", "On target"),
                       ("possession_pct", "Possession %"), ("pass_completion_pct", "Pass completion %"),
                       ("xt", "xT generated"), ("ppda", "PPDA (press)"), ("high_regains", "Regains ≤40m from goal"),
                       ("counters_to_shot", "Counters → shot"), ("final_third_entries", "Final-third entries"),
                       ("set_piece_xg", "Set-piece xG"), ("fouls", "Fouls"), ("yellows", "Yellows"), ("reds", "Reds")]:
        fh, fa = store.f_team[meta["home"]][key], store.f_team[meta["away"]][key]
        rows.append({"Metric": label, meta["home_name"]: store.facts[fh].show(), meta["away_name"]: store.facts[fa].show(),
                     "Fact ids": f"{fh} · {fa}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.markdown("### Player ratings")
    pt = store.players.copy()
    pt["team"] = pt.team.map(store.names)
    st.dataframe(pt[["player", "team", "position", "rating", "goals", "assists", "xg", "xt", "key_passes", "def_actions", "pass_pct"]]
                 .head(16).round(2), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ studio
with tab_studio:
    if not run:
        st.info("Run the AI crew to produce broadcaster, fan and social outputs.")
    else:
        st.caption(f"Persona: {'neutral' if not run['persona']['team'] else CLUBS[run['persona']['team']].name + ' fan'} · "
                   f"{run['persona']['expertise']} · {run['persona']['language']} · hover any blue chip to see the fact behind it")
        c1, c2, c3 = st.columns(3)
        for col, key, title in ((c1, "broadcast", "📺 Broadcaster brief"), (c2, "fan", "❤️ Your recap"), (c3, "social", "📱 Social pack")):
            with col.container(border=True):
                st.markdown(f"#### {title}")
                st.markdown(badge(run["checks"].get(key)) + "<br>" + cite(run["outputs"].get(key, ""), store), unsafe_allow_html=True)
                st.caption("by " + run["used"].get({"broadcast": "Broadcast Producer", "fan": "Fan Producer",
                                                     "social": "Social Producer"}[key], "?"))
        st.markdown("#### 🎬 Studio cue sheet: clips and graphics, ranked by impact")
        cues = []
        for rank, tp in enumerate(sorted(store.turning_points, key=lambda t: -t["impact"]), 1):
            ev = store.df[store.df.event_id == tp["event_id"]].iloc[0]
            start = store.df[store.df.event_id == tp["evidence"][0]].iloc[0]
            graphic = ("Win-probability swing" if tp["kind"] in ("Goal", "Penalty", "Red card")
                       else "Counterfactual: 'if it had gone in'")
            cues.append({"#": rank, "Clock": tp["clock"], "Clip in": f"{int(start.t // 60):02d}:{int(start.t % 60):02d}",
                         "Clip out": f"{int((ev.t + 6) // 60):02d}:{int((ev.t + 6) % 60):02d}", "Moment": f"{tp['kind']}: {tp['player']}",
                         "Team": tp["team_name"], "WP swing": f"{tp['dwp']:+d} pts" if tp["counterfactual_dwp"] is None
                         else f"(+{tp['counterfactual_dwp']} if scored)",
                         "Graphic": graphic + (" + xT build-up map" if tp["buildup_xt"] > 0.05 else ""),
                         "Events": f"#{tp['evidence'][0]}–#{tp['event_id']}"})
        st.dataframe(pd.DataFrame(cues), hide_index=True, use_container_width=True)
        with st.container(border=True):
            st.markdown("#### 📝 " + ("In-play report" if store.live else "Match report") + " (Commentator, approved by Fact-Checker)")
            st.markdown(badge(run["checks"].get("recap")) + "<br>" + cite(run["recap"], store), unsafe_allow_html=True)

# ------------------------------------------------------------------ trace
with tab_trace:
    if not run:
        st.info("Run the AI crew to see the agents collaborate.")
    else:
        st.markdown("### How the crew worked this match")
        n_tool = sum(t["step"] == "tool" for t in run["trace"])
        n_fail = sum(t["status"] in ("error", "fallback") for t in run["trace"])
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Agent events", len(run["trace"]))
        k2.metric("Tool calls", n_tool)
        k3.metric("Fact-check revisions", run["revisions"])
        k4.metric("Failures recovered", n_fail)
        st.plotly_chart(viz.gantt(run["trace"]), use_container_width=True)
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("#### Workflow graph (exported from Agent Framework)")
            st.graphviz_chart(cached_dot(), use_container_width=True)
        with c2:
            st.markdown("#### Fact-Checker verdicts")
            for v in run["verdicts"]:
                ok = "✓ approved" if v["ok"] else "✕ sent back to Commentator"
                st.markdown(f"**Draft {v['revision']}** · {ok}")
                for i in v["issues"][:6]:
                    st.caption(f"{'🔴' if i['severity'] == 'error' else '🟠'} {i['problem']}")
            st.markdown("#### Who wrote what")
            st.dataframe(pd.DataFrame([{"Agent": k, "Provider": v} for k, v in run["used"].items()]),
                         hide_index=True, use_container_width=True)
        with st.expander("Full trace"):
            st.dataframe(pd.DataFrame(run["trace"]), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------ ask
with tab_ask:
    st.markdown("### Ask the match")
    st.caption("A grounded Q&A agent with the same tools as the crew. Every answer is fact-checked.")
    hist_key = f"chat-{meta['match_id']}"
    st.session_state.setdefault(hist_key, [])
    for q, ans in st.session_state[hist_key]:
        st.chat_message("user").write(q)
        st.chat_message("assistant").markdown(badge(ans["check"]) + "<br>" + cite(ans["answer"], store) +
                                              f'<br><span class="muted">{ans["provider"]}</span>', unsafe_allow_html=True)
    examples = ["Why did the result go the way it did?", "Who was the best player and why?",
                f"How did {meta['away_name']} try to press?", "What was the biggest missed chance?"]
    q = st.chat_input("e.g. " + examples[0])
    ex_cols = st.columns(len(examples))
    for col, ex in zip(ex_cols, examples):
        if col.button(ex, use_container_width=True):
            q = ex
    if q:
        with st.spinner("Thinking with match tools…"):
            ans = asyncio.run(ask(store, q, outage=outage, providers=providers))
        st.session_state[hist_key].append((q, ans))
        st.rerun()

# ------------------------------------------------------------------ data
with tab_data:
    st.markdown("### The synthetic data")
    st.markdown(
        "Every event is generated by Touchline's seeded **possession-chain engine**. No real match, club or player is "
        "used. For each action it samples shoot / move / lose-the-ball from league-wide *zone priors* (aggregate rates "
        "on an 18×12 grid estimated from public StatsBomb Open Data), then applies club style: quality, press, "
        "directness, tempo and aggression. Passes, carries, shots with xG, tackles, fouls, cards, corners, throw-ins and "
        "substitutions are all emitted with x/y coordinates and timestamps.")

    @st.cache_data(show_spinner="Simulating 60 matches…")
    def calibration() -> pd.DataFrame:
        import itertools
        pairs = list(itertools.permutations(CLUBS, 2))
        rows = []
        for i in range(60):
            hh, aa = pairs[i % len(pairs)]
            d, m = simulate_match(hh, aa, 900 + i)
            sh, ps = d[d.type == "Shot"], d[d.type == "Pass"]
            rows.append({"Goals": sum(m["score"]), "Shots per team": len(sh) / 2, "xG": sh.xg.sum(),
                         "Pass completion %": 100 * (ps.outcome == "Complete").mean(),
                         "Fouls": (d.type == "Foul Committed").sum(), "Corners": (d.set_piece == "Corner").sum(),
                         "Events": len(d)})
        return pd.DataFrame(rows)

    cal = calibration()
    ref = {"Goals": "2.5–3.3", "Shots per team": "11–14", "xG": "2.5–3.2", "Pass completion %": "78–85",
           "Fouls": "18–24", "Corners": "9–11", "Events": "n/a (detail level)"}
    st.dataframe(pd.DataFrame({"Per match": cal.columns, "Touchline engine (60 sims)": cal.mean().round(2).values,
                               "Typical top flight": [ref[c] for c in cal.columns]}), hide_index=True, use_container_width=True)
    st.markdown(f"#### This match's event stream ({len(store.df):,} events)")
    cols = ["event_id", "clock", "team", "player", "position", "type", "outcome", "x", "y", "end_x", "end_y", "xg",
            "xt_added", "possession_id", "play_pattern", "score_home", "score_away"]
    st.dataframe(store.df[cols].round(3), hide_index=True, use_container_width=True, height=320)
    st.download_button("Download events (CSV)", store.df[cols].to_csv(index=False), f"{meta['match_id']}_events.csv")
