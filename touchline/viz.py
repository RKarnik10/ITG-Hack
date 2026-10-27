"""Plotly figures for the Touchline UI (dark broadcast theme, validated palette)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

SURFACE = "#1a1a19"
PITCH = "#1f2a1f"
LINE = "rgba(240,239,236,0.35)"
TEXT = "#f0efec"
MUTED = "#c3c2b7"
GRID = "rgba(195,194,183,0.12)"
HOME = "#3987e5"     # categorical slot 1 (dark step)
AWAY = "#d95926"     # categorical slot 2 (dark step)
DRAW = "#8a8984"
STATUS = {"ok": "#0ca30c", "cached": "#6b7280", "warn": "#ec835a", "fallback": "#fab219", "error": "#d03b3b"}
STATUS_LABEL = {"ok": "✓ ok", "cached": "↺ cached", "warn": "! revised", "fallback": "⇢ fallback", "error": "✕ error"}


def _layout(fig: go.Figure, height: int, title: str | None = None, **kw) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=36 if title else 8, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, size=12), title=dict(text=title, font=dict(color=TEXT, size=14), x=0.01) if title else None,
        hoverlabel=dict(bgcolor="#2a2a28", font_color=TEXT, bordercolor="#3a3a37"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        **kw,
    )
    return fig


def pitch(height: int = 380) -> go.Figure:
    fig = go.Figure()
    s = dict(type="rect", line=dict(color=LINE, width=1.5), layer="below")
    shapes = [
        dict(type="rect", x0=0, y0=0, x1=120, y1=80, fillcolor=PITCH, line=dict(color=LINE, width=1.5), layer="below"),
        dict(type="line", x0=60, y0=0, x1=60, y1=80, line=dict(color=LINE, width=1.5), layer="below"),
        dict(type="circle", x0=50, y0=30, x1=70, y1=50, line=dict(color=LINE, width=1.5), layer="below"),
        {**s, "x0": 0, "y0": 18, "x1": 18, "y1": 62}, {**s, "x0": 102, "y0": 18, "x1": 120, "y1": 62},
        {**s, "x0": 0, "y0": 30, "x1": 6, "y1": 50}, {**s, "x0": 114, "y0": 30, "x1": 120, "y1": 50},
        {**s, "x0": -2, "y0": 36, "x1": 0, "y1": 44}, {**s, "x0": 120, "y0": 36, "x1": 122, "y1": 44},
    ]
    fig.update_layout(shapes=shapes)
    fig.update_xaxes(range=[-3, 123], visible=False, fixedrange=True)
    fig.update_yaxes(range=[83, -3], visible=False, fixedrange=True, scaleanchor="x", scaleratio=1)
    return _layout(fig, height, showlegend=True)


def shot_map(df: pd.DataFrame, meta: dict, upto_s: float | None = None, height: int = 380) -> go.Figure:
    fig = pitch(height)
    shots = df[df.type == "Shot"]
    if upto_s is not None:
        shots = shots[shots.elapsed <= upto_s]
    names = {meta["home"]: meta["home_name"], meta["away"]: meta["away_name"]}
    for team, colour in ((meta["home"], HOME), (meta["away"], AWAY)):
        sub = shots[shots.team == team]
        goal = sub.outcome == "Goal"
        for mask, sym, nm in ((~goal, "circle", f"{names[team]} shot"), (goal, "star", f"{names[team]} goal")):
            d = sub[mask]
            fig.add_trace(go.Scatter(
                x=d.dx, y=d.dy, mode="markers", name=nm,
                marker=dict(size=8 + 40 * np.sqrt(d.xg.fillna(0.05)), symbol=sym,
                            color=colour if sym == "star" else "rgba(0,0,0,0)",
                            line=dict(color=colour if sym == "circle" else SURFACE, width=2)),
                customdata=np.stack([d.clock, d.player, d.xg, d.outcome], axis=-1) if len(d) else None,
                hovertemplate="%{customdata[0]} %{customdata[1]}<br>xG %{customdata[2]:.2f} · %{customdata[3]}<extra></extra>",
            ))
    return fig


def chain_map(chain: list[dict], colour: str, height: int = 300) -> go.Figure:
    """Evidence view: the move that led to a turning point (display coordinates)."""
    fig = pitch(height)
    for i, c in enumerate(chain):
        if c["end_x"] is None or pd.isna(c["end_x"]):
            continue
        dash = "dot" if c["type"] == "Carry" else "solid"
        fig.add_trace(go.Scatter(x=[c["x"], c["end_x"]], y=[c["y"], c["end_y"]], mode="lines",
                                 line=dict(color=colour, width=2, dash=dash), showlegend=False, hoverinfo="skip"))
    xs = [c["x"] for c in chain]
    ys = [c["y"] for c in chain]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text", showlegend=False,
        marker=dict(size=9, color=colour, line=dict(color=SURFACE, width=2)),
        text=[str(i + 1) for i in range(len(chain))], textposition="top center", textfont=dict(color=TEXT, size=10),
        customdata=[[c["clock"], c["type"], c["player"] or "", c["id"]] for c in chain],
        hovertemplate="#%{customdata[3]} %{customdata[0]} %{customdata[1]}<br>%{customdata[2]}<extra></extra>",
    ))
    return fig


def win_prob(store, upto_s: float | None = None, height: int = 260) -> go.Figure:
    df, wp, meta = store.df, store.wp, store.meta
    m = (df.elapsed / 60).to_numpy()
    keep = np.ones(len(df), bool) if upto_s is None else (df.elapsed <= upto_s).to_numpy()
    fig = go.Figure()
    for col, name, colour in ((0, meta["home_name"], HOME), (1, "Draw", DRAW), (2, meta["away_name"], AWAY)):
        fig.add_trace(go.Scatter(x=m[keep], y=100 * wp[keep, col], mode="lines", name=name,
                                 line=dict(color=colour, width=2, shape="hv"),
                                 hovertemplate=f"{name}: %{{y:.0f}}%<extra></extra>"))
    for tp in store.turning_points:
        if upto_s is not None and tp["minute"] * 60 > upto_s + 60:
            continue
        if tp["kind"] in ("Goal", "Penalty", "Red card"):
            fig.add_vline(x=tp["minute"] - 0.5, line=dict(color=GRID, width=1, dash="dot"))
            fig.add_annotation(x=tp["minute"] - 0.5, y=104, text=("⚽" if tp["kind"] != "Red card" else "🟥") + " " + tp["clock"],
                               showarrow=False, font=dict(size=10, color=MUTED))
    fig.update_xaxes(range=[0, max(m.max(), 95)], title="minute", gridcolor=GRID, zeroline=False)
    fig.update_yaxes(range=[0, 108], ticksuffix="%", gridcolor=GRID, zeroline=False)
    return _layout(fig, height, "Live win probability", hovermode="x unified")


def momentum(store, upto_s: float | None = None, height: int = 250) -> go.Figure:
    mom, meta = store.momentum, store.meta
    if upto_s is not None:
        mom = mom.loc[: int(upto_s // 60)]
    x = mom.index + 1
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=mom["home"], name=meta["home_name"], marker=dict(color=HOME, line=dict(width=0)),
                         hovertemplate="%{x}' threat %{y:.2f}<extra>" + meta["home_name"] + "</extra>"))
    fig.add_trace(go.Bar(x=x, y=-mom["away"], name=meta["away_name"], marker=dict(color=AWAY, line=dict(width=0)),
                         customdata=mom["away"],
                         hovertemplate="%{x}' threat %{customdata:.2f}<extra>" + meta["away_name"] + "</extra>"))
    fig.add_trace(go.Scatter(x=x, y=mom["smooth"] * 3, mode="lines", name="Momentum (smoothed)",
                             line=dict(color=TEXT, width=2), hoverinfo="skip"))
    fig.update_xaxes(range=[0, max(len(store.momentum), 95)], gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=True, zerolinecolor=LINE, title="threat (xT + xG)")
    fig = _layout(fig, height, "Momentum: threat per minute (up = home, down = away)", barmode="relative", bargap=0.15)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0))
    return fig


def gantt(trace: list[dict], height: int = 360) -> go.Figure:
    runs = [t for t in trace if t["step"] in ("run", "verify", "fallback", "error", "retry", "repair", "handoff", "publish", "approve", "tool")]
    agents = []
    for t in runs:
        if t["agent"] not in agents:
            agents.append(t["agent"])
    fig = go.Figure()
    seen = set()
    total = max((t["end_ms"] for t in runs), default=1)
    for t in runs:
        status = t["status"] if t["status"] in STATUS else "ok"
        dur = max(t["end_ms"] - t["start_ms"], total * (0.006 if t["step"] != "tool" else 0.004))
        show = status not in seen
        seen.add(status)
        fig.add_trace(go.Bar(
            y=[t["agent"]], x=[dur], base=[t["start_ms"]], orientation="h", name=STATUS_LABEL[status],
            legendgroup=status, showlegend=show,
            marker=dict(color=STATUS[status], line=dict(color=SURFACE, width=2)),
            width=0.35 if t["step"] == "tool" else 0.7,
            customdata=[[t["step"], t["detail"][:90], t.get("provider") or "", round(dur)]],
            hovertemplate="<b>%{y}</b> · %{customdata[0]}<br>%{customdata[1]}<br>%{customdata[2]}<br>%{customdata[3]} ms<extra></extra>",
        ))
    fig.update_yaxes(categoryorder="array", categoryarray=agents[::-1], gridcolor=GRID)
    fig.update_xaxes(title="ms since kickoff of the crew run", gridcolor=GRID, zeroline=False)
    return _layout(fig, height, "Agent timeline", barmode="overlay")
