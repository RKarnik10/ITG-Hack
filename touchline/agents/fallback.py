"""Deterministic template writers: the last link of the provider chain.

They produce every artefact the crew produces, built straight from the fact
store, so they are grounded by construction and the product never goes blank.
"""
from __future__ import annotations

from ..analytics.store import MatchStore


def _p(name: str) -> str:
    return name + ("'" if name.endswith("s") else "'s")


def _lead(store: MatchStore):
    h, a = store.meta["score"]
    if h == a:
        return None, None
    return (store.home, store.away) if h > a else (store.away, store.home)


def analyst(store: MatchStore) -> str:
    n, c = store.names, store.c
    fh, fa = store.f_team[store.home], store.f_team[store.away]
    xg_h, xg_a = store.facts[fh["xg"]].value, store.facts[fa["xg"]].value
    better = store.home if xg_h >= xg_a else store.away
    out = [f"- **{n[better]} made the better chances**: xG {c(store.f_team[better]['xg'])} vs "
           f"{c(store.f_team[store.away if better == store.home else store.home]['xg'])}."]
    pos = store.home if store.facts[fh["possession_pct"]].value >= 50 else store.away
    out.append(f"- **{n[pos]} had the ball**: {c(store.f_team[pos]['possession_pct'])} possession at "
               f"{c(store.f_team[pos]['pass_completion_pct'])} completion.")
    if store.turning_points:
        tp = max(store.turning_points, key=lambda t: abs(t["dwp"]))
        out.append(f"- **The decisive moment came at {tp['clock']}**: {tp['player']}'s {tp['kind'].lower()} swung "
                   f"{_p(tp['team_name'])} win probability by {c(tp['facts']['dwp'])}.")
        misses = [t for t in store.turning_points if "cf" in t["facts"]]
        if misses:
            m = max(misses, key=lambda t: t["counterfactual_dwp"])
            out.append(f"- **The one that got away**: {m['player']}'s chance at {m['clock']} (xG {c(m['facts']['xg'])}) "
                       f"would have been worth {c(m['facts']['cf'])} of win probability.")
    if store.f_player:
        name, f = max(store.f_player.items(), key=lambda kv: store.facts[kv[1]["rating"]].value)
        out.append(f"- **Player of the match: {name}**: rated {c(f['rating'])} with xT {c(f['xt'])}.")
    return "\n".join(out)


def tactician(store: MatchStore) -> str:
    n, c = store.names, store.c
    parts = []
    for team in (store.home, store.away):
        f = store.f_team[team]
        lanes = {k.split("_")[1]: v for k, v in f.items() if k.startswith("lane_")}
        main = max(lanes, key=lambda k: store.facts[lanes[k]].value) if lanes else "centre"
        parts.append(f"### {n[team]}\nPossession {c(f['possession_pct'])} with average pass length "
                     f"{c(f['avg_pass_length'])} yards. Pressed at PPDA {c(f['ppda'])} and won the ball back "
                     f"within 40m of the opponent's goal {c(f['high_regains'])} times. Most final-third entries came down the "
                     f"{main} ({c(lanes[main]) if lanes else 'n/a'}).")
    if store.spells:
        s = max(store.spells, key=lambda s: s["threat_for"])
        why = (f"### Why momentum shifted\n{_p(n[s['team']])} strongest spell ran from minute {s['start']} to {s['end']} "
               f"[{s['facts']['window']}], generating threat {c(s['facts']['for'])} against {c(s['facts']['against'])}.")
    else:
        why = "### Why momentum shifted\nNeither side sustained a long spell of dominance; the game turned on moments."
    parts.append(why)
    return "\n\n".join(parts)


def commentary(store: MatchStore) -> str:
    c = store.c
    lines = []
    for tp in store.turning_points:
        f = tp["facts"]
        if tp["kind"] in ("Goal", "Penalty"):
            txt = (f"GOAL! {tp['player']} scores for {tp['team_name']} - {tp['score_after']}. Win probability "
                   f"{c(f['wp_before'])} to {c(f['wp_after'])}.")
        elif tp["kind"] == "Red card":
            txt = f"Red card! {tp['player']} is off and {_p(tp['team_name'])} win probability drops to {c(f['wp_after'])}."
        else:
            txt = (f"Big chance for {tp['player']} (xG {c(f['xg'])}) but it stays out - a goal would have added "
                   f"{c(f['cf'])} to {_p(tp['team_name'])} win probability.")
        lines.append(f"{tp['clock']} | {txt}")
    return "LIVE:\n" + "\n".join(lines) + "\nRECAP:\n" + recap(store)


def recap(store: MatchStore) -> str:
    n, c = store.names, store.c
    win, lose = _lead(store)
    h, a = store.meta["score"]
    if store.live:
        head = (f"At {store.live}, {n[win]} lead {n[lose]} {max(h, a)}-{min(h, a)} [{store.f_score}]." if win
                else f"At {store.live}, {n[store.home]} and {n[store.away]} are level at {h}-{a} [{store.f_score}].")
    else:
        head = (f"{n[win]} beat {n[lose]} {max(h, a)}-{min(h, a)} [{store.f_score}]." if win
                else f"{n[store.home]} and {n[store.away]} drew {h}-{a} [{store.f_score}].")
    fh, fa = store.f_team[store.home], store.f_team[store.away]
    body = (f" {n[store.home]} produced xG {c(fh['xg'])} from {c(fh['shots'])} shots; {n[store.away]} {c(fa['xg'])} "
            f"from {c(fa['shots'])}.")
    if store.turning_points:
        tp = max(store.turning_points, key=lambda t: abs(t["dwp"]))
        body += (f" The biggest swing came at {tp['clock']} when {tp['player']}'s {tp['kind'].lower()} moved "
                 f"{_p(tp['team_name'])} win probability from {c(tp['facts']['wp_before'])} to {c(tp['facts']['wp_after'])}.")
    if store.live:
        f = store.f_live
        body += (f" Live win probability: {n[store.home]} {c(f['wp_home'])}, {n[store.away]} {c(f['wp_away'])}. "
                 f"Watch for the next goal: {c(f['more'])} chance of another, {n[store.home]} {c(f['next_home'])} "
                 f"to score it.")
    return head + body


def broadcast(store: MatchStore) -> str:
    return "**Headline**\n" + recap(store).split(".")[0] + ".\n\n" + analyst(store) + \
        "\n\n**On-screen graphics**\n- Win-probability timeline\n- Shot map sized by xG\n- Momentum (xT) chart"


# Fan-recap phrasebook, so the persona's language survives a fall-back to templates.
FAN_TEXT = {
    "English": {
        "win": "{w} beat {l} {s}.", "draw": "{h} and {a} drew {s}.",
        "live_lead": "At {t}, {w} lead {l} {s}.", "live_draw": "At {t}, {h} and {a} are level at {s}.",
        "xg": "{t1} created xG {x1} from {n1} shots, {t2} xG {x2} from {n2}.",
        "swing": "The biggest moment came at {clock}: {player} moved {team} win probability from {b} to {a}.",
        "live": "Live win probability for {team}: {p}.",
        "happy": "A great day to be a {team} fan!", "sad": "A tough one for {team} fans.",
        "level": "Honours even for {team}.", "hope": "Still everything to play for, {team} fans!",
        "gloss": "(xG = chance quality; win probability = live odds of winning.)", "ht": "half-time",
    },
    "Spanish": {
        "win": "{w} venció a {l} {s}.", "draw": "{h} y {a} empataron {s}.",
        "live_lead": "En el {t}, {w} gana a {l} {s}.", "live_draw": "En el {t}, {h} y {a} empatan {s}.",
        "xg": "{t1} generó un xG de {x1} en {n1} tiros, {t2} un xG de {x2} en {n2}.",
        "swing": "El momento clave llegó en el {clock}: {player} cambió la probabilidad de victoria de {team} de {b} a {a}.",
        "live": "Probabilidad de victoria en vivo de {team}: {p}.",
        "happy": "¡Un gran día para ser aficionado de {team}!", "sad": "Un día duro para la afición de {team}.",
        "level": "Reparto de puntos para {team}.", "hope": "¡Todo sigue abierto, afición de {team}!",
        "gloss": "(xG = calidad de las ocasiones; probabilidad de victoria = opciones de ganar en directo.)", "ht": "descanso",
    },
    "French": {
        "win": "{w} bat {l} {s}.", "draw": "{h} et {a} font match nul {s}.",
        "live_lead": "Score à {t} : {w} mène face à {l} {s}.", "live_draw": "Score à {t} : {h} et {a} sont à égalité {s}.",
        "xg": "{t1} a généré un xG de {x1} sur {n1} tirs, {t2} un xG de {x2} sur {n2}.",
        "swing": "Le moment clé est survenu à {clock} : {player} a fait passer la probabilité de victoire de {team} de {b} à {a}.",
        "live": "Probabilité de victoire en direct pour {team} : {p}.",
        "happy": "Une belle journée pour les supporters de {team} !", "sad": "Une journée difficile pour les supporters de {team}.",
        "level": "Un point chacun pour {team}.", "hope": "Rien n'est encore joué, supporters de {team} !",
        "gloss": "(xG = qualité des occasions ; probabilité de victoire = chances de gagner en direct.)", "ht": "la mi-temps",
    },
    "Portuguese": {
        "win": "{w} venceu {l} por {s}.", "draw": "{h} e {a} empataram {s}.",
        "live_lead": "Ao {t}, {w} vence {l} por {s}.", "live_draw": "Ao {t}, {h} e {a} estão empatados {s}.",
        "xg": "{t1} criou xG de {x1} em {n1} remates, {t2} xG de {x2} em {n2}.",
        "swing": "O momento decisivo foi aos {clock}: {player} mudou a probabilidade de vitória do {team} de {b} para {a}.",
        "live": "Probabilidade de vitória ao vivo do {team}: {p}.",
        "happy": "Um grande dia para os adeptos do {team}!", "sad": "Um dia difícil para os adeptos do {team}.",
        "level": "Um ponto para cada lado, {team}.", "hope": "Ainda está tudo em aberto, adeptos do {team}!",
        "gloss": "(xG = qualidade das oportunidades; probabilidade de vitória = hipóteses de vencer ao vivo.)", "ht": "intervalo",
    },
    "German": {
        "win": "{w} schlägt {l} {s}.", "draw": "{h} und {a} trennen sich {s}.",
        "live_lead": "Stand {t}: {w} führt gegen {l} {s}.", "live_draw": "Stand {t}: {h} und {a} stehen {s}.",
        "xg": "{t1} erzielte einen xG-Wert von {x1} aus {n1} Schüssen, {t2} {x2} aus {n2}.",
        "swing": "Der Schlüsselmoment ({clock}): {player} veränderte die Siegwahrscheinlichkeit von {team} von {b} auf {a}.",
        "live": "Live-Siegwahrscheinlichkeit für {team}: {p}.",
        "happy": "Ein großartiger Tag für die Fans von {team}!", "sad": "Ein schwerer Tag für die Fans von {team}.",
        "level": "Punkteteilung für {team}.", "hope": "Noch ist alles offen, {team}-Fans!",
        "gloss": "(xG = Qualität der Chancen; Siegwahrscheinlichkeit = aktuelle Gewinnchance.)", "ht": "Halbzeit",
    },
    "Hindi": {
        "win": "{w} ने {l} को {s} से हराया।", "draw": "{h} और {a} का मैच {s} से बराबर रहा।",
        "live_lead": "{t} पर {w}, {l} से {s} से आगे है।", "live_draw": "{t} पर {h} और {a} {s} से बराबर हैं।",
        "xg": "{t1} ने {n1} शॉट से {x1} xG बनाया, {t2} ने {n2} शॉट से {x2} xG।",
        "swing": "सबसे बड़ा पल {clock} पर आया: {player} ने {team} की जीत की संभावना {b} से {a} कर दी।",
        "live": "{team} की लाइव जीत की संभावना: {p}।",
        "happy": "{team} के प्रशंसकों के लिए शानदार दिन!", "sad": "{team} के प्रशंसकों के लिए मुश्किल दिन।",
        "level": "{team} के लिए एक अंक।", "hope": "{team} के प्रशंसकों, मुकाबला अभी बाकी है!",
        "gloss": "(xG = मौके की गुणवत्ता; जीत की संभावना = जीतने के लाइव आसार।)", "ht": "हाफ-टाइम",
    },
    "Japanese": {
        "win": "{w}が{l}に{s}で勝利。", "draw": "{h}と{a}は{s}で引き分け。",
        "live_lead": "{t}時点で{w}が{l}に{s}でリード。", "live_draw": "{t}時点で{h}と{a}は{s}の同点。",
        "xg": "{t1}は{n1}本のシュートでxG {x1}、{t2}は{n2}本でxG {x2}。",
        "swing": "最大の転機は{clock}：{player}が{team}の勝率を{b}から{a}に変えた。",
        "live": "{team}のライブ勝率：{p}。",
        "happy": "{team}ファンにとって最高の一日！", "sad": "{team}ファンには厳しい一日。",
        "level": "{team}は勝ち点1を分け合った。", "hope": "{team}ファンの皆さん、勝負はまだこれから！",
        "gloss": "（xG = チャンスの質、勝率 = ライブでの勝つ確率）", "ht": "ハーフタイム",
    },
}


def fan(store: MatchStore, persona: dict) -> str:
    """Persona-aware fan recap: the fan's club first, in their language and level."""
    n, c = store.names, store.c
    T = FAN_TEXT.get(persona.get("language") or "English", FAN_TEXT["English"])
    sp = " " if T is not FAN_TEXT["Japanese"] else ""
    team = persona.get("team") if persona.get("team") in n else None
    other = None if team is None else (store.away if team == store.home else store.home)
    win, lose = _lead(store)
    h, a = store.meta["score"]
    score = f"{max(h, a)}-{min(h, a)} [{store.f_score}]" if win else f"{h}-{a} [{store.f_score}]"
    key = ("live_lead" if win else "live_draw") if store.live else ("win" if win else "draw")
    head = T[key].format(w=n.get(win), l=n.get(lose), h=n[store.home], a=n[store.away], s=score, t=T["ht"] if store.live == "Half-time" else store.live)
    first = team or store.home
    second = other or store.away
    f1, f2 = store.f_team[first], store.f_team[second]
    lines = [head, T["xg"].format(t1=n[first], x1=c(f1["xg"]), n1=c(f1["shots"]),
                                  t2=n[second], x2=c(f2["xg"]), n2=c(f2["shots"]))]
    if store.turning_points:
        tp = max(store.turning_points, key=lambda t: abs(t["dwp"]))
        lines.append(T["swing"].format(clock=tp["clock"], player=tp["player"], team=tp["team_name"],
                                       b=c(tp["facts"]["wp_before"]), a=c(tp["facts"]["wp_after"])))
    if store.live and team:
        lines.append(T["live"].format(team=n[team], p=c(store.f_live["wp_home" if team == store.home else "wp_away"])))
    para1 = sp.join(lines)
    if persona.get("expertise", "beginner") == "beginner":
        para1 += sp + T["gloss"]
    if not team:
        return para1
    mood = "hope" if store.live else "level" if not win else "happy" if win == team else "sad"
    return para1 + "\n\n" + T[mood].format(team=n[team])


def social(store: MatchStore) -> str:
    n, c = store.names, store.c
    h, a = store.meta["score"]
    posts = [f"1) FT: {n[store.home]} {h}-{a} {n[store.away]} [{store.f_score}] ⚽ #Touchline"]
    fh = store.f_team[store.home]
    posts.append(f"2) {n[store.home]} created xG {c(fh['xg'])} 📊 #Touchline")
    if store.turning_points:
        tp = max(store.turning_points, key=lambda t: abs(t["dwp"]))
        posts.append(f"3) {tp['clock']}: {tp['player']} swings it - win probability {c(tp['facts']['dwp'])} 🔄 #Touchline")
    return "\n".join(posts)


def answer(store: MatchStore, question: str) -> str:
    return ("Offline mode: here is the verified match summary.\n" + store.summary_text())
