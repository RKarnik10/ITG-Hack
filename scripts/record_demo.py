"""Record a captioned < 2 min demo video of the running app.

    streamlit run app.py --server.port 8765 &
    python scripts/record_demo.py --url http://localhost:8765 --out demo.mp4

Needs `pip install playwright && playwright install chromium` and ffmpeg.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from playwright.async_api import Page, async_playwright

CAPTION_JS = """(text) => {
  let el = document.getElementById('demo-caption');
  if (!el) {
    el = document.createElement('div');
    el.id = 'demo-caption';
    el.style.cssText = 'position:fixed;left:50%;bottom:28px;transform:translateX(-50%);z-index:99999;' +
      'background:rgba(10,10,10,.88);color:#f0efec;font:600 22px/1.35 Helvetica,Arial,sans-serif;' +
      'padding:14px 26px;border-radius:14px;max-width:1100px;text-align:center;border:1px solid #3987e5;' +
      'box-shadow:0 8px 30px rgba(0,0,0,.5);transition:opacity .3s';
    document.body.appendChild(el);
  }
  el.style.opacity = text ? 1 : 0;
  el.innerHTML = text;
}"""


async def caption(page: Page, text: str, hold: float) -> None:
    await page.evaluate(CAPTION_JS, text)
    await page.wait_for_timeout(int(hold * 1000))


async def tab(page: Page, name: str) -> None:
    await page.get_by_role("tab", name=name).click()
    await page.wait_for_timeout(1800)


async def scroll(page: Page, px: int, steps: int = 6, pause: float = 0.5) -> None:
    for _ in range(steps):
        await page.mouse.wheel(0, px / steps)
        await page.wait_for_timeout(int(pause * 1000))


async def moment(page: Page, steps: int) -> None:
    """Move the sidebar 'Analyse the match at' slider by `steps` options."""
    handle = page.locator("[data-testid='stSidebar'] [data-testid='stSlider'] input[type='range']").first
    await handle.focus()
    for _ in range(abs(steps)):
        await page.keyboard.press("ArrowLeft" if steps < 0 else "ArrowRight")
        await page.wait_for_timeout(700)
    await page.wait_for_timeout(2500)


async def toggle(page: Page, label: str) -> None:
    await page.get_by_text(label, exact=False).first.click()
    await page.wait_for_timeout(2500)


async def record(url: str, out: Path) -> None:
    demos = json.loads((Path(__file__).resolve().parents[1] / "cache" / "demo_matches.json").read_text())
    away_name = re.split(r"\d+-\d+ ", demos[0]["label"], maxsplit=1)[1]
    tmp = Path(tempfile.mkdtemp())
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        ctx = await browser.new_context(viewport={"width": 1440, "height": 900}, record_video_dir=str(tmp),
                                        record_video_size={"width": 1440, "height": 900})
        page = await ctx.new_page()
        await page.goto(url, wait_until="networkidle")
        await page.wait_for_timeout(7000)

        await caption(page, "⚽ <b>Touchline</b>: a Synthetic Match Insights Engine for the studio.<br>"
                            "Synthetic match → explainable analytics → a fact-checked multi-agent crew", 5)
        await page.get_by_role("button", name="▶ Replay").click()
        await caption(page, "📺 Live replay: win probability, momentum and agent-written commentary build minute by minute", 8)

        await moment(page, -3)  # Full time -> Half-time
        await tab(page, "🧠 Intelligence")
        await caption(page, "⏱️ Real-time: analyse at half-time. The engine and crew only see events so far.<br>"
                            "Live win probability and next-goal odds", 6)
        await tab(page, "🎙️ Studio")
        await caption(page, "🎙️ The half-time studio brief, an in-play report, and a clip & graphics cue sheet", 6)
        await moment(page, +3)  # back to Full time

        await tab(page, "🧠 Intelligence")
        await caption(page, "🧠 Key storylines, and every turning point explained: the win-probability swing, "
                            "the counterfactual, and the move behind it", 4)
        await scroll(page, 1000, steps=5, pause=0.6)
        await page.mouse.wheel(0, -6000)
        await tab(page, "🎙️ Studio")
        await caption(page, "One verified package: broadcaster brief, fan recap, social posts. Hover a chip "
                            "to see the fact behind any number", 3)
        try:  # chips also exist in hidden tabs; hover a visible one
            await page.locator(".cite >> visible=true").nth(2).hover(timeout=3000)
        except Exception:
            pass
        await page.wait_for_timeout(3000)
        await tab(page, "🤖 Agent trace")
        await caption(page, "🤖 8 agents on <b>Microsoft Agent Framework</b>: parallel analysts, a Fact-Checker "
                            "revise loop, 3 audience producers", 3)
        await scroll(page, 700, steps=4, pause=0.7)
        await page.mouse.wheel(0, -4000)

        await toggle(page, "Inject a wrong number")
        await tab(page, "🤖 Agent trace")
        await scroll(page, 700, steps=2, pause=0.4)
        await caption(page, "🛡️ Trust: we corrupt one number in the draft. The Fact-Checker catches it "
                            "and sends it back for revision", 6)
        await page.mouse.wheel(0, -4000)
        await toggle(page, "Inject a wrong number")

        await toggle(page, "outage")
        await page.get_by_role("button", name="▶ Run the AI crew").click()
        await caption(page, "🔁 Resilience: the model provider goes down on matchday. The crew fails over "
                            "and still publishes verified output", 7)
        await toggle(page, "outage")

        await page.get_by_text(f"{away_name} fan", exact=True).click()
        await page.wait_for_timeout(1500)
        lang = page.locator("[data-testid='stSelectbox']").filter(has_text="Language")
        await lang.click()
        await page.get_by_role("option", name="Spanish").click()
        await page.wait_for_timeout(2500)
        await tab(page, "🎙️ Studio")
        await caption(page, "❤️ Personalised: the same match for an away fan, beginner level, in Spanish, "
                            "still fact-checked against the data", 6)
        await tab(page, "🧪 Synthetic data")
        await caption(page, "🧪 100% synthetic, about 2,200 events per match, calibrated against real-world ranges by tests", 5)
        await caption(page, "<b>Touchline</b> · Microsoft Agent Framework · GitHub Models · Azure AI Foundry & Container Apps ready", 4)
        video = page.video
        await ctx.close()
        await browser.close()
        src = Path(await video.path())
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        shutil.copy(src, out.with_suffix(".webm"))
        print("ffmpeg not found; wrote", out.with_suffix(".webm"))
        return
    # drop the initial page-load seconds, cap at 1:58
    subprocess.run([ffmpeg, "-y", "-ss", "6", "-i", str(src), "-t", "118", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "22", "-preset", "medium", str(out)], check=True, capture_output=True)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8765")
    ap.add_argument("--out", default="demo.mp4")
    a = ap.parse_args()
    asyncio.run(record(a.url, Path(a.out)))
