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
        main = page.locator("section.main, [data-testid='stMain']").first
        await main.hover()

        await caption(page, "⚽ <b>Touchline</b>: an AI match-intelligence desk.<br>"
                            "Synthetic match → explainable analytics → a fact-checked multi-agent crew", 5)
        await page.get_by_role("button", name="▶ Replay").click()
        await caption(page, "📺 Live replay: win probability, momentum and agent-written commentary build minute by minute", 11)
        await tab(page, "🧠 Intelligence")
        await caption(page, "🧠 Every turning point explained: the win-probability swing, the counterfactual "
                            "(“if it had gone in”), and the move behind it", 4)
        await scroll(page, 900, steps=6, pause=0.7)
        await caption(page, "Analyst and Tactician reports: every number carries a fact id, verified against the data", 4)
        await scroll(page, 1100, steps=4, pause=0.6)
        await page.mouse.wheel(0, -6000)
        await tab(page, "🎙️ Studio")
        await caption(page, "🎙️ One verified package becomes a broadcaster brief, a fan recap and social posts.<br>"
                            "Hover a chip to see the fact behind any number", 3)
        chip = page.locator(".cite").first
        if await chip.count():
            await chip.hover()
        await page.wait_for_timeout(3500)
        await tab(page, "🤖 Agent trace")
        await caption(page, "🤖 8 agents on <b>Microsoft Agent Framework</b>: parallel analysts, a Commentator, "
                            "a Fact-Checker revise loop, and 3 audience producers", 4)
        await scroll(page, 700, steps=4, pause=0.8)
        await page.mouse.wheel(0, -4000)

        await toggle(page, "Inject a wrong number")
        await tab(page, "🤖 Agent trace")
        await scroll(page, 700, steps=2, pause=0.4)
        await caption(page, "🛡️ Trust: we corrupt one number in the draft, the Fact-Checker catches it "
                            "and sends it back for revision", 7)
        await page.mouse.wheel(0, -4000)
        await toggle(page, "Inject a wrong number")

        await toggle(page, "outage")
        await page.get_by_role("button", name="▶ Run the AI crew").click()
        await caption(page, "🔁 Resilience: the model provider goes down mid-show. The crew fails over "
                            "and still publishes verified output", 9)
        await toggle(page, "outage")

        await page.get_by_text(f"{away_name} fan", exact=True).click()
        await page.wait_for_timeout(1500)
        lang = page.locator("[data-testid='stSelectbox']").filter(has_text="Language")
        await lang.click()
        await page.get_by_role("option", name="Spanish").click()
        await page.wait_for_timeout(2500)
        await tab(page, "🎙️ Studio")
        await caption(page, "❤️ Personalised: the same match for an away fan, beginner level, in Spanish", 7)
        await tab(page, "🧪 Synthetic data")
        await caption(page, "🧪 100% synthetic, about 2,200 events per match, calibrated against real-world ranges by tests", 6)
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
