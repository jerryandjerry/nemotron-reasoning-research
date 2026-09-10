"""Scrape all Nemotron competition discussion threads via Playwright.

Strategy: walk multiple sort orders (hotness, published, active) and multiple
pages; each may expose different threads. Collect the union.
"""
import asyncio
import json
import re
from pathlib import Path
from playwright.async_api import async_playwright

OUT_DIR = Path(__file__).parent / "discussion_scrape"
OUT_DIR.mkdir(exist_ok=True)
BASE = "https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion"


async def harvest_list(page, url: str) -> set[str]:
    ids: set[str] = set()
    await page.goto(url, wait_until="domcontentloaded")
    await page.wait_for_timeout(3500)
    for _ in range(30):
        hrefs = await page.eval_on_selector_all(
            "a[href*='/discussion/']", "els => els.map(e => e.getAttribute('href'))"
        )
        for h in hrefs:
            m = re.search(r"/discussion/(\d+)(?:$|[?#])", h or "")
            if m:
                ids.add(m.group(1))
        await page.mouse.wheel(0, 6000)
        await page.wait_for_timeout(500)
    return ids


async def gather_all_ids(page) -> set[str]:
    all_ids: set[str] = set()
    for sort in ("hotness", "published", "active"):
        for p in range(1, 8):
            url = f"{BASE}?sort={sort}&page={p}"
            print(f"list {sort} page={p}")
            got = await harvest_list(page, url)
            new = got - all_ids
            all_ids.update(got)
            print(f"  got {len(got)}, new {len(new)}, total {len(all_ids)}")
            if not new and p > 1:
                break
    return all_ids


async def scrape_thread(page, tid):
    url = f"{BASE}/{tid}"
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        await page.wait_for_timeout(2500)
        for _ in range(10):
            await page.mouse.wheel(0, 6000)
            await page.wait_for_timeout(350)
        title = await page.title()
        body = await page.eval_on_selector("body", "el => el.innerText")
        return {"id": tid, "url": url, "title": title, "body": body}
    except Exception as e:
        return {"id": tid, "url": url, "error": str(e)}


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
            viewport={"width": 1400, "height": 900},
        )
        page = await ctx.new_page()
        ids = await gather_all_ids(page)
        print(f"TOTAL unique threads: {len(ids)}")
        (OUT_DIR / "_index.json").write_text(
            json.dumps(sorted(ids), indent=2), encoding="utf-8"
        )
        for i, tid in enumerate(sorted(ids)):
            out = OUT_DIR / f"thread_{tid}.json"
            if out.exists():
                continue
            print(f"[{i+1}/{len(ids)}] {tid}")
            data = await scrape_thread(page, tid)
            out.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        await browser.close()
    print("done")


if __name__ == "__main__":
    asyncio.run(main())
