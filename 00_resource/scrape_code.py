"""Scrape all Nemotron competition notebooks via Playwright.

Walks multiple sort orders and collects unique notebook refs, then visits
each notebook page to capture title, author, votes, score, last run."""
import asyncio
import json
import re
from pathlib import Path
from playwright.async_api import async_playwright

OUT_DIR = Path(__file__).parent / "code_scrape"
OUT_DIR.mkdir(exist_ok=True)
BASE = "https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/code"


async def harvest_list(page, url: str) -> set[str]:
    refs: set[str] = set()
    await page.goto(url, wait_until="domcontentloaded")
    await page.wait_for_timeout(3500)
    for _ in range(40):
        hrefs = await page.eval_on_selector_all(
            "a[href*='/code/']", "els => els.map(e => e.getAttribute('href'))"
        )
        for h in hrefs:
            m = re.match(r"^/code/([^/]+)/([^/?#]+)$", h or "")
            if m:
                refs.add(f"{m.group(1)}/{m.group(2)}")
        await page.mouse.wheel(0, 6000)
        await page.wait_for_timeout(500)
    return refs


async def gather_all_refs(page) -> set[str]:
    all_refs: set[str] = set()
    for sort in ("hotness", "voteCount", "scoreDescending", "dateRun"):
        for p in range(1, 6):
            url = f"{BASE}?sortBy={sort}&page={p}&competitionId=129716"
            print(f"list {sort} p={p}")
            got = await harvest_list(page, url)
            new = got - all_refs
            all_refs.update(got)
            print(f"  got {len(got)}, new {len(new)}, total {len(all_refs)}")
            if not new and p > 1:
                break
    return all_refs


async def scrape_notebook(page, ref: str):
    url = f"https://www.kaggle.com/code/{ref}"
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        await page.wait_for_timeout(2500)
        for _ in range(6):
            await page.mouse.wheel(0, 5000)
            await page.wait_for_timeout(300)
        title = await page.title()
        body = await page.eval_on_selector("body", "el => el.innerText")
        body = body[:15000]  # cap
        return {"ref": ref, "url": url, "title": title, "body": body}
    except Exception as e:
        return {"ref": ref, "url": url, "error": str(e)}


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
            viewport={"width": 1400, "height": 900},
        )
        page = await ctx.new_page()
        refs = await gather_all_refs(page)
        print(f"TOTAL unique notebooks: {len(refs)}")
        (OUT_DIR / "_index.json").write_text(
            json.dumps(sorted(refs), indent=2), encoding="utf-8"
        )
        for i, ref in enumerate(sorted(refs)):
            safe = ref.replace("/", "__")
            out = OUT_DIR / f"nb_{safe}.json"
            if out.exists():
                continue
            print(f"[{i+1}/{len(refs)}] {ref}")
            data = await scrape_notebook(page, ref)
            out.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        await browser.close()
    print("done")


if __name__ == "__main__":
    asyncio.run(main())
