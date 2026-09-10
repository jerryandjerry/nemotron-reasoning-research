"""Final post-competition scrape of Nemotron discussion board via Playwright.
Reuses the proven harvest+scrape logic from scrape_discussions.py, but:
 - force-refreshes (overwrites) so winner write-ups posted at close are current
 - saves to final_scrape/ (keeps old discussion_scrape/ as reference)
 - prints a digest highlighting solution/place/winning threads
"""
import asyncio, json, re, sys
from pathlib import Path
from playwright.async_api import async_playwright

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
OUT_DIR = Path(__file__).parent / "final_scrape"
OUT_DIR.mkdir(exist_ok=True)
BASE = "https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion"

async def harvest_list(page, url):
    ids = set()
    await page.goto(url, wait_until="domcontentloaded")
    await page.wait_for_timeout(3500)
    for _ in range(25):
        hrefs = await page.eval_on_selector_all(
            "a[href*='/discussion/']", "els => els.map(e => e.getAttribute('href'))")
        for h in hrefs:
            m = re.search(r"/discussion/(\d+)(?:$|[?#])", h or "")
            if m:
                ids.add(m.group(1))
        await page.mouse.wheel(0, 6000)
        await page.wait_for_timeout(500)
    return ids

async def gather_ids(page):
    all_ids = set()
    plan = [("hotness", 6), ("published", 4), ("active", 3)]
    for sort, npages in plan:
        for p in range(1, npages + 1):
            url = f"{BASE}?sort={sort}&page={p}"
            got = await harvest_list(page, url)
            new = got - all_ids
            all_ids.update(got)
            print(f"list {sort} p{p}: got {len(got)}, new {len(new)}, total {len(all_ids)}", flush=True)
            if not new and p > 1:
                break
    return all_ids

async def scrape_thread(page, tid):
    url = f"{BASE}/{tid}"
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        await page.wait_for_timeout(2200)
        for _ in range(8):
            await page.mouse.wheel(0, 6000)
            await page.wait_for_timeout(300)
        title = await page.title()
        body = await page.eval_on_selector("body", "el => el.innerText")
        return {"id": tid, "url": url, "title": title, "body": body}
    except Exception as e:
        return {"id": tid, "url": url, "error": str(e)}

async def main():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
            viewport={"width": 1400, "height": 900})
        page = await ctx.new_page()
        ids = await gather_ids(page)
        print(f"TOTAL unique threads: {len(ids)}", flush=True)
        (OUT_DIR / "_index.json").write_text(json.dumps(sorted(ids), indent=2), encoding="utf-8")
        digest = []
        for i, tid in enumerate(sorted(ids, key=int, reverse=True)):  # newest ids first
            data = await scrape_thread(page, tid)
            (OUT_DIR / f"thread_{tid}.json").write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            t = data.get("title", "") or data.get("error", "")
            digest.append((tid, t))
            if (i + 1) % 10 == 0:
                print(f"scraped {i+1}/{len(ids)}", flush=True)
        await browser.close()
    # print digest: solution/place/winning first
    kw = re.compile(r"\b(\d+(st|nd|rd|th)\s*place|solution|winning|winner|1st|first place|gold|write.?up|approach)\b", re.I)
    print("\n===== SOLUTION / PLACE / WINNER THREADS =====", flush=True)
    for tid, t in digest:
        if kw.search(t or ""):
            print(f"[{tid}] {t}", flush=True)
    print("\n===== ALL THREAD TITLES (newest first) =====", flush=True)
    for tid, t in digest:
        print(f"[{tid}] {t}", flush=True)
    print(f"\nDONE. {len(digest)} threads -> {OUT_DIR}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
