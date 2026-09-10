"""Fetch specific known-important Nemotron notebooks by ref."""
import asyncio
import json
from pathlib import Path
from playwright.async_api import async_playwright

OUT_DIR = Path(__file__).parent / "code_scrape"
OUT_DIR.mkdir(exist_ok=True)

EXTRA = [
    "huikang/adapter-validation-notebook",
    "huikang/nvidia-nemotron-all-linear",
    "amanatar/nemotron-ultimate-sft-grpo-v3",
    "amanatar/same-as-tong-hui-kang-s-tinker-submission-noteboo",
    "amanatar/nvidia-nemotron-sfttrainer-training",
    "amanatar/notebookc27c750a42",
    "mayukh18/unsloth-sft-full-data-training",
    "newduck/nvidia-nemotron-soft-balanced-sampling-sft",
    "llkh0a/nemotron-unsloth-sft-training-3-30-2",
    "johnnyhyland/nvidia-nemotron-sft-grpo-colab-faster",
    "kalyankkr/all-6-puzzle-types-decoded-sft-training-data",
    "mohankrishnathalla/nemotron-6-puzzle-types-decoded-rule-solvers",
    "konbu17/bit-manipulation-solver-cot-generator",
    "konbu17/doc-to-lora-knowledge-injection-nemotron-3-nano",
    "atahalam/nvidia-nemotron-0-72-making-0-78-public-soon",
    "torpidoff/full-pipeline-nvidia-nemotron-3-reasoning",
    "citerne/from-local-dev-rtx-6000-kaggle-cli-guide",
    "jiazhuang/nemotron-mainstream-llm-performance-comparison",
    "hastws/sfttrainer-training",
    "waterjoe/fork-of-nvidia-nemotron-sft-grpo-trainer",
    "comistrymo/nvidia-nemotron-training-cot-labels",
]


async def scrape(page, ref):
    url = f"https://www.kaggle.com/code/{ref}"
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=45000)
        await page.wait_for_timeout(2500)
        for _ in range(6):
            await page.mouse.wheel(0, 5000)
            await page.wait_for_timeout(300)
        title = await page.title()
        body = await page.eval_on_selector("body", "el => el.innerText")
        return {"ref": ref, "url": url, "title": title, "body": body[:15000]}
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
        for i, ref in enumerate(EXTRA):
            safe = ref.replace("/", "__")
            out = OUT_DIR / f"nb_{safe}.json"
            if out.exists():
                print(f"skip {ref}")
                continue
            print(f"[{i+1}/{len(EXTRA)}] {ref}")
            data = await scrape(page, ref)
            out.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        await browser.close()
    print("done")


if __name__ == "__main__":
    asyncio.run(main())
