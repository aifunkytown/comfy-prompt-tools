"""
Scrape image generation prompts from a Civitai-style tag feed
(e.g. https://civitai.red/images?tags=2013&view=feed) into a CSV file,
deduplicated by prompt text, stopping once a target number of unique
prompts has been collected.

Why this drives a real browser instead of calling the API directly: the
site's public API no longer returns generation metadata (prompts) at all -
this was deliberately removed to prevent bulk scraping. The prompt text is
still shown on each image's own detail page in the web UI, so this script
opens a logged-in browser session and reads it from there directly, the same
way a human browsing the site would.

Setup (one-time):
    pip install playwright
    playwright install chromium
    python civitai_login.py            # log in once, saves a session file

Usage:
    python scrape_civitai_prompts.py 2013 50
        # tags=2013 in the feed URL, stop after 50 unique prompts

    python scrape_civitai_prompts.py 2013 50 --base-url https://civitai.red \\
        --output civitai_prompts.csv --storage-state civitai_auth_state.json

The "tags" argument is substituted directly into the feed URL's tags= query
parameter, so it can be any tag id (or value) the site accepts there.
"""

import argparse
import csv
import hashlib
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

GRID_SCROLL_SELECTOR = ".scroll-area.flex-1"


def collect_image_ids(page, base_url, tags, needed, max_scrolls=300, stall_limit=6):
    """Scroll the feed, accumulating unique image ids (the grid is virtualized,
    so already-scrolled-past items get removed from the DOM - we must keep a
    running union rather than reading the DOM once at the end)."""
    feed_url = f"{base_url}/images?tags={tags}&view=feed"
    page.goto(feed_url, wait_until="domcontentloaded", timeout=30000)
    page.wait_for_timeout(2500)

    def current_ids():
        hrefs = page.eval_on_selector_all('a[href*="/images/"]', 'els => els.map(e => e.getAttribute("href"))')
        ids = []
        for h in hrefs:
            part = h.rstrip("/").rsplit("/", 1)[-1]
            if part.isdigit():
                ids.append(part)
        return ids

    seen_order = list(dict.fromkeys(current_ids()))
    seen_set = set(seen_order)
    stall = 0

    for _ in range(max_scrolls):
        if len(seen_set) >= needed:
            break
        page.evaluate(
            "(sel) => { const el = document.querySelector(sel); if (el) el.scrollTop = el.scrollHeight; }",
            GRID_SCROLL_SELECTOR,
        )
        page.wait_for_timeout(1200)
        new_ids = [i for i in current_ids() if i not in seen_set]
        if new_ids:
            stall = 0
            for i in new_ids:
                seen_order.append(i)
                seen_set.add(i)
        else:
            stall += 1
            if stall >= stall_limit:
                break  # feed appears exhausted - stopped returning new items

    return seen_order


TRAILING_ELLIPSIS_RE = re.compile(r"\.{2,}\s*$")


def _clean_prompt_text(text):
    if not text:
        return ""
    return TRAILING_ELLIPSIS_RE.sub("", text).strip()


def extract_prompt(page, image_id, base_url):
    page.goto(f"{base_url}/images/{image_id}", wait_until="domcontentloaded", timeout=30000)
    page.wait_for_timeout(1200)

    # The prompt text is never actually truncated in the DOM - "Show more"/"Show
    # less" is a CSS-only clamp toggle - but its label text and a trailing
    # ellipsis get picked up by textContent, so both must be stripped.
    result = page.evaluate(
        """
        () => {
          function byHeading(headingText) {
            const heading = Array.from(document.querySelectorAll('p'))
              .find(el => el.textContent.trim() === headingText);
            if (!heading) return null;
            const container = heading.closest('.flex.flex-col') || heading.parentElement.parentElement;
            const textEl = container ? container.querySelector('div.text-sm') : null;
            if (!textEl) return null;
            const clone = textEl.cloneNode(true);
            clone.querySelectorAll('span').forEach(s => {
              const t = s.textContent.trim();
              if (t === 'Show more' || t === 'Show less') s.remove();
            });
            return clone.textContent.trim();
          }
          return { positive: byHeading('Prompt'), negative: byHeading('Negative prompt') };
        }
        """
    )
    positive = _clean_prompt_text(result.get("positive"))
    negative = _clean_prompt_text(result.get("negative"))
    return positive, negative


def hash_prompt(positive, negative):
    combined = (positive.strip() + "\x00" + negative.strip()).encode("utf-8")
    return hashlib.sha256(combined).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tags", help="Value substituted into the feed URL's tags= parameter (e.g. 2013)")
    parser.add_argument("count", type=int, help="Stop after this many unique prompts have been collected")
    parser.add_argument("--base-url", default="https://civitai.red", help="Site base URL (default: https://civitai.red)")
    parser.add_argument("--output", default=None, help="Output CSV path (default: civitai_prompts_<tags>.csv)")
    parser.add_argument(
        "--storage-state",
        default=None,
        help="Path to the session file saved by civitai_login.py (default: civitai_auth_state.json next to this script)",
    )
    parser.add_argument("--no-headless", action="store_true", help="Show the browser window while scraping (default: headless)")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds to wait between detail-page visits (default: 0.5)")
    args = parser.parse_args()

    storage_state = Path(args.storage_state) if args.storage_state else Path(__file__).parent / "civitai_auth_state.json"
    if not storage_state.is_file():
        print(
            f"Error: no saved login session found at {storage_state}.\n"
            "Run civitai_login.py first to log in and save a session.",
            file=sys.stderr,
        )
        sys.exit(1)

    output_path = Path(args.output) if args.output else Path(f"civitai_prompts_{args.tags}.csv")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.no_headless)
        context = browser.new_context(storage_state=str(storage_state), viewport={"width": 1400, "height": 1000})
        page = context.new_page()

        print(f"Collecting image links for tags={args.tags} ...")
        candidate_ids = collect_image_ids(page, args.base_url, args.tags, needed=args.count * 3)
        print(f"Found {len(candidate_ids)} candidate image(s). Visiting detail pages...")

        seen_hashes = set()
        rows_written = 0

        with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Image ID", "Image URL", "Positive Prompt", "Negative Prompt", "Prompt Hash (SHA-256)"])

            for image_id in candidate_ids:
                if rows_written >= args.count:
                    break
                try:
                    positive, negative = extract_prompt(page, image_id, args.base_url)
                except Exception as e:
                    print(f"  {image_id}: failed to load ({e})", file=sys.stderr)
                    continue

                if not positive and not negative:
                    print(f"  {image_id}: no prompt found, skipping")
                    continue

                phash = hash_prompt(positive, negative)
                if phash in seen_hashes:
                    print(f"  {image_id}: duplicate prompt, skipping")
                    continue
                seen_hashes.add(phash)

                writer.writerow([image_id, f"{args.base_url}/images/{image_id}", positive, negative, phash])
                csv_file.flush()
                rows_written += 1
                print(f"  [{rows_written}/{args.count}] {image_id}: prompt captured")

                time.sleep(args.delay)

        browser.close()

    print(f"\nDone. Wrote {rows_written} unique prompt(s) to {output_path}")
    if rows_written < args.count:
        print(f"Note: only found {rows_written} unique prompt(s) before running out of candidate images.")


if __name__ == "__main__":
    main()
