"""
One-time interactive login helper for scrape_civitai_prompts.py.

Opens a real, visible browser window pointed at the site's login page. Log in
there yourself - this script never sees your username/password. Once you're
logged in (and, if you want NSFW-tagged content, have enabled NSFW browsing
in your account's Content settings), come back to this terminal and press
Enter; your session is saved to a local file that scrape_civitai_prompts.py
reuses on future runs, so you only need to do this again once the session
expires.

Usage:
    python civitai_login.py
    python civitai_login.py --base-url https://civitai.red --out civitai_auth_state.json

Requires:
    pip install playwright
    playwright install chromium
"""

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="https://civitai.red", help="Site base URL (default: https://civitai.red)")
    parser.add_argument("--out", default=str(Path(__file__).parent / "civitai_auth_state.json"), help="Where to save the session file")
    args = parser.parse_args()

    out_path = Path(args.out)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto(f"{args.base_url}/login")

        print("A browser window has opened.")
        print("1. Log in to your account there.")
        print("2. If you want to scrape mature/NSFW-tagged content, also go to your")
        print("   account's Content settings and enable NSFW browsing.")
        input("Once done, come back here and press Enter to save the session... ")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        page.context.storage_state(path=str(out_path))
        browser.close()

    print(f"Saved session to {out_path}")


if __name__ == "__main__":
    main()
