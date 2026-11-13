"""Smoke-test every page: load each route in headless Chromium, report console errors / render errors.

Usage:  ~/my_env/bin/python scripts/check_pages.py [--tab concepts] [--only id1,id2] [--shots DIR]
Needs the local server running (tmux ml-dash:serve -> http://127.0.0.1:8765).
Exit code 1 if any page has errors or is still a placeholder.
"""
import argparse, glob, os, re, sys, pathlib
from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8765/"
ROOT = pathlib.Path(__file__).resolve().parents[1]


def routes():
    """Parse js/registry.js for (tab, dir, id) without a JS runtime."""
    src = (ROOT / "js/registry.js").read_text()
    out = []
    for block in re.split(r"\n  \{\n    id: ", src)[1:]:
        tab = re.match(r"'([\w-]+)'", block).group(1)
        d = re.search(r"dir: '([\w-]+)'", block).group(1)
        for tid in re.findall(r"\[\s*'([\w-]+)',\s*'", block):
            out.append((tab, d, tid))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tab"); ap.add_argument("--only"); ap.add_argument("--shots")
    ap.add_argument("--dark", action="store_true")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    bad = 0
    with sync_playwright() as p:
        exe = sorted(glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium_headless_shell-*/*/chrome-headless-shell")))
        b = p.chromium.launch(executable_path=exe[-1] if exe else None)
        page = b.new_page(viewport={"width": 1366, "height": 900}, color_scheme="dark" if a.dark else "light")
        errs = []
        page.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errs.append(str(e)))
        for tab, d, tid in routes():
            if a.tab and tab != a.tab: continue
            if only and tid not in only: continue
            exists = (ROOT / "js" / d / f"{tid}.js").exists()
            errs.clear()
            page.goto(f"{BASE}#/{tab}/{tid}")
            page.wait_for_timeout(1500)
            body = page.inner_text("main")
            placeholder = "still being written" in body
            render_err = "hit an error while rendering" in body
            ok = exists and not placeholder and not render_err and not errs
            if a.shots and exists:
                pathlib.Path(a.shots).mkdir(parents=True, exist_ok=True)
                page.screenshot(path=f"{a.shots}/{tab}__{tid}.png", full_page=True)
            status = "OK " if ok else ("MISSING" if not exists else "FAIL")
            if not ok: bad += 1
            if not exists and not (only or a.tab): continue
            print(f"{status:7} {tab}/{tid}" + ("" if ok else f"  {errs[:3]} {'render-error: ' + body[body.find('error while rendering'):][:200] if render_err else ''}"))
        b.close()
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
