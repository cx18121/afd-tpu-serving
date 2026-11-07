#!/usr/bin/env python3
"""Render poster/poster.html to poster/poster.pdf (48 x 36 in, landscape).

usage: python3 poster/build.py
Measures each figure slot (<img data-fig=...>) in the laid-out page, redraws
the figures at exactly those sizes (make_figures.py), then prints the PDF.
Fails if any column's content overflows the page.
"""
import json
import subprocess
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 48 * 96, "height": 36 * 96})
    page.goto((HERE / "poster.html").as_uri())
    page.wait_for_load_state("networkidle")
    sizes = page.evaluate("""() => Object.fromEntries([...document.querySelectorAll('img[data-fig]')]
        .map(i => { const r = i.parentElement.getBoundingClientRect();
                    return [i.dataset.fig, [r.width / 96, r.height / 96]]; }))""")
    (HERE / "fig_sizes.json").write_text(json.dumps(sizes, indent=1))
    subprocess.run([sys.executable, str(HERE / "make_figures.py")], check=True)
    page.reload()
    page.wait_for_load_state("networkidle")
    overflow = page.evaluate("""() => [...document.querySelectorAll('main > *')]
        .map((s, i) => ({col: i + 1, over: s.scrollHeight - s.clientHeight}))
        .filter(x => x.over > 1)""")
    page.pdf(path=str(HERE / "poster.pdf"), width="48in", height="36in",
             print_background=True, prefer_css_page_size=True)
    browser.close()

if overflow:
    print("OVERFLOW (in px at 96/in):", overflow)
    sys.exit(1)
print("wrote poster/poster.pdf")
