#!/usr/bin/env python3
"""Render poster/poster.html to poster/poster.pdf (48 x 36 in, landscape).

usage: python3 poster/make_figures.py && python3 poster/build.py
Fails if any column's content overflows the page.
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 48 * 96, "height": 36 * 96})
    page.goto((HERE / "poster.html").as_uri())
    page.wait_for_load_state("networkidle")
    overflow = page.evaluate("""() => [...document.querySelectorAll('main > section')]
        .map((s, i) => ({col: i + 1, over: s.scrollHeight - s.clientHeight}))
        .filter(x => x.over > 1)""")
    page.pdf(path=str(HERE / "poster.pdf"), width="48in", height="36in",
             print_background=True, prefer_css_page_size=True)
    page.screenshot(path=str(HERE / "preview" / "poster.png"), full_page=False, scale="css")
    browser.close()

if overflow:
    print("OVERFLOW (in px at 96/in):", overflow)
    sys.exit(1)
print("wrote poster/poster.pdf")
