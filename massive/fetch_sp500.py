"""
fetch_sp500.py -- refresh sp500.txt from Wikipedia's constituents table.

Stdlib HTML parsing only (no lxml/bs4 needed). Run it whenever you want to
re-sync the universe:

    python fetch_sp500.py

Note: this is the CURRENT index membership, so a universe built from it carries
survivorship bias. For honest backtests you want point-in-time membership --
Massive's delisted tickers are available, but index history is not.
"""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

import requests

URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
OUT = Path(__file__).with_name("sp500.txt")


class _ConstituentsParser(HTMLParser):
    """Pull the first cell of every row in the table with id='constituents'."""

    def __init__(self) -> None:
        super().__init__()
        self.in_table = False
        self.depth = 0
        self.in_row = False
        self.cell_index = -1
        self.capture = False
        self.buf: list[str] = []
        self.symbols: list[str] = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "table":
            if attrs.get("id") == "constituents":
                self.in_table = True
                self.depth = 1
            elif self.in_table:
                self.depth += 1
        elif self.in_table and self.depth == 1:
            if tag == "tr":
                self.in_row = True
                self.cell_index = -1
            elif tag in ("td", "th") and self.in_row:
                self.cell_index += 1
                self.capture = self.cell_index == 0 and tag == "td"
                self.buf = []

    def handle_endtag(self, tag):
        if tag == "table" and self.in_table:
            self.depth -= 1
            if self.depth == 0:
                self.in_table = False
        elif self.in_table and self.depth == 1:
            if tag in ("td", "th") and self.capture:
                text = "".join(self.buf).strip()
                if text:
                    self.symbols.append(text)
                self.capture = False
            elif tag == "tr":
                self.in_row = False

    def handle_data(self, data):
        if self.capture:
            self.buf.append(data)


def fetch_symbols() -> list[str]:
    resp = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    resp.raise_for_status()

    parser = _ConstituentsParser()
    parser.feed(resp.text)

    # Massive uses the dotted form for class shares (BRK.B, BF.B), same as
    # Wikipedia -- do NOT rewrite dots to dashes, that returns zero bars.
    symbols = sorted({s.upper() for s in parser.symbols if s})
    if len(symbols) < 400:
        raise RuntimeError(f"Only parsed {len(symbols)} symbols -- page layout may have changed")
    return symbols


def main() -> None:
    symbols = fetch_symbols()
    OUT.write_text("\n".join(symbols) + "\n", encoding="utf-8")
    print(f"wrote {len(symbols)} tickers to {OUT}")
    print("first:", ", ".join(symbols[:8]))
    print("last: ", ", ".join(symbols[-8:]))


if __name__ == "__main__":
    main()
