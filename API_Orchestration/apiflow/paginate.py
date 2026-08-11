"""Pagination strategies.

Each Paginator is a small state machine:

    p = make_paginator(cfg)
    while True:
        url, params = p.next_request(base_url, base_params)   # None when exhausted
        ...perform request...
        p.observe(response_body, response_headers, record_count)

Keeping the strategies separate from the HTTP loop means a new API pagination
style is one class, not a fork of the extractor.
"""

from __future__ import annotations

from typing import Any, Protocol

from .config import (
    CursorPagination,
    LinkHeaderPagination,
    NoPagination,
    OffsetPagination,
    PagePagination,
    PaginationConfig,
)
from .paths import MISSING, dig


class Paginator(Protocol):
    def next_request(
        self, base_url: str, base_params: dict[str, Any]
    ) -> tuple[str, dict[str, Any]] | None: ...

    def observe(self, body: Any, headers: dict[str, str], record_count: int) -> None: ...


class _NonePaginator:
    def __init__(self, cfg: NoPagination) -> None:
        self.done = False

    def next_request(self, base_url, base_params):
        if self.done:
            return None
        self.done = True
        return base_url, dict(base_params)

    def observe(self, body, headers, record_count):
        return


class _PagePaginator:
    def __init__(self, cfg: PagePagination) -> None:
        self.cfg = cfg
        self.page = cfg.start_page
        self.pages_done = 0
        self.exhausted = False
        self.total_pages: int | None = None

    def next_request(self, base_url, base_params):
        if self.exhausted or self.pages_done >= self.cfg.max_pages:
            return None
        if self.total_pages is not None and self.pages_done >= self.total_pages:
            return None
        params = dict(base_params)
        params[self.cfg.page_param] = self.page
        if self.cfg.size_param:
            params[self.cfg.size_param] = self.cfg.size
        return base_url, params

    def observe(self, body, headers, record_count):
        self.pages_done += 1
        self.page += 1
        if self.cfg.total_pages_path:
            found = dig(body, self.cfg.total_pages_path)
            if found is not MISSING and found is not None:
                try:
                    self.total_pages = int(found)
                except (TypeError, ValueError):
                    pass
        # A short or empty page means we've reached the end.
        if record_count == 0 or record_count < self.cfg.size:
            self.exhausted = True


class _OffsetPaginator:
    def __init__(self, cfg: OffsetPagination) -> None:
        self.cfg = cfg
        self.offset = 0
        self.pages_done = 0
        self.exhausted = False

    def next_request(self, base_url, base_params):
        if self.exhausted or self.pages_done >= self.cfg.max_pages:
            return None
        params = dict(base_params)
        params[self.cfg.offset_param] = self.offset
        params[self.cfg.limit_param] = self.cfg.limit
        return base_url, params

    def observe(self, body, headers, record_count):
        self.pages_done += 1
        self.offset += self.cfg.limit
        if record_count == 0 or record_count < self.cfg.limit:
            self.exhausted = True


class _CursorPaginator:
    def __init__(self, cfg: CursorPagination) -> None:
        self.cfg = cfg
        self.cursor: Any = None
        self.started = False
        self.pages_done = 0
        self.exhausted = False

    def next_request(self, base_url, base_params):
        if self.exhausted or self.pages_done >= self.cfg.max_pages:
            return None
        if self.started and self.cursor in (None, "", MISSING):
            return None
        params = dict(base_params)
        if self.cfg.size_param:
            params[self.cfg.size_param] = self.cfg.size
        if self.started:
            params[self.cfg.cursor_param] = self.cursor
        self.started = True
        return base_url, params

    def observe(self, body, headers, record_count):
        self.pages_done += 1
        found = dig(body, self.cfg.cursor_path)
        self.cursor = None if found is MISSING else found
        if record_count == 0:
            self.exhausted = True


def parse_link_header(value: str, rel: str = "next") -> str | None:
    """Extract a URL from an RFC 5988 Link header for the given rel."""
    for part in value.split(","):
        section = part.split(";")
        if len(section) < 2:
            continue
        url = section[0].strip()
        if not (url.startswith("<") and url.endswith(">")):
            continue
        for param in section[1:]:
            if "=" not in param:
                continue
            key, _, val = param.partition("=")
            if key.strip().lower() == "rel" and val.strip().strip('"') == rel:
                return url[1:-1]
    return None


class _LinkHeaderPaginator:
    def __init__(self, cfg: LinkHeaderPagination) -> None:
        self.cfg = cfg
        self.next_url: str | None = None
        self.started = False
        self.pages_done = 0

    def next_request(self, base_url, base_params):
        if self.pages_done >= self.cfg.max_pages:
            return None
        if not self.started:
            self.started = True
            return base_url, dict(base_params)
        if not self.next_url:
            return None
        # The next URL already carries its own query string.
        return self.next_url, {}

    def observe(self, body, headers, record_count):
        self.pages_done += 1
        link = headers.get("link") or headers.get("Link") or ""
        self.next_url = parse_link_header(link, self.cfg.rel) if link else None


def make_paginator(cfg: PaginationConfig) -> Paginator:
    if isinstance(cfg, NoPagination):
        return _NonePaginator(cfg)
    if isinstance(cfg, PagePagination):
        return _PagePaginator(cfg)
    if isinstance(cfg, OffsetPagination):
        return _OffsetPaginator(cfg)
    if isinstance(cfg, CursorPagination):
        return _CursorPaginator(cfg)
    if isinstance(cfg, LinkHeaderPagination):
        return _LinkHeaderPaginator(cfg)
    raise ValueError(f"unsupported pagination config: {cfg!r}")  # pragma: no cover
