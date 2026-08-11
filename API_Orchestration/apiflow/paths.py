"""Tiny path accessor for JSON payloads.

Supports the subset of JSONPath that API mapping actually needs:

    "$"                 -> the whole document
    "id"                -> obj["id"]
    "user.name"         -> obj["user"]["name"]
    "items[0].sku"      -> obj["items"][0]["sku"]
    "tags[-1]"          -> last element
    "a.b"               -> also matches a literal key "a.b" if present

Missing keys, out-of-range indices, and type mismatches all return the sentinel
`MISSING` rather than raising, so a partially-populated API response degrades to
NULL columns instead of a failed run.
"""

from __future__ import annotations

import re
from typing import Any

MISSING = object()

_SEGMENT_RE = re.compile(r"([^.\[\]]+)|\[(-?\d+)\]")


def parse_path(path: str) -> list[str | int]:
    """Split a path string into a list of dict keys (str) and list indices (int)."""
    if path in ("$", ""):
        return []
    if path.startswith("$."):
        path = path[2:]
    segments: list[str | int] = []
    pos = 0
    for m in _SEGMENT_RE.finditer(path):
        if m.start() > pos and path[pos : m.start()] not in (".",):
            raise ValueError(f"malformed path segment in {path!r} at offset {pos}")
        key, index = m.group(1), m.group(2)
        segments.append(key if key is not None else int(index))
        pos = m.end()
    if pos != len(path):
        raise ValueError(f"malformed path: {path!r}")
    if not segments:
        raise ValueError(f"malformed path: {path!r}")
    return segments


def dig(obj: Any, path: str, default: Any = MISSING) -> Any:
    """Follow `path` into `obj`, returning `default` if any step fails."""
    # Fast path: the whole key exists verbatim (handles keys containing dots).
    if isinstance(obj, dict) and path in obj:
        return obj[path]

    try:
        segments = parse_path(path)
    except ValueError:
        return default

    cur = obj
    for seg in segments:
        if isinstance(seg, int):
            if not isinstance(cur, (list, tuple)):
                return default
            try:
                cur = cur[seg]
            except IndexError:
                return default
        else:
            if not isinstance(cur, dict) or seg not in cur:
                return default
            cur = cur[seg]
    return cur
