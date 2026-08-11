from apiflow.config import (
    CursorPagination,
    LinkHeaderPagination,
    NoPagination,
    OffsetPagination,
    PagePagination,
)
from apiflow.paginate import make_paginator, parse_link_header


def drive(cfg, page_sizes, bodies=None, headers=None, base_params=None):
    """Simulate a full pagination loop and return the params sent for each request."""
    p = make_paginator(cfg)
    sent = []
    for i in range(len(page_sizes) + 3):  # extra iterations prove it terminates
        req = p.next_request("https://api.test/items", base_params or {})
        if req is None:
            break
        sent.append(req)
        count = page_sizes[i] if i < len(page_sizes) else 0
        body = (bodies or [{}] * 10)[i] if bodies else {}
        header = (headers or [{}] * 10)[i] if headers else {}
        p.observe(body, header, count)
    return sent


def test_none_pagination_issues_exactly_one_request():
    sent = drive(NoPagination(), [50])
    assert len(sent) == 1
    assert sent[0][1] == {}


def test_page_pagination_stops_on_short_page():
    cfg = PagePagination(type="page", size=100)
    sent = drive(cfg, [100, 100, 40])
    assert len(sent) == 3
    assert [s[1]["page"] for s in sent] == [1, 2, 3]
    assert all(s[1]["per_page"] == 100 for s in sent)


def test_page_pagination_stops_on_empty_page():
    sent = drive(PagePagination(type="page", size=10), [10, 0])
    assert len(sent) == 2


def test_page_pagination_honours_total_pages_path():
    cfg = PagePagination(type="page", size=2, total_pages_path="meta.total_pages")
    sent = drive(cfg, [2, 2, 2], bodies=[{"meta": {"total_pages": 2}}] * 3)
    assert len(sent) == 2


def test_page_pagination_respects_max_pages():
    sent = drive(PagePagination(type="page", size=5, max_pages=2), [5, 5, 5, 5])
    assert len(sent) == 2


def test_offset_pagination_advances_by_limit():
    sent = drive(OffsetPagination(type="offset", limit=50), [50, 50, 10])
    assert [s[1]["offset"] for s in sent] == [0, 50, 100]
    assert all(s[1]["limit"] == 50 for s in sent)


def test_cursor_pagination_follows_body_cursor_then_stops():
    cfg = CursorPagination(type="cursor", cursor_path="meta.next", cursor_param="after")
    bodies = [{"meta": {"next": "c1"}}, {"meta": {"next": "c2"}}, {"meta": {"next": None}}]
    sent = drive(cfg, [10, 10, 10], bodies=bodies)
    assert len(sent) == 3
    assert "after" not in sent[0][1]          # first request carries no cursor
    assert sent[1][1]["after"] == "c1"
    assert sent[2][1]["after"] == "c2"


def test_cursor_pagination_stops_when_cursor_key_absent():
    cfg = CursorPagination(type="cursor", cursor_path="meta.next")
    sent = drive(cfg, [10, 10], bodies=[{}, {}])
    assert len(sent) == 1


def test_link_header_pagination_follows_next_url():
    cfg = LinkHeaderPagination(type="link_header")
    headers = [
        {"link": '<https://api.test/items?page=2>; rel="next", <https://x>; rel="last"'},
        {"link": '<https://api.test/items?page=1>; rel="prev"'},
    ]
    sent = drive(cfg, [100, 40], headers=headers)
    assert len(sent) == 2
    assert sent[1][0] == "https://api.test/items?page=2"
    assert sent[1][1] == {}  # the next URL already carries its query string


def test_base_params_are_preserved_and_not_mutated():
    base = {"state": "all"}
    sent = drive(PagePagination(type="page", size=5), [5, 1], base_params=base)
    assert base == {"state": "all"}
    assert all(s[1]["state"] == "all" for s in sent)


def test_parse_link_header():
    header = '<https://a/1>; rel="prev", <https://a/3>; rel="next"'
    assert parse_link_header(header, "next") == "https://a/3"
    assert parse_link_header(header, "prev") == "https://a/1"
    assert parse_link_header(header, "last") is None
    assert parse_link_header("garbage", "next") is None
