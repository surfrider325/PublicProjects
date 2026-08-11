"""Extraction against a mocked transport — no network, no database."""

import json
import textwrap

import httpx
import pytest

from apiflow.auth import Authenticator
from apiflow.config import ApiKeyAuth, BasicAuth, BearerAuth, load_source
from apiflow.extract import ExtractError, Extractor
from apiflow.load import deduplicate

PAGED = """
name: things
request:
  url: https://api.test/things
pagination:
  type: page
  size: 2
retry:
  max_attempts: 3
  backoff_seconds: 0
records_path: data
fields:
  id: {path: id, type: bigint, required: true}
  name: {path: name, type: text}
target:
  schema: raw
  table: things
  primary_key: [id]
"""


def make_source(tmp_path, text=PAGED, name="s.yaml"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return load_source(path)


def client_for(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_extractor_walks_all_pages(tmp_path):
    pages = {
        1: [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
        2: [{"id": 3, "name": "c"}],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        page = int(request.url.params["page"])
        return httpx.Response(200, json={"data": pages.get(page, [])})

    with Extractor(make_source(tmp_path), client=client_for(handler)) as ex:
        records = list(ex.iter_records())

    assert [r["id"] for r in records] == [1, 2, 3]
    assert ex.requests_made == 2  # the short second page ends the loop


def test_extractor_retries_transient_500_then_succeeds(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json={"data": [{"id": 1, "name": "a"}]})

    with Extractor(make_source(tmp_path), client=client_for(handler)) as ex:
        records = list(ex.iter_records())

    assert calls["n"] == 2
    assert len(records) == 1


def test_extractor_does_not_retry_client_errors(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404, text="nope")

    with Extractor(make_source(tmp_path), client=client_for(handler)) as ex:
        with pytest.raises(ExtractError) as exc:
            list(ex.iter_records())

    assert calls["n"] == 1          # 404 is not retryable
    assert "404" in str(exc.value)


def test_extractor_gives_up_after_max_attempts(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unavailable")

    with Extractor(make_source(tmp_path), client=client_for(handler)) as ex:
        with pytest.raises(ExtractError) as exc:
            list(ex.iter_records())

    assert "after 3 attempts" in str(exc.value)


def test_non_json_response_is_a_clear_error(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>maintenance</html>")

    with Extractor(make_source(tmp_path), client=client_for(handler)) as ex:
        with pytest.raises(ExtractError) as exc:
            list(ex.iter_records())

    assert "non-JSON" in str(exc.value)


def test_incremental_watermark_becomes_a_query_param(tmp_path):
    from datetime import datetime, timezone

    config = PAGED.replace(
        "pagination:\n  type: page\n  size: 2\n", "pagination:\n  type: none\n"
    ) + textwrap.dedent(
        """
        incremental:
          mode: timestamp
          field: updated_at
          param: since
          lookback_seconds: 60
        """
    )
    config = config.replace(
        "  name: {path: name, type: text}",
        "  name: {path: name, type: text}\n  updated_at: {path: updated_at, type: timestamp}",
    )
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["since"] = request.url.params.get("since")
        return httpx.Response(200, json={"data": []})

    source = make_source(tmp_path, config)
    with Extractor(source, client=client_for(handler)) as ex:
        list(ex.iter_records(datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)))

    assert seen["since"] == "2026-08-11T11:59:00Z"  # 60s lookback applied


# ------------------------------------------------------------------------ auth


def _applied(auth):
    """These strategies are stateless, so no HTTP client is ever touched."""
    headers, params = {}, {}
    Authenticator(auth).apply(None, headers, params)
    return headers, params


def test_bearer_auth_header():
    headers, _ = _applied(BearerAuth(type="bearer", token="t0k"))
    assert headers["Authorization"] == "Bearer t0k"


def test_api_key_in_header_or_query():
    headers, _ = _applied(ApiKeyAuth(type="api_key", key="k", name="X-Key"))
    assert headers["X-Key"] == "k"
    _, params = _applied(
        ApiKeyAuth(type="api_key", key="k", name="api_key", location="query")
    )
    assert params["api_key"] == "k"


def test_basic_auth_is_base64_encoded():
    headers, _ = _applied(BasicAuth(type="basic", username="u", password="p"))
    assert headers["Authorization"] == "Basic dTpw"


def test_oauth2_token_is_fetched_once_and_cached():
    from apiflow.config import OAuth2ClientCredentials

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"access_token": "abc", "expires_in": 3600})

    cfg = OAuth2ClientCredentials(
        type="oauth2_client_credentials",
        token_url="https://api.test/token",
        client_id="id",
        client_secret="secret",
    )
    auth = Authenticator(cfg)
    client = client_for(handler)
    for _ in range(3):
        headers = {}
        auth.apply(client, headers, {})
        assert headers["Authorization"] == "Bearer abc"
    assert calls["n"] == 1


# ------------------------------------------------------------------------ load


def test_deduplicate_keeps_last_occurrence_per_key():
    rows = [{"id": 1, "v": "old"}, {"id": 2, "v": "x"}, {"id": 1, "v": "new"}]
    raws = ["r1", "r2", "r3"]
    out_rows, out_raws = deduplicate(rows, raws, ["id"])
    assert out_rows == [{"id": 1, "v": "new"}, {"id": 2, "v": "x"}]
    assert out_raws == ["r3", "r2"]


def test_deduplicate_is_a_noop_without_a_primary_key():
    rows = [{"id": 1}, {"id": 1}]
    assert deduplicate(rows, rows, [])[0] == rows


def test_loader_builds_tuples_in_column_order(tmp_path):
    from datetime import datetime, timezone

    from apiflow.load import Loader

    source = make_source(tmp_path)
    loader = Loader.__new__(Loader)  # no database connection needed
    loader.source = source
    loader.run_id = 7
    loader.extracted_at = datetime(2026, 8, 11, tzinfo=timezone.utc)

    tup = Loader._to_tuple(loader, {"id": 1, "name": "a"}, {"id": 1, "name": "a"})
    assert tup[0] == 1
    assert tup[1] == "a"
    assert tup[2] == "things"                 # _source
    assert tup[3] == loader.extracted_at      # _extracted_at
    assert tup[4] == 7                        # _run_id
    assert json.loads(tup[5]) == {"id": 1, "name": "a"}
