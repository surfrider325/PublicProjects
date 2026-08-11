from datetime import datetime, timezone
from decimal import Decimal

import pytest

from apiflow.config import FieldSpec
from apiflow.paths import MISSING, dig, parse_path
from apiflow.transform import TransformError, coerce, extract_records, transform_record


def test_parse_path_variants():
    assert parse_path("$") == []
    assert parse_path("id") == ["id"]
    assert parse_path("user.name") == ["user", "name"]
    assert parse_path("labels[0].name") == ["labels", 0, "name"]
    assert parse_path("$.data.items[2]") == ["data", "items", 2]


def test_dig_nested_and_indexed():
    doc = {"user": {"name": "ada"}, "labels": [{"name": "bug"}, {"name": "docs"}]}
    assert dig(doc, "user.name") == "ada"
    assert dig(doc, "labels[1].name") == "docs"
    assert dig(doc, "labels[-1].name") == "docs"
    assert dig(doc, "$") is doc


def test_dig_missing_returns_sentinel_not_exception():
    doc = {"a": {"b": 1}}
    assert dig(doc, "a.c") is MISSING
    assert dig(doc, "a.b.c.d") is MISSING
    assert dig(doc, "list[5]") is MISSING
    assert dig(doc, "a.c", default=None) is None


def test_dig_prefers_literal_key_with_dots():
    assert dig({"a.b": 7, "a": {"b": 1}}, "a.b") == 7


@pytest.mark.parametrize(
    "value,expected",
    [
        (5, 5),
        ("5", 5),
        (" 1,234 ", 1234),
        (5.9, 5),
        (True, 1),
        ("", None),
    ],
)
def test_coerce_int(value, expected):
    assert coerce(value, "bigint") == expected


@pytest.mark.parametrize("value", ["abc", "1.2.3"])
def test_coerce_int_rejects_garbage(value):
    with pytest.raises(TransformError):
        coerce(value, "bigint")


@pytest.mark.parametrize(
    "value,expected", [("yes", True), ("N", False), (1, True), (0, False), (False, False)]
)
def test_coerce_bool(value, expected):
    assert coerce(value, "bool") == expected


def test_coerce_numeric_is_exact():
    assert coerce("19.99", "numeric") == Decimal("19.99")


def test_coerce_timestamp_formats():
    expected = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
    assert coerce("2026-08-11T12:00:00Z", "timestamp") == expected
    assert coerce("2026-08-11T12:00:00+00:00", "timestamp") == expected
    assert coerce(1786449600, "timestamp") == expected          # epoch seconds
    assert coerce(1786449600000, "timestamp") == expected       # epoch millis
    assert coerce("1786449600", "timestamp") == expected        # epoch as a string
    # A naive timestamp is assumed to be UTC rather than rejected.
    assert coerce("2026-08-11 12:00:00", "timestamp") == expected


def test_coerce_json_and_text_of_containers():
    assert coerce({"a": 1}, "json") == '{"a": 1}'
    assert coerce([1, 2], "text") == "[1, 2]"


def test_coerce_nan_becomes_null():
    assert coerce(float("nan"), "float") is None


FIELDS = {
    "id": FieldSpec(path="id", type="bigint", required=True),
    "author": FieldSpec(path="user.login", type="text"),
    "active": FieldSpec(path="active", type="bool", default=False),
    "missing": FieldSpec(path="nope.nothing", type="text"),
}


def test_transform_record_maps_nested_and_defaults():
    row = transform_record({"id": "42", "user": {"login": "ada"}}, FIELDS)
    assert row == {"id": 42, "author": "ada", "active": False, "missing": None}


def test_transform_record_requires_required_fields():
    with pytest.raises(TransformError) as exc:
        transform_record({"user": {"login": "ada"}}, FIELDS)
    assert "id" in str(exc.value)


def test_extract_records_paths():
    assert extract_records([1, 2], "$") == [1, 2]
    assert extract_records({"data": {"items": [1]}}, "data.items") == [1]
    assert extract_records({"nothing": 1}, "data.items") == []
    # A single-object endpoint is normalised to a one-element list.
    assert extract_records({"id": 1}, "$") == [{"id": 1}]
