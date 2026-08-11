"""Map raw API records to typed rows using the `fields` block of a source config."""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from .config import FieldSpec, SourceConfig
from .paths import MISSING, dig

_TRUE = {"true", "t", "yes", "y", "1", "on"}
_FALSE = {"false", "f", "no", "n", "0", "off", ""}


class TransformError(Exception):
    """Raised when a required field is absent or a value cannot be coerced."""


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # Heuristic: values past ~year 2286 in seconds are really milliseconds.
        seconds = value / 1000.0 if abs(value) > 1e11 else float(value)
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            return _parse_timestamp(int(text))
        # datetime.fromisoformat on <3.11 chokes on a trailing "Z".
        normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%d/%m/%Y", "%m/%d/%Y"):
                try:
                    parsed = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
            else:
                raise TransformError(f"cannot parse timestamp: {value!r}")
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    raise TransformError(f"cannot parse timestamp: {value!r}")


def coerce(value: Any, column_type: str) -> Any:
    """Convert a raw JSON value to the Python type psycopg will bind correctly."""
    if value is None:
        return None

    if column_type == "json":
        return json.dumps(value, default=str)

    if column_type == "text":
        if isinstance(value, (dict, list)):
            return json.dumps(value, default=str)
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    if column_type == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return None if text == "" else False
        raise TransformError(f"cannot parse bool: {value!r}")

    if column_type in ("int", "bigint"):
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
            return int(value)
        text = str(value).strip().replace(",", "")
        if text == "":
            return None
        try:
            return int(Decimal(text))
        except (InvalidOperation, ValueError) as exc:
            raise TransformError(f"cannot parse {column_type}: {value!r}") from exc

    if column_type == "float":
        try:
            out = float(str(value).strip().replace(",", ""))
        except ValueError as exc:
            raise TransformError(f"cannot parse float: {value!r}") from exc
        return None if math.isnan(out) or math.isinf(out) else out

    if column_type == "numeric":
        try:
            return Decimal(str(value).strip().replace(",", ""))
        except (InvalidOperation, ValueError) as exc:
            raise TransformError(f"cannot parse numeric: {value!r}") from exc

    if column_type == "timestamp":
        return _parse_timestamp(value)

    if column_type == "date":
        if isinstance(value, date) and not isinstance(value, datetime):
            return value
        return _parse_timestamp(value).date()

    raise TransformError(f"unknown column type: {column_type}")


def transform_record(record: Any, fields: dict[str, FieldSpec]) -> dict[str, Any]:
    """Apply every field mapping to one API record. Raises TransformError on bad data."""
    row: dict[str, Any] = {}
    for column, spec in fields.items():
        raw = dig(record, spec.path)
        if raw is MISSING or raw is None:
            if spec.default is not None:
                raw = spec.default
            elif spec.required:
                raise TransformError(
                    f"required field {column!r} missing at path {spec.path!r}"
                )
            else:
                row[column] = None
                continue
        try:
            row[column] = coerce(raw, spec.type)
        except TransformError as exc:
            raise TransformError(f"column {column!r}: {exc}") from exc
    return row


def transform_batch(
    records: list[Any], source: SourceConfig, *, on_error: str = "fail"
) -> tuple[list[dict[str, Any]], list[tuple[Any, str]]]:
    """Transform a batch, returning (rows, rejects).

    on_error="fail" raises on the first bad record; "skip" collects it into
    `rejects` so one malformed record can't sink an otherwise good run.
    """
    rows: list[dict[str, Any]] = []
    rejects: list[tuple[Any, str]] = []
    for record in records:
        try:
            rows.append(transform_record(record, source.fields))
        except TransformError as exc:
            if on_error == "fail":
                raise
            rejects.append((record, str(exc)))
    return rows, rejects


def extract_records(payload: Any, records_path: str) -> list[Any]:
    """Pull the list of records out of a response body."""
    found = dig(payload, records_path)
    if found is MISSING:
        return []
    if isinstance(found, list):
        return found
    if isinstance(found, dict):
        return [found]  # single-object endpoint
    return []
