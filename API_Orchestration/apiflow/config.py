"""Source configuration: YAML -> validated pydantic models.

A "source" is one API endpoint mapped to one Postgres table. Everything about
how to call the API, paginate it, map its fields, and write it lives in a single
YAML file under sources/. No Python is required to add a new source.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ${VAR} or ${VAR:-default}
_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(Exception):
    """Raised when a source YAML file is malformed or missing required env vars."""


def expand_env(value: Any, *, strict: bool = True) -> Any:
    """Recursively expand ${ENV_VAR} references in strings, lists, and dicts."""
    if isinstance(value, str):

        def sub(m: re.Match[str]) -> str:
            name, default = m.group(1), m.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            if strict:
                raise ConfigError(f"environment variable {name!r} is referenced but not set")
            return ""

        return _ENV_RE.sub(sub, value)
    if isinstance(value, dict):
        return {k: expand_env(v, strict=strict) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v, strict=strict) for v in value]
    return value


class Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- auth


class NoAuth(Base):
    type: Literal["none"] = "none"


class BearerAuth(Base):
    type: Literal["bearer"]
    token: str
    header: str = "Authorization"
    prefix: str = "Bearer"


class ApiKeyAuth(Base):
    type: Literal["api_key"]
    key: str
    name: str = "X-API-Key"
    location: Literal["header", "query"] = "header"


class BasicAuth(Base):
    type: Literal["basic"]
    username: str
    password: str


class OAuth2ClientCredentials(Base):
    type: Literal["oauth2_client_credentials"]
    token_url: str
    client_id: str
    client_secret: str
    scope: str | None = None
    audience: str | None = None
    # Some providers want credentials in the body rather than a Basic header.
    send_credentials_in: Literal["body", "header"] = "body"


AuthConfig = NoAuth | BearerAuth | ApiKeyAuth | BasicAuth | OAuth2ClientCredentials


# --------------------------------------------------------------------- pagination


class NoPagination(Base):
    type: Literal["none"] = "none"


class PagePagination(Base):
    """?page=1&per_page=100 style."""

    type: Literal["page"]
    page_param: str = "page"
    size_param: str | None = "per_page"
    size: int = 100
    start_page: int = 1
    max_pages: int = 1000
    total_pages_path: str | None = None  # e.g. "meta.total_pages" for an exact stop


class OffsetPagination(Base):
    """?offset=0&limit=100 style."""

    type: Literal["offset"]
    offset_param: str = "offset"
    limit_param: str = "limit"
    limit: int = 100
    max_pages: int = 1000


class CursorPagination(Base):
    """Response carries an opaque cursor for the next page."""

    type: Literal["cursor"]
    cursor_path: str  # where in the response body the next cursor lives
    cursor_param: str = "cursor"
    size_param: str | None = None
    size: int = 100
    max_pages: int = 1000


class LinkHeaderPagination(Base):
    """RFC 5988 Link: <url>; rel="next" (GitHub, Stripe-ish APIs)."""

    type: Literal["link_header"]
    rel: str = "next"
    max_pages: int = 1000


PaginationConfig = (
    NoPagination | PagePagination | OffsetPagination | CursorPagination | LinkHeaderPagination
)


# ------------------------------------------------------------------------ request


class RequestConfig(Base):
    url: str
    method: Literal["GET", "POST"] = "GET"
    headers: dict[str, str] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    json_body: dict[str, Any] | None = None
    timeout: float = 30.0


class RetryConfig(Base):
    max_attempts: int = 5
    backoff_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 60.0
    retry_on_status: list[int] = Field(default_factory=lambda: [408, 425, 429, 500, 502, 503, 504])
    # Honour a Retry-After header when the server sends one.
    respect_retry_after: bool = True


# ------------------------------------------------------------------------- fields

ColumnType = Literal[
    "text", "int", "bigint", "float", "numeric", "bool", "timestamp", "date", "json"
]

PG_TYPES: dict[str, str] = {
    "text": "text",
    "int": "integer",
    "bigint": "bigint",
    "float": "double precision",
    "numeric": "numeric",
    "bool": "boolean",
    "timestamp": "timestamptz",
    "date": "date",
    "json": "jsonb",
}


class FieldSpec(Base):
    path: str  # dot/bracket path into the record, e.g. "user.address[0].city"
    type: ColumnType = "text"
    required: bool = False
    default: Any = None


# ------------------------------------------------------------------- incremental


class IncrementalConfig(Base):
    """Only pull records newer than the high-water mark from the previous run."""

    mode: Literal["none", "timestamp"] = "none"
    # Column (as named in `fields`) holding the watermark value.
    field: str | None = None
    # Query parameter used to send the watermark to the API.
    param: str | None = None
    # Optional strftime format; default is ISO-8601 with timezone.
    format: str | None = None
    # Re-fetch a little before the watermark to tolerate late-arriving records.
    lookback_seconds: int = 0

    @model_validator(mode="after")
    def _check(self) -> IncrementalConfig:
        if self.mode == "timestamp" and not (self.field and self.param):
            raise ValueError("incremental.mode=timestamp requires both 'field' and 'param'")
        return self


# ----------------------------------------------------------------------- target


class TargetConfig(Base):
    schema_: str = Field(default="raw", alias="schema")
    table: str
    primary_key: list[str] = Field(default_factory=list)
    mode: Literal["upsert", "append", "replace"] = "upsert"
    # Keep the untouched API payload alongside the typed columns.
    store_raw: bool = True
    batch_size: int = 1000

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @model_validator(mode="after")
    def _check(self) -> TargetConfig:
        if self.mode == "upsert" and not self.primary_key:
            raise ValueError("target.mode=upsert requires a non-empty primary_key")
        return self


# ----------------------------------------------------------------------- source


class SourceConfig(Base):
    name: str
    enabled: bool = True
    description: str | None = None
    schedule: str | None = None  # 5-field cron expression
    request: RequestConfig
    auth: AuthConfig = Field(default_factory=NoAuth, discriminator="type")
    pagination: PaginationConfig = Field(default_factory=NoPagination, discriminator="type")
    retry: RetryConfig = Field(default_factory=RetryConfig)
    # Minimum seconds between HTTP requests, for rate-limited APIs.
    min_request_interval: float = 0.0
    # Path to the array of records in the response. "$" means the body is the array.
    records_path: str = "$"
    fields: dict[str, FieldSpec]
    incremental: IncrementalConfig = Field(default_factory=IncrementalConfig)
    target: TargetConfig

    @field_validator("name")
    @classmethod
    def _name_is_identifier(cls, v: str) -> str:
        if not re.fullmatch(r"[a-z0-9_]+", v):
            raise ValueError("name must be lowercase letters, digits, and underscores only")
        return v

    @field_validator("fields")
    @classmethod
    def _fields_not_empty(cls, v: dict[str, FieldSpec]) -> dict[str, FieldSpec]:
        if not v:
            raise ValueError("at least one field mapping is required")
        for col in v:
            if not re.fullmatch(r"[a-z_][a-z0-9_]*", col):
                raise ValueError(f"column {col!r} must be a lowercase SQL identifier")
            if col.startswith("_"):
                raise ValueError(f"column {col!r} is reserved (leading underscore)")
        return v

    @model_validator(mode="after")
    def _cross_checks(self) -> SourceConfig:
        missing = [c for c in self.target.primary_key if c not in self.fields]
        if missing:
            raise ValueError(f"primary_key columns not present in fields: {missing}")
        if self.incremental.mode == "timestamp" and self.incremental.field not in self.fields:
            raise ValueError(
                f"incremental.field {self.incremental.field!r} is not a mapped field"
            )
        if self.schedule:
            from croniter import croniter

            if not croniter.is_valid(self.schedule):
                raise ValueError(f"invalid cron expression: {self.schedule!r}")
        return self


def load_source(path: str | Path, *, strict_env: bool = True) -> SourceConfig:
    """Parse and validate a single source YAML file."""
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name}: expected a YAML mapping at the top level")
    data = expand_env(data, strict=strict_env)
    try:
        return SourceConfig.model_validate(data)
    except Exception as exc:  # pydantic ValidationError
        raise ConfigError(f"{path.name}: {exc}") from exc


def load_all_sources(directory: str | Path, *, strict_env: bool = True) -> list[SourceConfig]:
    """Load every *.yaml / *.yml file in a directory, sorted by name."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ConfigError(f"sources directory not found: {directory}")
    files = sorted(p for p in directory.iterdir() if p.suffix in {".yaml", ".yml"})
    sources = [load_source(p, strict_env=strict_env) for p in files]
    seen: set[str] = set()
    for s in sources:
        if s.name in seen:
            raise ConfigError(f"duplicate source name: {s.name}")
        seen.add(s.name)
    return sources
