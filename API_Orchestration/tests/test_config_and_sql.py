import textwrap
from pathlib import Path

import pytest

from apiflow import sqlgen
from apiflow.config import ConfigError, SourceConfig, expand_env, load_all_sources, load_source

MINIMAL = """
name: widgets
request:
  url: https://api.test/widgets
records_path: data
fields:
  id: {path: id, type: bigint, required: true}
  name: {path: attrs.name, type: text}
target:
  schema: raw
  table: widgets
  primary_key: [id]
"""


def write(tmp_path: Path, text: str, name: str = "s.yaml") -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def test_load_minimal_source_applies_defaults(tmp_path):
    src = load_source(write(tmp_path, MINIMAL))
    assert src.name == "widgets"
    assert src.enabled is True
    assert src.auth.type == "none"
    assert src.pagination.type == "none"
    assert src.target.mode == "upsert"
    assert src.target.schema_ == "raw"
    assert src.retry.max_attempts == 5


def test_env_expansion_with_default_and_strict_error(monkeypatch):
    monkeypatch.setenv("TOK", "secret")
    assert expand_env({"a": "${TOK}"}) == {"a": "secret"}
    assert expand_env("${NOPE:-fallback}") == "fallback"
    assert expand_env(["${TOK}"]) == ["secret"]
    with pytest.raises(ConfigError):
        expand_env("${DEFINITELY_NOT_SET}")
    assert expand_env("${DEFINITELY_NOT_SET}", strict=False) == ""


def test_upsert_requires_primary_key(tmp_path):
    bad = MINIMAL.replace("  primary_key: [id]\n", "")
    with pytest.raises(ConfigError) as exc:
        load_source(write(tmp_path, bad))
    assert "primary_key" in str(exc.value)


def test_primary_key_must_be_a_mapped_field(tmp_path):
    bad = MINIMAL.replace("primary_key: [id]", "primary_key: [sku]")
    with pytest.raises(ConfigError):
        load_source(write(tmp_path, bad))


def test_reserved_and_invalid_column_names_rejected(tmp_path):
    bad = MINIMAL.replace("  name: {path: attrs.name, type: text}", "  _raw: {path: x}")
    with pytest.raises(ConfigError):
        load_source(write(tmp_path, bad))
    bad2 = MINIMAL.replace("  name: {path: attrs.name, type: text}", "  Drop Table: {path: x}")
    with pytest.raises(ConfigError):
        load_source(write(tmp_path, bad2))


def test_unknown_key_is_rejected_not_ignored(tmp_path):
    bad = MINIMAL + "\ntyop_key: 1\n"
    with pytest.raises(ConfigError):
        load_source(write(tmp_path, bad))


def test_invalid_cron_rejected(tmp_path):
    bad = MINIMAL + '\nschedule: "not a cron"\n'
    with pytest.raises(ConfigError):
        load_source(write(tmp_path, bad))


def test_incremental_requires_field_and_param(tmp_path):
    bad = MINIMAL + "\nincremental:\n  mode: timestamp\n  field: id\n"
    with pytest.raises(ConfigError):
        load_source(write(tmp_path, bad))


def test_duplicate_source_names_rejected(tmp_path):
    write(tmp_path, MINIMAL, "a.yaml")
    write(tmp_path, MINIMAL, "b.yaml")
    with pytest.raises(ConfigError) as exc:
        load_all_sources(tmp_path)
    assert "duplicate" in str(exc.value)


def test_shipped_example_sources_are_valid():
    root = Path(__file__).resolve().parents[1] / "sources"
    sources = load_all_sources(root, strict_env=False)
    assert {s.name for s in sources} == {
        "jsonplaceholder_posts",
        "github_issues",
        "pypi_files",
    }


# ------------------------------------------------------------------------- SQL


@pytest.fixture
def source(tmp_path):
    return load_source(write(tmp_path, MINIMAL))


def test_create_table_includes_types_meta_and_pk(source):
    ddl = sqlgen.create_table(source)
    assert 'create table if not exists "raw"."widgets"' in ddl
    assert '"id" bigint not null' in ddl
    assert '"name" text' in ddl
    assert '"_source" text not null' in ddl
    assert '"_extracted_at" timestamptz' in ddl
    assert '"_raw" jsonb' in ddl
    assert 'primary key ("id")' in ddl


def test_insert_statement_is_an_upsert_with_matching_placeholders(source):
    stmt = sqlgen.insert_statement(source)
    columns = sqlgen.target_columns(source)
    assert stmt.count("%s") == len(columns)
    assert 'on conflict ("id") do update set' in stmt
    # The primary key is never overwritten by the update clause.
    assert '"id" = excluded."id"' not in stmt
    assert '"name" = excluded."name"' in stmt
    # _extracted_at is refreshed rather than copied.
    assert '"_extracted_at" = now()' in stmt


def test_append_mode_has_no_conflict_clause(tmp_path):
    src = load_source(write(tmp_path, MINIMAL.replace("primary_key: [id]", "mode: append")))
    assert "on conflict" not in sqlgen.insert_statement(src)


def test_column_order_is_stable_across_calls(source):
    assert sqlgen.target_columns(source) == sqlgen.target_columns(source)
    assert sqlgen.target_columns(source)[:2] == ["id", "name"]


def test_identifier_quoting_blocks_injection():
    with pytest.raises(sqlgen.SqlError):
        sqlgen.ident('widgets"; drop table users; --')
    with pytest.raises(sqlgen.SqlError):
        sqlgen.ident("")
    assert sqlgen.ident("widgets") == '"widgets"'


def test_add_missing_columns_only_targets_absent_ones(source):
    present = {"id", "_source", "_extracted_at", "_run_id", "_raw"}
    statements = sqlgen.add_missing_columns(source, present)
    assert len(statements) == 1
    assert '"name" text' in statements[0]
    assert statements[0].startswith('alter table "raw"."widgets" add column')
