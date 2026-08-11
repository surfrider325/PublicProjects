"""Parse every generated statement with libpg_query, the real PostgreSQL parser.

This catches syntax mistakes in generated DDL/DML without needing a live server.
Install the parser with:  pip install pglast
"""

from pathlib import Path

import pytest

from apiflow import sqlgen
from apiflow.config import load_all_sources

pglast = pytest.importorskip("pglast", reason="pglast not installed")

SOURCES_DIR = Path(__file__).resolve().parents[1] / "sources"


def statements_for_all_sources():
    out: list[tuple[str, str]] = [("meta ddl", sqlgen.META_DDL)]
    for source in load_all_sources(SOURCES_DIR, strict_env=False):
        out.append((f"{source.name} create schema", sqlgen.create_schema(source.target.schema_)))
        out.append((f"{source.name} create table", sqlgen.create_table(source)))
        out.append((f"{source.name} insert", sqlgen.insert_statement(source)))
        out.append((f"{source.name} truncate", sqlgen.truncate(source)))
        for i, alter in enumerate(sqlgen.add_missing_columns(source, set())):
            out.append((f"{source.name} alter {i}", alter))
        if source.incremental.mode == "timestamp":
            out.append((f"{source.name} watermark", sqlgen.max_watermark(source)))
    return out


@pytest.mark.parametrize(
    "label,sql", statements_for_all_sources(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_generated_sql_parses(label, sql):
    # Bind placeholders are not part of the SQL grammar; substitute a literal.
    pglast.parse_sql(sql.replace("%s", "null"))
