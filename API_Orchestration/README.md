# apiflow

A config-driven orchestrator that pulls REST API data into Postgres. Adding a new
source is a YAML file, not a Python module.

```
sources/*.yaml  ──▶  extract  ──▶  transform  ──▶  load  ──▶  Postgres
                    (auth,        (path map,      (batched      raw.<table>
                     paging,       type coercion)  upsert)      apiflow.runs
                     retries)                                   apiflow.cursors
```

## Quick start

```bash
docker compose up -d                     # Postgres on localhost:5432
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env                     # then edit

apiflow validate                         # check every source config
apiflow run pypi_files --dry-run         # fetch + transform, write nothing
apiflow init-db                          # create bookkeeping + target tables
apiflow run pypi_files                   # for real
apiflow status                           # run history
apiflow schedule                         # foreground cron loop
```

`pypi_files` needs no credentials, so it works immediately after setup and is the
fastest way to confirm the pipeline is wired up correctly.

## Adding a source

Copy `sources/_template.yaml.example` to `sources/<name>.yaml` and fill it in. The
template documents every option inline. A minimal source is:

```yaml
name: widgets
request:
  url: https://api.example.com/v1/widgets
records_path: data          # where the array lives; "$" if the body is the array
fields:
  id:         { path: id,              type: bigint, required: true }
  name:       { path: attributes.name, type: text }
  updated_at: { path: updated_at,      type: timestamp }
target:
  schema: raw
  table: widgets
  primary_key: [id]
```

Then `apiflow validate --show-ddl` prints the exact table it will create, and
`apiflow run widgets --dry-run` shows the rows it would write.

### Field paths

`path` is a small JSONPath subset evaluated against one record:

| Path | Meaning |
| --- | --- |
| `id` | top-level key |
| `user.login` | nested key |
| `labels[0].name` | array index (negative indices work) |
| `$` | the whole record, useful with `type: json` |

A path that doesn't resolve produces `NULL` rather than an error, unless the field
is marked `required: true`.

### Column types

`text`, `int`, `bigint`, `float`, `numeric`, `bool`, `timestamp`, `date`, `json`
map to the obvious Postgres types. Coercion is deliberately forgiving: `"1,234"`
becomes `1234`, `"yes"` becomes `true`, and timestamps parse from ISO-8601,
epoch seconds, epoch milliseconds, or common `YYYY-MM-DD HH:MM:SS` formats. A
naive timestamp is assumed to be UTC.

### Auth

`none`, `bearer`, `api_key` (header or query), `basic`, and
`oauth2_client_credentials` (tokens are fetched once and cached until expiry).
Never put a secret in the YAML — reference an environment variable instead:

```yaml
auth:
  type: bearer
  token: ${GITHUB_TOKEN}
```

`${VAR}` fails loudly if the variable is unset; `${VAR:-fallback}` supplies a
default. `.env` in the working directory is loaded automatically.

### Pagination

| Type | Shape | Stops when |
| --- | --- | --- |
| `none` | one request | always |
| `page` | `?page=1&per_page=100` | short/empty page, or `total_pages_path` |
| `offset` | `?offset=0&limit=100` | fewer records than `limit` |
| `cursor` | next cursor read from the body via `cursor_path` | cursor is null/absent |
| `link_header` | `Link: <url>; rel="next"` | no `next` link |

Every strategy also respects `max_pages` as a safety valve.

### Incremental loading

```yaml
incremental:
  mode: timestamp
  field: updated_at         # a column from `fields`
  param: since              # the query param the API expects
  lookback_seconds: 300     # re-fetch a small overlap for late arrivals
```

After a successful run the highest `updated_at` seen is stored in
`apiflow.cursors` and sent as `?since=...` next time. `apiflow run <src>
--full-refresh` ignores the stored value for one run; `apiflow reset <src>`
clears it permanently.

## What lands in Postgres

Each source writes one table containing your mapped columns plus:

| Column | Purpose |
| --- | --- |
| `_source` | which source produced the row |
| `_extracted_at` | when it was fetched |
| `_run_id` | joins to `apiflow.runs` |
| `_raw` | the untouched API payload as `jsonb` |

Keeping `_raw` means a mapping mistake is recoverable without re-hitting the API —
you can backfill a new column straight from the JSON. Set `store_raw: false` if
the payloads are large and you don't want the storage.

Bookkeeping lives in the `apiflow` schema: `runs` (one row per execution with
timings, counts, and errors), `cursors` (incremental watermarks), and `rejects`
(records that failed transformation, kept with the reason).

```sql
-- what ran recently and how it went
select source, status, rows_written, duration_ms, error
from apiflow.runs order by started_at desc limit 20;

-- why records were dropped
select reason, count(*) from apiflow.rejects group by 1 order by 2 desc;
```

## Write modes

`upsert` (default) requires a `primary_key` and updates existing rows in place.
`append` inserts unconditionally — right for immutable event streams. `replace`
truncates and reloads inside the same transaction, for small reference tables
with no natural key.

## Scheduling

Give a source a 5-field cron expression (UTC) and run `apiflow schedule`. It runs
in the foreground, executes due sources in a thread pool, and logs each result.

Because each run takes a Postgres advisory lock on its source name, you can run
several scheduler processes for redundancy without any risk of double-loading —
the second one sees the lock and skips that tick. That also makes it safe to
trigger `apiflow run` by hand while the scheduler is up.

For production, run `apiflow schedule` under systemd or as a container with
`restart: unless-stopped`. If you'd rather not run a long-lived process, drop the
`schedule:` keys and invoke `apiflow run <source>` from cron or Task Scheduler.

## Failure behaviour

Each run is a single transaction. If page 7 of 9 fails, nothing is committed and
the watermark doesn't move, so the next run re-reads the same window. Partial
loads never reach the target table.

Retries use exponential backoff with jitter on 429 and 5xx responses and honour
`Retry-After`. A 4xx other than 429 fails immediately — retrying a 404 or a 403
just wastes the rate limit. A 401 mid-run triggers one credential refresh before
giving up, which covers tokens that expire between pages.

Individual malformed records are written to `apiflow.rejects` and the rest of the
run proceeds. Pass `--strict` to fail the whole run on the first bad record.

Set `min_request_interval` on rate-limited APIs to space requests out; it's a
floor on wall-clock time between calls, applied before each attempt.

## Tests

```bash
pytest
```

113 checks, no network and no database required — HTTP is mocked with
`httpx.MockTransport`, and generated SQL is parsed by `pglast` (libpg_query, the
real PostgreSQL grammar) rather than compared to strings.

## Layout

| Path | Contents |
| --- | --- |
| `apiflow/config.py` | YAML schema, validation, env expansion |
| `apiflow/paths.py` | JSONPath subset used by field mappings |
| `apiflow/auth.py` | auth strategies |
| `apiflow/paginate.py` | pagination state machines |
| `apiflow/extract.py` | HTTP loop, retries, rate limiting |
| `apiflow/transform.py` | field mapping and type coercion |
| `apiflow/sqlgen.py` | SQL generation (no driver import, unit-testable) |
| `apiflow/db.py` | connections, DDL, run bookkeeping |
| `apiflow/load.py` | batching and upserts |
| `apiflow/runner.py` | one source, end to end |
| `apiflow/scheduler.py` | cron loop |
| `apiflow/cli.py` | command line entry point |

## Extending it

Adding a new pagination style means one class in `paginate.py` plus a variant in
the `PaginationConfig` union — the HTTP loop doesn't change. Same for auth: a new
branch in `Authenticator.apply` and a model in `config.py`. Non-JSON payloads
(CSV, XML) would hook in at `Extractor._decode`, which is the only place response
bytes are interpreted.
