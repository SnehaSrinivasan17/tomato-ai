# Durable review storage

When `REVIEW_DATABASE_URL` is configured, the review workflow uses PostgreSQL
instead of ephemeral local SQLite. `REVIEW_STORAGE_BACKEND="postgresql"` can make
this requirement explicit: an unavailable database produces a sanitized error,
never a local fallback. `sqlite` is an explicit offline-development option.

Set actual values only in private Streamlit secrets or environment variables:

```toml
REVIEW_DATABASE_URL = "postgresql://USER:PASSWORD@HOST/neondb?sslmode=require"
REVIEW_STORAGE_BACKEND = "postgresql"
REVIEW_DB_NAMESPACE = "tomato"
```

Use Neon's connection URI, not a `psql` shell command. Retain its TLS and
channel-binding parameters. Credentials, URLs and raw provider exceptions are not
included in application errors. The example files contain placeholders only.

## What persists

- Decision snapshots and every human-review revision, with the existing IDs,
  timestamps, draft hashes, notes, rule versions and output schema.
- Demo run history, version-aware pipeline cache and cache-use events.
- Active demo epoch and archived epoch references. Reset changes a database
  pointer without deleting history; it survives application restarts.
- Human classification annotations and policy-reference review revisions.

Each app namespace has a global schema, a control schema and schemas for each
demo epoch. An epoch handle contains no credential. All workers obtain the same
active epoch from PostgreSQL. Feed reads are batched rather than issuing two
database calls per post. Connections are short-lived and transaction-scoped;
Neon's pooled URI is supported.

Database triggers reject UPDATE/DELETE on audit snapshots, review events,
annotations and policy-reference revisions. Review updates are serialized with
transaction-level advisory locks and stale-revision checks. Identical duplicate
submissions are rejected. Snapshot/cache-index saves commit together or roll
back together. An approval still never sends or publishes a response.

These protections apply to normal application operations; the database owner
remains privileged. Storage configuration alone does not authenticate reviewers.
Before any Cloud deployment is shared, restrict application access to authorized
reviewers or configure application authentication/authorization. Do not make the
self-identified reviewer form publicly writable.

## Migrate existing local history

From the project root:

```powershell
python scripts/migrate_reviews.py --secrets-file .streamlit/secrets.toml
```

The import reads local SQLite files in read-only mode and retains all IDs and
stored payloads. Existing identical rows are skipped. Conflicting IDs abort that
scope's transaction rather than overwriting data. It also imports existing
annotations and policy-reference reviews. Local databases are retained as backups
and must remain Git-ignored.

On the initial migration, the local active session is adopted if no populated
remote session exists. Later migration runs do not reverse a remote demo reset.
The command is idempotent; repeating it must report zero new inserts for unchanged
sources. Migration reports contain counts only, not connection credentials.

## Verification performed locally against Neon

The initial import copied 5 global decisions, 11 demo decisions, 11 demo-run
index rows and 1 cache-use event. There were no existing human-review events or
annotations to import. A repeat import inserted zero rows. Source SQLite files
were not deleted or rewritten.

Eight real database checks passed in separate `qa_...` namespaces: review write,
fresh-interpreter readback, duplicate prevention, append-only database trigger,
concurrent review serialization, durable pipeline cache, non-destructive reset
and archived-record readback. These use explicitly labelled fictional fixtures,
not production human approvals or empirical LLM/ML evaluation.

All six local Streamlit sections were also exercised with Neon enabled, without
triggering paid model inference. Ordinary pytest tests force isolated SQLite so
they cannot modify live Neon records, even if private secrets are present. See
the local Git-ignored `phase7_results/` reports for actual run evidence. Successful
local database checks do not establish a completed Streamlit Cloud deployment;
the deployed application still requires its own live verification.
