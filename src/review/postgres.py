"""Durable, namespaced PostgreSQL storage; a configured outage never falls back to SQLite.

Scope handles contain public identifiers only. Connections/transactions are short
lived, with append-only audit tables and database-side review serialization.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import re
from uuid import uuid4
from config import get_setting


class StorageError(RuntimeError):
    pass


@dataclass(frozen=True)
class Scope:
    kind: str = 'global'
    epoch: str = ''
    namespace: str = 'tomato'

    def __post_init__(self):
        if not re.fullmatch(r'[a-z][a-z0-9_]{0,15}', self.namespace):
            raise ValueError('REVIEW_DB_NAMESPACE must be a short lowercase identifier.')
        if self.kind not in {'global', 'demo', 'control'}:
            raise ValueError('Invalid review scope.')
        if self.kind == 'demo' and not re.fullmatch(r'[0-9a-f]{32}', self.epoch):
            raise ValueError('Invalid demo epoch.')
        if self.kind != 'demo' and self.epoch:
            raise ValueError('Only demo scopes have epochs.')

    @property
    def schema(self):
        return f'ta_{self.namespace}_{self.kind}' + ('_' + self.epoch if self.epoch else '')

    def __str__(self):
        return 'PostgreSQL · ' + self.kind + (' · ' + self.epoch if self.epoch else '')


def configured():
    backend = get_setting('REVIEW_STORAGE_BACKEND', 'auto').lower()
    if backend not in {'auto', 'sqlite', 'postgresql'}:
        raise StorageError('REVIEW_STORAGE_BACKEND must be auto, sqlite or postgresql.')
    return backend == 'postgresql' or (backend == 'auto' and bool(get_setting('REVIEW_DATABASE_URL')))


def scope(kind='global', epoch=''):
    return Scope(kind, epoch, get_setting('REVIEW_DB_NAMESPACE', 'tomato'))


def _dsn():
    value = get_setting('REVIEW_DATABASE_URL')
    if not value.startswith(('postgresql://', 'postgres://')):
        raise StorageError('REVIEW_DATABASE_URL must be a PostgreSQL connection URI, not a psql command.')
    return value


def _open():
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError:
        raise StorageError('Install the project psycopg[binary] dependency for PostgreSQL storage.') from None
    # libpq's SSL verification setting in the supplied URI is retained. Reject
    # plaintext/optional TLS modes; never include the URI in diagnostics.
    try:
        from psycopg.conninfo import conninfo_to_dict
        options = conninfo_to_dict(_dsn())
        if options.get('sslmode') not in {'require', 'verify-ca', 'verify-full'}:
            raise StorageError('REVIEW_DATABASE_URL must require TLS (sslmode=require or stricter).')
        connection = psycopg.connect(_dsn(), connect_timeout=15, row_factory=dict_row)
        try:
            # Transaction-local settings also work through Neon's pooled URI;
            # poolers may reject the libpq startup "options" parameter.
            connection.execute("SET LOCAL statement_timeout = '30s'")
            connection.execute("SET LOCAL lock_timeout = '15s'")
            return connection
        except Exception:
            connection.close()
            raise
    except StorageError:
        raise
    except Exception:
        raise StorageError('PostgreSQL connection failed. Check the private URI, TLS and database availability; no SQLite fallback was used.') from None


DDL = [
    'CREATE TABLE IF NOT EXISTS decisions (decision_id TEXT PRIMARY KEY,post_id TEXT NOT NULL,created_at TEXT NOT NULL,decision_json TEXT NOT NULL,pipeline_json TEXT NOT NULL,draft_version TEXT)',
    'CREATE TABLE IF NOT EXISTS review_events (event_id TEXT PRIMARY KEY,decision_id TEXT NOT NULL REFERENCES decisions(decision_id),revision INTEGER NOT NULL,status TEXT NOT NULL,reviewer TEXT NOT NULL,reviewed_at TEXT NOT NULL,notes TEXT NOT NULL,proposed_action TEXT NOT NULL,final_action TEXT NOT NULL,rule_ids TEXT NOT NULL,draft_version TEXT,edited_draft TEXT,validation_json TEXT,UNIQUE(decision_id,revision))',
    'CREATE INDEX IF NOT EXISTS decision_post ON decisions(post_id)',
    'CREATE TABLE IF NOT EXISTS demo_runs (decision_id TEXT PRIMARY KEY REFERENCES decisions(decision_id),post_id TEXT NOT NULL,cache_key TEXT NOT NULL,version_fingerprint TEXT NOT NULL,processed_at TEXT NOT NULL,elapsed_ms DOUBLE PRECISION NOT NULL,complete INTEGER NOT NULL,rowid BIGSERIAL UNIQUE)',
    'CREATE INDEX IF NOT EXISTS demo_cache_key ON demo_runs(cache_key,complete,rowid)',
    'CREATE TABLE IF NOT EXISTS demo_cache_hits (id BIGSERIAL PRIMARY KEY,decision_id TEXT NOT NULL REFERENCES decisions(decision_id),at TEXT NOT NULL)',
    'CREATE TABLE IF NOT EXISTS annotations (post_id TEXT,reviewer TEXT,version INTEGER,reviewed_at TEXT,human_verified INTEGER,labels TEXT,notes TEXT,PRIMARY KEY(post_id,reviewer,version))',
    'CREATE TABLE IF NOT EXISTS reference_reviews (event_id TEXT PRIMARY KEY,post_id TEXT NOT NULL,reviewed_at TEXT NOT NULL,payload TEXT NOT NULL,rowid BIGSERIAL UNIQUE)',
]


@lru_cache(maxsize=128)
def _initialize(schema, connection_fingerprint):
    from psycopg import sql
    try:
        with _open() as conn:
            conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))', ('initialize:' + schema,))
            conn.execute(sql.SQL('CREATE SCHEMA IF NOT EXISTS {}').format(sql.Identifier(schema)))
            conn.execute(sql.SQL('SET LOCAL search_path TO {}').format(sql.Identifier(schema)))
            if schema.endswith('_control'):
                conn.execute('CREATE TABLE IF NOT EXISTS app_state (name TEXT PRIMARY KEY,value TEXT NOT NULL)')
                conn.execute('CREATE TABLE IF NOT EXISTS demo_epochs (epoch TEXT PRIMARY KEY,created_at TEXT NOT NULL)')
            else:
                for statement in DDL:
                    conn.execute(statement)
                conn.execute("""CREATE OR REPLACE FUNCTION reject_audit_mutation() RETURNS trigger
                    LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'Audit records are append-only'; END $$""")
                for table in ['decisions', 'review_events', 'annotations', 'reference_reviews']:
                    # Initialization is serialized across app instances. Never
                    # drop/recreate triggers or overwrite existing audit rows.
                    exists = conn.execute('SELECT 1 FROM pg_trigger WHERE tgname=%s AND tgrelid=%s::regclass',
                                          ('append_only_' + table, table)).fetchone()
                    if not exists:
                        conn.execute(sql.SQL('CREATE TRIGGER {} BEFORE UPDATE OR DELETE ON {} FOR EACH ROW EXECUTE FUNCTION reject_audit_mutation()')
                                     .format(sql.Identifier('append_only_' + table), sql.Identifier(table)))
    except StorageError:
        raise
    except Exception:
        raise StorageError('PostgreSQL schema initialization failed. Check database permissions; no local fallback was used.') from None


class Row(dict):
    def __getitem__(self, key):
        return list(self.values())[key] if isinstance(key, int) else super().__getitem__(key)


class Cursor:
    def __init__(self, cursor):
        self.cursor = cursor

    @property
    def rowcount(self):
        return self.cursor.rowcount

    def fetchone(self):
        try:
            value = self.cursor.fetchone()
            return Row(value) if value is not None else None
        except Exception:
            raise StorageError('PostgreSQL results could not be read; no local fallback was used.') from None

    def fetchall(self):
        try:
            return [Row(value) for value in self.cursor.fetchall()]
        except Exception:
            raise StorageError('PostgreSQL results could not be read; no local fallback was used.') from None

    def __iter__(self):
        try:
            for value in self.cursor:yield Row(value)
        except Exception:
            raise StorageError('PostgreSQL results could not be read; no local fallback was used.') from None


class Connection:
    """Small adapter for this project's parameterized SQLite-style statements."""
    def __init__(self, target):
        from psycopg import sql
        self.target = target
        _initialize(target.schema, hashlib.sha256(_dsn().encode()).hexdigest())
        self.conn = _open()
        try:
            self.conn.execute(sql.SQL('SET LOCAL search_path TO {}').format(sql.Identifier(target.schema)))
        except Exception:
            self.conn.close()
            raise StorageError('PostgreSQL scope could not be opened.') from None

    def execute(self, statement, parameters=()):
        try:
            if statement.strip().upper() == 'BEGIN IMMEDIATE':
                # Equivalent serialization for read/check/append review operations.
                return Cursor(self.conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(%s,0))',
                                                ('review:' + self.target.schema,)))
            # All callers use fixed SQL strings; user values remain parameters.
            return Cursor(self.conn.execute(statement.replace('?', '%s'), parameters))
        except Exception:
            raise StorageError('PostgreSQL operation failed; the transaction will roll back. No credentials or database details are shown.') from None

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        try:
            if kind is None:
                self.conn.commit()
            else:
                self.conn.rollback()
        except Exception:
            raise StorageError('PostgreSQL transaction could not be completed; refresh before retrying.') from None
        finally:
            self.conn.close()


def connect(target=None):
    return Connection(target or scope())


def active_scope():
    with connect(scope('control')) as conn:
        conn.execute("INSERT INTO app_state(name,value) VALUES ('active_epoch',?) ON CONFLICT(name) DO NOTHING", (uuid4().hex,))
        epoch = conn.execute("SELECT value FROM app_state WHERE name='active_epoch'").fetchone()[0]
        conn.execute('INSERT INTO demo_epochs VALUES (?,?) ON CONFLICT(epoch) DO NOTHING', (epoch, datetime.now(timezone.utc).isoformat()))
    return scope('demo', epoch)


def epochs():
    with connect(scope('control')) as conn:
        return [scope('demo', r['epoch']) for r in conn.execute('SELECT epoch FROM demo_epochs ORDER BY created_at DESC')]


def rotate_epoch():
    with connect(scope('control')) as conn:
        conn.execute('BEGIN IMMEDIATE')
        old = conn.execute("SELECT value FROM app_state WHERE name='active_epoch'").fetchone()
        epoch = uuid4().hex
        conn.execute('INSERT INTO demo_epochs VALUES (?,?)', (epoch, datetime.now(timezone.utc).isoformat()))
        conn.execute("INSERT INTO app_state VALUES ('active_epoch',?) ON CONFLICT(name) DO UPDATE SET value=excluded.value", (epoch,))
    return dict(archived_epoch=old[0] if old else None, active_epoch=epoch, deleted_files=0)
