"""Offline PostgreSQL adapter tests; no real credentials or network requests."""
from unittest.mock import Mock,patch
import hashlib,sqlite3
import pytest
from src.review import postgres
from src.review.migrate import copy_table


@pytest.mark.parametrize('target',[
    dict(kind='demo',epoch='../unsafe'),dict(namespace='unsafe;drop'),
    dict(kind='unknown'),dict(kind='global',epoch='a'*32),dict(namespace='a'*17),
])
def test_scope_rejects_untrusted_identifiers(target):
    with pytest.raises(ValueError):postgres.Scope(**target)


def test_scope_and_rows_do_not_include_credentials():
    target=postgres.Scope('demo','a'*32)
    assert target.schema=='ta_tomato_demo_'+'a'*32
    assert 'postgresql://' not in repr(target)
    row=postgres.Row(first='value',second=2)
    assert row[0]==row['first']=='value' and row[1]==2 and dict(row)==dict(first='value',second=2)


def test_configured_remote_failure_never_falls_back_to_sqlite(monkeypatch):
    from src.review.store import connect
    monkeypatch.setattr(postgres,'configured',lambda:True)
    with patch.object(postgres,'connect',side_effect=postgres.StorageError('Unavailable')),patch('sqlite3.connect') as local:
        with pytest.raises(postgres.StorageError):connect()
        assert not local.called


def test_private_connection_failure_is_sanitized(monkeypatch):
    password='fixture-private-value'
    uri=f'postgresql://fixture:{password}@example.invalid/db?sslmode=require'
    monkeypatch.setattr(postgres,'get_setting',lambda key,default='':uri)
    with patch('psycopg.connect',side_effect=RuntimeError(uri)):
        with pytest.raises(postgres.StorageError) as error:postgres._open()
    assert password not in str(error.value) and uri not in str(error.value)


def test_tls_is_required_and_psql_commands_are_rejected(monkeypatch):
    monkeypatch.setattr(postgres,'get_setting',lambda *a:'postgresql://fixture@example.invalid/db?sslmode=disable')
    with patch('psycopg.connect') as connect:
        with pytest.raises(postgres.StorageError,match='TLS'):postgres._open()
        assert not connect.called
    monkeypatch.setattr(postgres,'get_setting',lambda *a:"psql 'private-uri'")
    with pytest.raises(postgres.StorageError,match='connection URI'):postgres._dsn()


def test_review_lock_and_connection_transaction_lifecycle():
    target=postgres.Scope();connection=object.__new__(postgres.Connection)
    connection.target=target;connection.conn=Mock()
    connection.execute('BEGIN IMMEDIATE')
    assert 'pg_advisory_xact_lock' in connection.conn.execute.call_args.args[0]
    assert connection.conn.execute.call_args.args[1]==('review:'+target.schema,)
    connection.__exit__(None,None,None)
    connection.conn.commit.assert_called_once();connection.conn.close.assert_called_once()
    connection.conn=Mock();connection.__exit__(ValueError,ValueError('fixture'),None)
    connection.conn.rollback.assert_called_once();connection.conn.close.assert_called_once()


def test_copy_is_idempotent_and_rejects_conflicts(tmp_path):
    source=sqlite3.connect(tmp_path/'source.db');source.row_factory=sqlite3.Row
    target=sqlite3.connect(tmp_path/'target.db');target.row_factory=sqlite3.Row
    for db in [source,target]:db.execute('CREATE TABLE decisions (decision_id TEXT PRIMARY KEY,post_id TEXT,created_at TEXT,decision_json TEXT,pipeline_json TEXT,draft_version TEXT)')
    original=('stable-id','post','timestamp','{}','{}',None)
    source.execute('INSERT INTO decisions VALUES (?,?,?,?,?,?)',original)
    assert copy_table(source,target,'decisions')==dict(inserted=1,already_present=0)
    assert copy_table(source,target,'decisions')==dict(inserted=0,already_present=1)
    source.execute("UPDATE decisions SET pipeline_json='changed'")
    with pytest.raises(ValueError,match='no historical row'):copy_table(source,target,'decisions')
    assert tuple(target.execute('SELECT * FROM decisions').fetchone())==original
    source.close();target.close()


def test_explicit_sqlite_remains_available_for_offline_fixtures(tmp_path):
    from src.review.store import connect
    with connect(tmp_path/'review.db') as db:
        assert db.execute('SELECT count(*) FROM decisions').fetchone()[0]==0


def test_snapshot_and_cache_save_roll_back_together(tmp_path):
    from src.demo.runner import db,analyze,history
    from tests.test_demo_phase5 import callback
    path=tmp_path/'review.db'
    with db(path) as connection:
        connection.execute("CREATE TRIGGER reject_cache BEFORE INSERT ON demo_runs BEGIN SELECT RAISE(ABORT,'fixture interruption'); END")
    with pytest.raises(sqlite3.IntegrityError):
        analyze(dict(post_id='atomic-test',text='Fixture text.'),path=path,workflow=callback(),version_info={})
    with db(path) as connection:assert connection.execute('SELECT count(*) FROM decisions').fetchone()[0]==0
    assert history(path)==[]
