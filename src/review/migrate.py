"""Idempotently copy local audit history to PostgreSQL without rewriting it."""
from datetime import datetime,timezone
import hashlib,json,sqlite3
from pathlib import Path
from src.review import postgres

ROOT=Path(__file__).resolve().parents[2]
TABLES={
    'decisions': ('decision_id','post_id','created_at','decision_json','pipeline_json','draft_version'),
    'review_events': ('event_id','decision_id','revision','status','reviewer','reviewed_at','notes','proposed_action','final_action','rule_ids','draft_version','edited_draft','validation_json'),
    'demo_runs': ('decision_id','post_id','cache_key','version_fingerprint','processed_at','elapsed_ms','complete'),
    'demo_cache_hits': ('id','decision_id','at'),
    'annotations': ('post_id','reviewer','version','reviewed_at','human_verified','labels','notes'),
}
KEYS={name: (columns[0],) for name,columns in TABLES.items()}
KEYS['annotations']=('post_id','reviewer','version')


def copy_table(source,target,table):
    columns=TABLES[table];keys=KEYS[table]
    if source.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",(table,)).fetchone() is None:
        return dict(inserted=0,already_present=0)
    counts=dict(inserted=0,already_present=0)
    ordering='decision_id,revision' if table=='review_events' else 'rowid'
    for row in source.execute(f"SELECT {','.join(columns)} FROM {table} ORDER BY {ordering}"):
        original=dict(row)
        existing=target.execute(f"SELECT {','.join(columns)} FROM {table} WHERE "+' AND '.join(k+'=?' for k in keys),tuple(original[k] for k in keys)).fetchone()
        if existing is not None:
            if dict(existing)!=original:
                raise ValueError('Migration conflict in '+table+'; no historical row was overwritten.')
            counts['already_present']+=1
        else:
            target.execute(f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",tuple(original[k] for k in columns))
            counts['inserted']+=1
    return counts


def import_database(path,target,tables):
    path=Path(path)
    if not path.exists():return {}
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as source:
        source.row_factory=sqlite3.Row
        with postgres.connect(target) as destination:
            destination.execute('BEGIN IMMEDIATE')
            if 'demo_cache_hits' in tables:
                destination.execute('LOCK TABLE demo_cache_hits IN SHARE ROW EXCLUSIVE MODE')
            result={name:copy_table(source,destination,name) for name in tables}
            if 'demo_cache_hits' in tables:
                destination.execute("SELECT setval(pg_get_serial_sequence('demo_cache_hits','id'),COALESCE(MAX(id),1),MAX(id) IS NOT NULL) FROM demo_cache_hits")
    if hashlib.sha256(path.read_bytes()).hexdigest()!=before:
        raise ValueError('Local database changed during migration; rerun to reconcile the new records.')
    return result


def migrate(root=ROOT):
    root=Path(root)
    if not postgres.configured():raise ValueError('Configure REVIEW_DATABASE_URL before migration.')
    output=dict(global_history=import_database(root/'data/review/decisions.sqlite3',postgres.scope(),['decisions','review_events']),demo_epochs={},annotations={},policy_reference_reviews={})
    for path in sorted((root/'data/demo_state').glob('*/review.sqlite3')):
        epoch=path.parent.name;target=postgres.scope('demo',epoch)
        output['demo_epochs'][epoch]=import_database(path,target,['decisions','review_events','demo_runs','demo_cache_hits'])
        with postgres.connect(postgres.scope('control')) as db:
            db.execute('INSERT INTO demo_epochs VALUES (?,?) ON CONFLICT(epoch) DO NOTHING',(epoch,datetime.now(timezone.utc).isoformat()))
    output['annotations']=import_database(root/'data/llm_evaluation/human_annotations.sqlite3',postgres.scope(),['annotations'])
    references=root/'rag_results/relevance_reviews.jsonl';inserted=present=0
    if references.exists():
        with postgres.connect() as db:
            for line in references.read_text(encoding='utf8').splitlines():
                if not line.strip():continue
                row=json.loads(line);payload=json.dumps(row,ensure_ascii=False,sort_keys=True);key=hashlib.sha256(payload.encode()).hexdigest()
                existing=db.execute('SELECT payload FROM reference_reviews WHERE event_id=?',(key,)).fetchone()
                if existing:
                    if existing[0]!=payload:raise ValueError('Reference migration conflict; no rows overwritten.')
                    present+=1
                else:
                    db.execute('INSERT INTO reference_reviews(event_id,post_id,reviewed_at,payload) VALUES (?,?,?,?)',(key,row['post_id'],row['reviewed_at'],payload));inserted+=1
    output['policy_reference_reviews']=dict(inserted=inserted,already_present=present)
    pointer=root/'data/demo_state/active.json'
    if pointer.exists():
        local_epoch=json.loads(pointer.read_text())['epoch'];target=postgres.scope('demo',local_epoch)
        with postgres.connect(postgres.scope('control')) as db:
            db.execute('BEGIN IMMEDIATE')
            done=db.execute("SELECT value FROM app_state WHERE name='sqlite_import_complete'").fetchone()
            if not done:
                active=db.execute("SELECT value FROM app_state WHERE name='active_epoch'").fetchone()
                empty=not active
                if active:
                    with postgres.connect(postgres.scope('demo',active[0])) as current:
                        empty=current.execute('SELECT count(*) FROM decisions').fetchone()[0]==0
                # Adopt the existing local session on initial import only. A
                # later migration must never undo a user-initiated remote reset.
                if empty:
                    db.execute("INSERT INTO app_state VALUES ('active_epoch',?) ON CONFLICT(name) DO UPDATE SET value=excluded.value",(target.epoch,))
                db.execute("INSERT INTO app_state VALUES ('sqlite_import_complete','yes') ON CONFLICT(name) DO NOTHING")
    output['local_files_deleted']=0
    return output
