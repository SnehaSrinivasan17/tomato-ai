"""Version-aware real workflow caching and append-only demo epoch resets."""
import hashlib,json,sqlite3,time
from pathlib import Path
from uuid import uuid4
from datetime import datetime,timezone
from pydantic import BaseModel,ConfigDict
from src.demo.manifest import ROOT,DIRECTORY,sha
from src.review.store import record_decision,get_decision,get_decisions

STATE=ROOT/'data/demo_state'

class DemoRun(BaseModel):
    model_config=ConfigDict(extra='forbid')
    post_id:str
    decision_id:str
    cache_key:str
    version_fingerprint:str
    processed_at:str
    elapsed_ms:float
    complete:bool

def versions():
    from ml.bundle import active_model_directory
    from llm.config import classifier_model
    from llm.prompts import load_prompt,PROMPT_VERSION
    from src.rag.settings import collection,generation_model,EMBEDDING_MODEL
    from src.rag.generate import PROMPT_PATH,PROMPT_VERSION as GP
    from src.decision.engine import RULE_FINGERPRINT
    from src.decision.rules import VERSION
    from config import get_setting
    files=['rag_results/ingestion.json','rag_results/rag_chunks.jsonl','src/workflow/pipeline.py','src/rag/guardrails.py','src/rag/retrieve.py','src/rag/generate.py','llm/classifier.py','ml/inference.py','src/demo/runner.py','src/demo/manifest.py']
    model=active_model_directory()
    return dict(classifier_model=classifier_model(),classifier_prompt=PROMPT_VERSION,classifier_prompt_sha256=load_prompt()[1],generation_model=generation_model(),generation_prompt=GP,generation_prompt_sha256=sha(PROMPT_PATH),decision_version=VERSION,decision_fingerprint=RULE_FINGERPRINT,ml_version=model.name,ml_manifest=sha(model/'bundle_manifest.json'),policy_collection=collection(),policy_connection_fingerprint=fingerprint(get_setting('QDRANT_URL')),embedding_model=EMBEDDING_MODEL,embedding_revision=get_setting('BGE_MODEL_REVISION','main'),files={f:sha(ROOT/f) if (ROOT/f).exists() else None for f in files})

def fingerprint(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()

def active_path(root=STATE):
    from src.review import postgres
    if root==STATE and postgres.configured():return postgres.active_scope()
    root=Path(root);root.mkdir(parents=True,exist_ok=True);pointer=root/'active.json'
    if not pointer.exists():
        # Exclusive creation: concurrent first visits must share the same epoch.
        try:
            with pointer.open('x',encoding='utf8') as f:json.dump(dict(epoch=uuid4().hex),f)
        except FileExistsError:pass
    epoch=json.loads(pointer.read_text())['epoch']
    if not isinstance(epoch,str) or len(epoch)!=32 or any(c not in '0123456789abcdef' for c in epoch):raise ValueError('Invalid demo epoch.')
    path=root/epoch/'review.sqlite3';path.parent.mkdir(exist_ok=True);return path

def reset(root=STATE):
    from src.review import postgres
    if root==STATE and postgres.configured():return postgres.rotate_epoch()
    old=active_path(root);pointer=Path(root)/'active.json';temp=pointer.with_name(uuid4().hex+'.tmp')
    temp.write_text(json.dumps(dict(epoch=uuid4().hex)),encoding='utf8');temp.replace(pointer)
    return dict(archived_epoch=old.parent.name,active_epoch=active_path(root).parent.name,deleted_files=0)

def db(path):
    from src.review.store import connect
    from src.review import postgres
    conn=connect(path)
    if isinstance(conn,postgres.Connection):return conn
    conn.execute('CREATE TABLE IF NOT EXISTS demo_runs (decision_id TEXT PRIMARY KEY,post_id TEXT NOT NULL,cache_key TEXT NOT NULL,version_fingerprint TEXT NOT NULL,processed_at TEXT NOT NULL,elapsed_ms REAL NOT NULL,complete INTEGER NOT NULL)')
    conn.execute('CREATE TABLE IF NOT EXISTS demo_cache_hits (id INTEGER PRIMARY KEY,decision_id TEXT NOT NULL,at TEXT NOT NULL)')
    return conn

def history(path=None):
    with db(path or active_path()) as conn:return [dict(r) for r in conn.execute('SELECT * FROM demo_runs ORDER BY processed_at DESC,rowid DESC')]

def analyze(post,*,force=False,path=None,workflow=None,version_info=None):
    from src.workflow.pipeline import process_custom_post,predictor
    path=path or active_path();v=version_info if version_info is not None else versions();vf=fingerprint(v)
    # Whitelist existing measured features; excludes labels, final outcomes and display dates.
    raw=post.get('metadata') or {};features=predictor().schema['features']
    metadata={k:raw[k] for k in features if k in raw}
    key=fingerprint(dict(post_id=post['post_id'],text=post['text'],metadata=metadata,versions=v))
    if not force:
        with db(path) as conn:
            row=conn.execute('SELECT * FROM demo_runs WHERE cache_key=? AND complete=1 ORDER BY rowid DESC LIMIT 1',(key,)).fetchone()
        if row:
            result=get_decision(row['decision_id'],path)
            with db(path) as conn:conn.execute('INSERT INTO demo_cache_hits(decision_id,at) VALUES (?,?)',(row['decision_id'],datetime.now(timezone.utc).isoformat()))
            return dict(result['pipeline'],decision_id=row['decision_id'],review_status=result['review_status'],pipeline_cache_hit=True,version_info=v)
    tick=time.perf_counter();kwargs=dict(post_id=post['post_id'],persist=False)
    if force and workflow is None:
        from llm.classifier import classify_post
        kwargs['classifier']=lambda text,post_id:classify_post(text,post_id,use_cache=False)
    result=(workflow or process_custom_post)(post['text'],metadata,**kwargs)
    result['pipeline_versions']=v;result['input_metadata']=metadata
    c=result.get('classification') or {};ret=result.get('retrieval') or {};g=result.get('generation') or {}
    complete=c.get('status')=='success' and not result.get('errors') and ret.get('status') not in {'error','insufficient_evidence','conflict'} and g.get('status') not in {'error','blocked'}
    with db(path) as conn:
        # Snapshot and cache index commit together, preventing orphaned review
        # records if a database error interrupts the pipeline's final save.
        decision_id=record_decision(result,path,_connection=conn)
        run=DemoRun(post_id=post['post_id'],decision_id=decision_id,cache_key=key,version_fingerprint=vf,processed_at=datetime.now(timezone.utc).isoformat(),elapsed_ms=(time.perf_counter()-tick)*1000,complete=complete)
        conn.execute('INSERT INTO demo_runs(decision_id,post_id,cache_key,version_fingerprint,processed_at,elapsed_ms,complete) VALUES (?,?,?,?,?,?,?)',(run.decision_id,run.post_id,run.cache_key,run.version_fingerprint,run.processed_at,run.elapsed_ms,int(run.complete)))
    saved=get_decision(decision_id,path)
    return dict(saved['pipeline'],decision_id=decision_id,review_status=saved['review_status'],pipeline_cache_hit=False,version_info=v)

def feed(posts,path=None):
    path=path or active_path();latest={}
    for r in history(path):latest.setdefault(r['post_id'],r)
    records=get_decisions([latest[p['post_id']]['decision_id'] for p in posts if p['post_id'] in latest],path)
    rows=[]
    for post in posts:
        r=latest.get(post['post_id']);record=records[r['decision_id']] if r else None
        p=record['pipeline'] if record else {};c=(p.get('classification') or {}).get('classification') or {};d=p.get('decision') or {};ml=p.get('ml') or {}
        rows.append(dict(post_id=post['post_id'],text=post['text'],source=post.get('source','custom'),date=post.get('presentation_timestamp'),original_timestamp=post.get('original_timestamp'),language=c.get('language') or post.get('language_hint','Unknown'),language_source='model' if c else 'source hint; no classification available',sentiment=c.get('sentiment'),topic=c.get('topic'),urgency=c.get('urgency'),spread_probability=ml.get('probability'),high_spread=ml.get('high_spread'),action=d.get('action'),priority=d.get('priority'),review_status=record['review_status'] if record else None,processing_status=p.get('status','unprocessed'),complete=bool(r['complete']) if r else None,decision_id=r['decision_id'] if r else None,processed_at=r['processed_at'] if r else None,latency_ms=r['elapsed_ms'] if r else None))
    return rows

def all_feed(posts,path=None):
    path=path or active_path();known={r['post_id'] for r in posts};custom=[]
    unknown=[]
    for r in history(path):
        if r['post_id'] in known:continue
        known.add(r['post_id']);unknown.append(r)
    records=get_decisions([r['decision_id'] for r in unknown],path)
    for r in unknown:
        p=records[r['decision_id']]['pipeline']
        custom.append(dict(post_id=r['post_id'],text=p['customer_post'],source='custom submission',language_hint='Unknown',original_timestamp=None,presentation_timestamp=None))
    return feed(posts+custom,path)

def analytics(rows,path=None):
    from collections import Counter
    processed=[r for r in rows if r['decision_id']]
    path=path or active_path()
    with db(path) as conn:hits=conn.execute('SELECT count(*) FROM demo_cache_hits').fetchone()[0]
    return dict(available=len(rows),analyzed=len(processed),unprocessed=len(rows)-len(processed),failed_or_incomplete=sum(not r['complete'] for r in processed),cache_hits=hits,ml_available=sum(r['spread_probability'] is not None for r in processed),high_spread=sum(r['high_spread'] is True for r in processed),safety_escalations=sum(r['action']=='URGENT_ESCALATION' for r in processed),pending_review=sum(r['review_status']=='PENDING_REVIEW' for r in processed),review_queue=sum(r['review_status'] in {'PENDING_REVIEW','EDITED','ESCALATED'} for r in processed),approved=sum(r['review_status']=='APPROVED' for r in processed),rejected=sum(r['review_status']=='REJECTED' for r in processed),edited=sum(r['review_status']=='EDITED' for r in processed),mean_latency_ms=sum(r['latency_ms'] for r in processed)/len(processed) if processed else None,sentiment=dict(Counter(r['sentiment'] for r in processed if r['sentiment'])),topic=dict(Counter(r['topic'] for r in processed if r['topic'])),urgency=dict(Counter(r['urgency'] for r in processed if r['urgency'])))
