"""Append-only machine decisions and explicit human review revisions; never publishing."""
from datetime import datetime,timezone
from enum import Enum
import hashlib,json,sqlite3
from pathlib import Path
from uuid import uuid4
from src.decision.schemas import Decision,Action
from src.rag.guardrails import sanitize_record,validate_response
from src.rag.generate import Draft

ROOT=Path(__file__).resolve().parents[2]
DATABASE=ROOT/'data/review/decisions.sqlite3'
class ReviewStatus(str,Enum):
    PENDING_REVIEW='PENDING_REVIEW';APPROVED='APPROVED';EDITED='EDITED';REJECTED='REJECTED';ESCALATED='ESCALATED';RESOLVED='RESOLVED'


def connect(path=None):
    from src.review import postgres
    if isinstance(path,postgres.Scope) or (path is None and postgres.configured()):
        return postgres.connect(path)
    path=Path(path or DATABASE);path.parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(path,timeout=30);db.row_factory=sqlite3.Row
    db.execute('PRAGMA foreign_keys=ON')
    db.execute('CREATE TABLE IF NOT EXISTS decisions (decision_id TEXT PRIMARY KEY,post_id TEXT NOT NULL,created_at TEXT NOT NULL,decision_json TEXT NOT NULL,pipeline_json TEXT NOT NULL,draft_version TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS review_events (event_id TEXT PRIMARY KEY,decision_id TEXT NOT NULL REFERENCES decisions(decision_id),revision INTEGER NOT NULL,status TEXT NOT NULL,reviewer TEXT NOT NULL,reviewed_at TEXT NOT NULL,notes TEXT NOT NULL,proposed_action TEXT NOT NULL,final_action TEXT NOT NULL,rule_ids TEXT NOT NULL,draft_version TEXT,edited_draft TEXT,validation_json TEXT,UNIQUE(decision_id,revision))')
    db.execute('CREATE INDEX IF NOT EXISTS decision_post ON decisions(post_id)')
    for table in ['decisions','review_events']:
        for verb in ['UPDATE','DELETE']:
            db.execute(f"CREATE TRIGGER IF NOT EXISTS prevent_{table}_{verb} BEFORE {verb} ON {table} BEGIN SELECT RAISE(ABORT,'Audit records are append-only'); END")
    return db


def draft_hash(draft):
    return hashlib.sha256(json.dumps(draft,sort_keys=True,ensure_ascii=False).encode()).hexdigest() if draft else None


def record_decision(result,path=None,*,_connection=None):
    decision=Decision.model_validate(result['decision']);safe=sanitize_record(result)
    draft=(safe.get('generation') or {}).get('draft')
    # Source safety blocks are never made copyable by persisting them.
    if (safe.get('generation') or {}).get('status')=='blocked' and draft:
        draft['draft_response']='[BLOCKED DRAFT]'
    key=uuid4().hex;version=draft_hash(draft)
    def insert(db):
        db.execute('INSERT INTO decisions VALUES (?,?,?,?,?,?)',(key,decision.post_id,datetime.now(timezone.utc).isoformat(),decision.model_dump_json(),json.dumps(safe,ensure_ascii=False),version))
    if _connection is None:
        with connect(path) as db:insert(db)
    else:insert(_connection)
    return key


def unpack_record(row,events):
    result=dict(row);result['decision']=json.loads(result.pop('decision_json'));result['pipeline']=json.loads(result.pop('pipeline_json'));result['events']=events
    result['review_status']=events[-1]['status'] if events else ('PENDING_REVIEW' if result['decision']['requires_human_review'] else None)
    result['revision']=events[-1]['revision'] if events else 0
    return result


def get_decisions(decision_ids,path=None):
    """Fetch a feed's snapshots and events in batches rather than two remote calls per post."""
    ids=list(dict.fromkeys(decision_ids));records={}
    for start in range(0,len(ids),500):
        batch=ids[start:start+500];marks=','.join('?' for _ in batch)
        with connect(path) as db:
            rows=db.execute('SELECT * FROM decisions WHERE decision_id IN ('+marks+')',tuple(batch)).fetchall()
            events=db.execute('SELECT * FROM review_events WHERE decision_id IN ('+marks+') ORDER BY revision',tuple(batch)).fetchall()
        grouped={key:[] for key in batch}
        for event in events:grouped[event['decision_id']].append(dict(event))
        records.update({r['decision_id']:unpack_record(r,grouped[r['decision_id']]) for r in rows})
    return records


def get_decision(decision_id,path=None):
    result=get_decisions([decision_id],path).get(decision_id)
    if result is None:raise ValueError('Unknown stored decision.')
    return result


def review(decision_id,status,reviewer,*,notes='',final_action=None,edited_text=None,edited_support=None,human_confirmed=False,expected_revision=None,path=None):
    if human_confirmed is not True or not isinstance(reviewer,str) or not reviewer.strip():raise ValueError('An identified human must explicitly confirm the review.')
    status=ReviewStatus(status);record=get_decision(decision_id,path);proposal=record['decision']
    action=Action(final_action or proposal['action'])
    if action.value!=proposal['action'] and not notes.strip():raise ValueError('Changing the proposed action requires a rationale.')
    pipeline=record['pipeline'];g=pipeline.get('generation') or {};candidate=g.get('draft')
    latest_edit=next((e for e in reversed(record['events']) if e['edited_draft']),None)
    if latest_edit:candidate=json.loads(latest_edit['edited_draft'])
    if status==ReviewStatus.EDITED:
        if not candidate or not isinstance(edited_text,str) or not edited_text.strip():raise ValueError('An existing draft and nonempty edited text are required.')
        candidate=dict(candidate,draft_response=edited_text.strip())
        if edited_support is not None:candidate['supporting_evidence']=edited_support
    check=None
    if status in {ReviewStatus.APPROVED,ReviewStatus.EDITED}:
        if not candidate or not proposal['requires_response_draft'] or g.get('status')!='review_required' or proposal['action'] in {'URGENT_ESCALATION','ABSTAIN_FOR_HUMAN'}:raise ValueError('This decision has no eligible draft to approve or edit; escalate or resolve it instead.')
        parsed=Draft.model_validate(candidate)
        check=validate_response(parsed,(pipeline.get('retrieval') or {}).get('evidence',[]),(pipeline.get('classification') or {}).get('classification') or {},pipeline.get('customer_post',''))
        from src.rag.generate import response_language_hint
        import re
        requested=response_language_hint(pipeline.get('customer_post',''),(pipeline.get('classification') or {}).get('classification') or {})
        if parsed.response_language!=requested or (requested in {'English','Hinglish'} and re.search(r'[\u0900-\u097f]',parsed.draft_response)):
            check['issues'].append('response_language_or_script_mismatch');check['status']='blocked'
        if status==ReviewStatus.APPROVED and check['status']!='review_required':raise ValueError('Approval blocked by current grounding/safety/language checks: '+', '.join(check['issues']))
    safe_candidate=sanitize_record(candidate)
    if status in {ReviewStatus.APPROVED,ReviewStatus.EDITED} and safe_candidate!=candidate:raise ValueError('Remove private values before recording a draft revision.')
    with connect(path) as db:
        db.execute('BEGIN IMMEDIATE')
        revision=db.execute('SELECT COALESCE(MAX(revision),0) FROM review_events WHERE decision_id=?',(decision_id,)).fetchone()[0]
        if revision!=record['revision'] or (expected_revision is not None and revision!=expected_revision):raise ValueError('Another review was recorded; refresh before submitting.')
        last=db.execute('SELECT * FROM review_events WHERE decision_id=? ORDER BY revision DESC LIMIT 1',(decision_id,)).fetchone()
        if last and (last['status'],last['reviewer'],last['notes'],last['final_action'],last['draft_version'])==(status.value,reviewer.strip(),sanitize_record(notes),action.value,draft_hash(candidate)):
            raise ValueError('This review has already been recorded; no duplicate submission saved.')
        db.execute('INSERT INTO review_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',(uuid4().hex,decision_id,revision+1,status.value,reviewer.strip(),datetime.now(timezone.utc).isoformat(),sanitize_record(notes),proposal['action'],action.value,json.dumps(proposal['rule_ids']),draft_hash(candidate),json.dumps(safe_candidate,ensure_ascii=False) if status==ReviewStatus.EDITED else None,json.dumps(check) if check else None))
    return dict(status=status.value,revision=revision+1,final_action=action.value,publish_allowed=False)


def audit_rows(path=None):
    with connect(path) as db:ids=[r[0] for r in db.execute('SELECT decision_id FROM decisions ORDER BY created_at DESC')]
    rows=[]
    records=get_decisions(ids,path)
    for key in ids:
        r=records[key];d=r['decision'];last=r['events'][-1] if r['events'] else {}
        rows.append(dict(customer_post=r['pipeline'].get('customer_post',''),decision_id=key,post_id=r['post_id'],created_at=r['created_at'],proposed_action=d['action'],final_action=last.get('final_action',d['action']),priority=d['priority'],rule_ids=', '.join(d['rule_ids']),review_status=r['review_status'],reviewer=last.get('reviewer'),draft_version=last.get('draft_version',r['draft_version']),missing_inputs=', '.join(d['missing_inputs']),processing_status=d['processing_status'],rule_version=d['decision_rule_version']))
    return rows
