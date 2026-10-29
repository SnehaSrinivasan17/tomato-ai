"""One decision engine orchestrates the existing classifier, ML, retrieval and draft validators."""
from datetime import datetime,timezone
from functools import lru_cache
import json,time
from uuid import uuid4
import pandas as pd
from llm.classifier import classify_post
from llm.combined import validate_metadata
from llm.storage import redact
from src.rag.retrieve import retrieve_policies
from src.rag.generate import generate_response
from src.rag.guardrails import risk_flags,sanitize_record
from src.rag.settings import ROOT,RESULTS
from src.decision.engine import decide
from src.review.store import record_decision

@lru_cache(maxsize=1)
def dataset():
    return pd.read_json(ROOT/'data/synthetic/food_delivery.jsonl',lines=True).set_index('post_key')

@lru_cache(maxsize=1)
def predictor():
    from ml.inference import SpreadPredictor
    return SpreadPredictor()


def process_custom_post(post_text,metadata=None,post_id=None,*,classifier=None,model=None,retriever=None,generator=None,persist=True):
    start=time.perf_counter();post_id=post_id or 'RAG_LIVE_'+uuid4().hex
    result=dict(post_id=post_id,customer_post=redact(post_text) if isinstance(post_text,str) else '',classification=None,ml=None,retrieval=None,generation=None,errors=[],latencies_ms={},requires_human_review=True,publish_allowed=False,timestamp=datetime.now(timezone.utc).isoformat())
    tick=time.perf_counter()
    try:
        c=(classifier or classify_post)(post_text,post_id)
        result['classification']=c.model_dump(mode='json')
    except Exception:result['errors'].append('classification_error')
    result['latencies_ms']['classification']=(time.perf_counter()-tick)*1000
    tick=time.perf_counter()
    try:
        ml=model or predictor();values,errors=validate_metadata(metadata or {},ml.schema)
        result['metadata_errors']=errors
        if not errors: result['ml']=ml.predict(values)
    except Exception:result['errors'].append('ml_error')
    result['latencies_ms']['ml']=(time.perf_counter()-tick)*1000
    c=result['classification']
    flags=risk_flags(post_text if isinstance(post_text,str) else '',c['classification'] if c and c['status']=='success' else {})
    result['essential_safety']=dict(risk_flags=flags,human_escalation_required=bool(flags),human_approval_required=True,publish_allowed=False)
    tick=time.perf_counter()
    initial=decide(post_id,c,result['ml'],post_text=post_text)
    history=[initial.model_dump(mode='json')]
    decision=initial
    if initial.requires_policy_retrieval:
        retrieval_start=time.perf_counter()
        try:result['retrieval']=(retriever or retrieve_policies)(post_text,c['classification'] if c and c['status']=='success' else None)
        except Exception:result['retrieval']=dict(status='error',evidence=[],reasons=['retrieval_error'])
        result['latencies_ms']['retrieval']=(time.perf_counter()-retrieval_start)*1000
        decision=decide(post_id,c,result['ml'],post_text=post_text,retrieval=result['retrieval'],stage='post_retrieval')
        history.append(decision.model_dump(mode='json'))
        if decision.requires_response_draft:
            generation_start=time.perf_counter()
            try:result['generation']=(generator or generate_response)(post_text,c['classification'],result['retrieval'],spread=result['ml'])
            except Exception:result['generation']=dict(status='error',draft=None,error_code='generation_error',publish_allowed=False)
            result['latencies_ms']['generation']=(time.perf_counter()-generation_start)*1000
            decision=decide(post_id,c,result['ml'],post_text=post_text,retrieval=result['retrieval'],generation=result['generation'],stage='final')
            history.append(decision.model_dump(mode='json'))
        else:
            result['generation']=dict(status='skipped',draft=None,error_code='decision_does_not_allow_drafting',publish_allowed=False)
    elif initial.action.value=='MONITOR':
        result['generation']=dict(status='skipped',draft=None,error_code='monitor_no_response_required',publish_allowed=False)
    if not c or c['status']!='success':result['errors'].append('classification_unavailable')
    result['decision']=decision.model_dump(mode='json');result['decision_history']=history
    result['requires_human_review']=decision.requires_human_review
    result['review_status']='PENDING_REVIEW' if decision.requires_human_review else None
    result['latencies_ms']['decision_workflow']=(time.perf_counter()-tick)*1000
    g=result['generation'] or {}
    result['status']=g['status'] if g.get('status') in {'error','blocked','review_required'} else ('monitoring' if initial.action.value=='MONITOR' else 'abstained' if decision.action.value=='ABSTAIN_FOR_HUMAN' else 'escalation_required')
    if 'F_POLICY' in decision.rule_ids:result['status']='blocked'
    result['latencies_ms']['total']=(time.perf_counter()-start)*1000
    if persist:
        RESULTS.mkdir(exist_ok=True)
        # Do not persist blocked output that may contain private or unsafe content.
        result['decision_id']=record_decision(result)
        saved=sanitize_record(result)
        if g.get('status')=='blocked' and g.get('draft'):
            saved['generation']['draft']['draft_response']='[BLOCKED: inspect validation issues]'
        with (RESULTS/'pipeline_runs.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(saved,ensure_ascii=False)+'\n')
    return result


def process_post(post_id,**kwargs):
    try:row=dataset().loc[post_id].to_dict()
    except KeyError:return dict(post_id=post_id,status='error',errors=['unknown_post_id'],publish_allowed=False,requires_human_review=True)
    # Exactly the existing saved feature schema; no final outcomes/reference labels.
    schema=predictor().schema
    metadata={k:row[k] for k in schema['features'] if k in row and pd.notna(row[k])}
    return process_custom_post(row['text'],metadata,post_id,**kwargs)
