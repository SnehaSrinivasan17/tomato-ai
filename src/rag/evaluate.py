"""Reviewed/provisional retrieval evaluation with isolated reference revisions."""
from datetime import datetime,timezone
import hashlib,json,time
from pathlib import Path
import pandas as pd
from src.rag.settings import ROOT,RESULTS
from src.rag.ingest import rows,load_chunks

REFERENCES=RESULTS/'evaluation_candidates.jsonl'
REVIEWS=RESULTS/'relevance_reviews.jsonl'


def candidates():
    if REFERENCES.exists():return [json.loads(s) for s in REFERENCES.read_text(encoding='utf8').splitlines()]
    frame=pd.read_json(ROOT/'data/synthetic/food_delivery.jsonl',lines=True)
    existing_ids={r['post_id'] for r in [json.loads(s) for s in (ROOT/'data/llm_evaluation/candidates_balanced_v3.jsonl').read_text(encoding='utf8').splitlines()] if r['source']=='existing_ml_dataset'}
    examples=[]
    for topic,policy in [('late_delivery','CS-102'),('refund','CS-REF-01'),('food_safety','CR-SAFE-01'),('app_payment','CS-PAY-02')]:
        match=frame[(frame.reference_topic==topic)&(frame.reference_sentiment=='negative')&frame.post_key.isin(existing_ids)].iloc[0]
        examples.append(dict(post_id=match.post_key,text=match.text,source='existing_ml_dataset',expected_policy_ids=[policy],human_verified=False))
    scenarios=rows(ROOT/'data/policies/evaluation_scenarios.csv')
    # Include existing privacy/security/multi-risk/injection cases first where available.
    priority=['privacy','account','sarcas','injection']
    selected=[]
    for term in priority:
        matches=[r for r in scenarios if term in (r['category']+' '+r['customer_comment']).lower()]
        if matches and matches[0] not in selected:selected.append(matches[0])
    selected += [r for r in scenarios if r not in selected]
    for r in selected:
        examples.append(dict(post_id='RAG_'+r['test_id'],text=r['customer_comment'],source='existing_policy_scenarios',expected_policy_ids=[r['expected_policy_id']],human_verified=False))
    # Actual Phase 2 multilingual text, never added to the ML dataset.
    from llm.annotation import SUPPLEMENTS
    for name,text,*_ in SUPPLEMENTS:
        if name in ['Hindi_privacy','Hinglish_delivery','Hinglish_safety']:
            examples.append(dict(post_id='LLM_GAP_'+name,text=text,source='phase2_language_gap_case',expected_policy_ids=['CR-PRIV-01' if 'privacy' in name else 'CR-SAFE-01' if 'safety' in name else 'CS-102'],human_verified=False))
    RESULTS.mkdir(exist_ok=True)
    REFERENCES.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in examples)+'\n',encoding='utf8')
    return examples


def approved_references():
    latest={}
    from src.review import postgres
    if postgres.configured():
        with postgres.connect() as db:
            for row in db.execute('SELECT payload FROM reference_reviews ORDER BY rowid'):
                r=json.loads(row[0]);latest[r['post_id']]=r
        return {k:v for k,v in latest.items() if v['human_verified']}
    if REVIEWS.exists():
        for line in REVIEWS.read_text(encoding='utf8').splitlines():
            r=json.loads(line);latest[r['post_id']]=r
    return {k:v for k,v in latest.items() if v['human_verified']}


def save_review(post_id,policy_ids,reviewer,approved,notes=''):
    if not reviewer.strip():raise ValueError('Reviewer identifier required.')
    if post_id not in {r['post_id'] for r in candidates()}:raise ValueError('Unknown evaluation ID.')
    valid={r['policy_id'] for r in load_chunks() if r['kind']=='policy'}
    if not policy_ids or not set(policy_ids).issubset(valid):raise ValueError('Reference mappings must use existing policy IDs.')
    record=dict(post_id=post_id,expected_policy_ids=policy_ids,reviewer=reviewer.strip(),human_verified=bool(approved),notes=notes,reviewed_at=datetime.now(timezone.utc).isoformat())
    from src.review import postgres
    if postgres.configured():
        from uuid import uuid4
        with postgres.connect() as db:
            db.execute('INSERT INTO reference_reviews(event_id,post_id,reviewed_at,payload) VALUES (?,?,?,?)',
                       (uuid4().hex,post_id,record['reviewed_at'],json.dumps(record,ensure_ascii=False)))
        return
    with REVIEWS.open('a',encoding='utf8') as file:file.write(json.dumps(record,ensure_ascii=False)+'\n')


def retrieval_metrics(records,references):
    eligible=[r for r in records if r['post_id'] in references and references[r['post_id']]]
    if not eligible:return None
    recall3=[];recall5=[];precision5=[];rr=[];latency=[];failures=0
    for row in eligible:
        ref=set(references[row['post_id']]);retrieval=row.get('retrieval') or {};ids=[e['policy_id'] for e in retrieval.get('evidence',[])];ids=list(dict.fromkeys(ids))
        recall3.append(len(set(ids[:3])&ref)/len(ref));recall5.append(len(set(ids[:5])&ref)/len(ref));precision5.append(len(set(ids[:5])&ref)/5)
        rr.append(next((1/(i+1) for i,p in enumerate(ids) if p in ref),0))
        latency.append(float(row.get('latencies_ms',{}).get('retrieval',0)))
        failures+=retrieval.get('status')=='error' or not retrieval
    mean=lambda x:sum(x)/len(x)
    return dict(n=len(eligible),recall_at_3=mean(recall3),recall_at_5=mean(recall5),precision_at_5=mean(precision5),mrr=mean(rr),mean_retrieval_latency_ms=mean(latency),retrieval_failure_rate=failures/len(eligible))


def evaluate(limit=8,output=None,evaluation_cases=None,retry_transport=False):
    from src.workflow.pipeline import process_post,process_custom_post
    if evaluation_cases is None:
        all_cases=candidates()
        ordered=all_cases[:4]+[r for r in all_cases if r['source']=='phase2_language_gap_case']+[r for r in all_cases[4:] if r['source']!='phase2_language_gap_case']
    else:
        ordered=evaluation_cases
    cases=ordered[:limit] if limit else ordered
    output=Path(output) if output else RESULTS/'evaluations'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True,exist_ok=True);checkpoint=output/'pipeline.jsonl'
    from src.rag.settings import generation_model,collection,EMBEDDING_MODEL
    from llm.prompts import load_prompt
    from src.rag.generate import PROMPT_PATH,PROMPT_VERSION
    prompt_path=PROMPT_PATH
    from config import get_setting
    from src.decision.rules import VERSION as DECISION_VERSION
    provenance=dict(decision_rule_version=DECISION_VERSION,decision_engine_sha256=hashlib.sha256((ROOT/'src/decision/engine.py').read_bytes()+(ROOT/'src/decision/rules.py').read_bytes()).hexdigest(),max_completion_tokens=max(256,min(int(get_setting('GROQ_DRAFT_MAX_TOKENS','1600')),4096)),reasoning_effort='low' if generation_model().startswith('openai/gpt-oss') else None,generation_prompt_version=PROMPT_VERSION,model=generation_model(),collection=collection(),embedding_model=EMBEDDING_MODEL,
        classification_prompt_sha256=load_prompt()[1],generation_prompt_sha256=hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
        evaluation_post_ids=[r['post_id'] for r in cases],provisional_reference_sha256=hashlib.sha256(json.dumps(cases,sort_keys=True,ensure_ascii=False).encode()).hexdigest())
    manifest=output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf8'))!=provenance:
        raise ValueError('Evaluation resume configuration changed; choose a new output directory.')
    manifest.write_text(json.dumps(provenance,indent=2),encoding='utf8')
    completed={}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding='utf8').splitlines():
            r=json.loads(line);completed[r['post_id']]=r
    for case in cases:
        prior=completed.get(case['post_id'])
        if prior:
            errors=[(prior.get(k) or {}).get('error_code') for k in ['classification','generation']]
            if not (retry_transport and any(e in {'rate_limited','api_error','missing_key'} for e in errors)):continue
        r=process_post(case['post_id']) if case['source']=='existing_ml_dataset' else process_custom_post(case['text'],post_id=case['post_id'])
        # Persist a privacy-safe result; blocked drafts are never exposed as copyable content.
        if (r.get('generation') or {}).get('status')=='blocked' and r['generation'].get('draft'):
            r['generation']['draft']['draft_response']='[BLOCKED DRAFT]'
        from src.rag.guardrails import sanitize_record
        r=sanitize_record(r)
        with checkpoint.open('a',encoding='utf8') as file:file.write(json.dumps(r,ensure_ascii=False)+'\n')
        completed[r['post_id']]=r;print(case['post_id'],r['status'],flush=True)
        time.sleep(2)
    records=[completed[r['post_id']] for r in cases if r['post_id'] in completed]
    refs={r['post_id']:r['expected_policy_ids'] for r in cases};human=approved_references()
    generated=[r['generation'] for r in records if (r.get('generation') or {}).get('draft')]
    validated=[g.get('validation') for g in generated if g.get('validation')]
    report=dict(metadata=provenance,dataset='existing_policy_scenarios_plus_existing_ML_posts_and_Phase2_language_cases',provisional_retrieval=retrieval_metrics(records,refs),human_verified_retrieval=retrieval_metrics(records,{k:v['expected_policy_ids'] for k,v in human.items()}),reference_status='provisional unless explicitly reviewed',attempted_posts=len(records),generation=dict(valid_schema_outputs=len(generated),attempted_posts=len(records),blocked=sum(r['status']=='blocked' for r in records),failed=sum(r['status'] in {'error','partial_failure'} for r in records),citation_checks_passed=sum(v['citation_valid'] for v in validated),privacy_checks_passed=sum(v['privacy_pass'] for v in validated),human_approval_required=len(records),publish_allowed=0),subjective_quality='Pending human rubric review; no fabricated tone/language/entailment scores.')
    from collections import Counter
    issues=Counter(issue for v in validated for issue in v['issues'])
    report['validation_issue_counts']=dict(issues)
    report['record_attempt_count']=len(checkpoint.read_text(encoding='utf8').splitlines())
    report['attempt_semantics']='Metrics use latest record per ID; all prior attempts remain in the append-only checkpoint.'
    report['essential_safety_escalations']=sum(bool(r.get('essential_safety',{}).get('human_escalation_required')) for r in records)
    report['policy_conflict_blocks']=sum((r.get('retrieval') or {}).get('status')=='conflict' for r in records)
    (output/'reference_snapshot.json').write_text(json.dumps(dict(provisional=refs,human_verified=human),indent=2,ensure_ascii=False),encoding='utf8')
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    (RESULTS/'latest_evaluation.json').write_text(json.dumps({'directory':str(output.resolve())}),encoding='utf8')
    return report,output
