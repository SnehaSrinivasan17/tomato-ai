"""Versioned, checkpointed canonical evaluation, separate from legacy benchmarks."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,time
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix
from llm.annotation import build_candidates,agreed_references,DIRECTORY
from llm.classifier import classify_post
from llm.config import classifier_model
from llm.prompts import load_prompt,PROMPT_VERSION
from llm.schemas import SafetyFlag
from llm.storage import redact

RUNS=DIRECTORY/'runs'

def summarize(rows,predictions,references):
    by_id={p['post_id']:p for p in predictions}
    requested=list(rows)
    rows=[r for r in requested if r['post_id'] in by_id]
    predictions=[by_id[r['post_id']] for r in rows]
    references={r['post_id']:references[r['post_id']] for r in rows if r['post_id'] in references}
    output={'sample_count':len(rows),'reference_count':len(references),'fields':{},'errors':[], 'by_language':[], 'safety':[], 'latencies_ms':[]}
    output['planned_sample_count']=len(requested)
    output['pending_count']=len(requested)-len(rows)
    valid=sum(by_id.get(r['post_id'],{}).get('status')=='success' for r in rows)
    output['completed_count']=sum(r['post_id'] in by_id for r in rows)
    output['error_counts']=dict(Counter(p.get('error_code') or 'success' for p in predictions))
    output['json_validity_rate']=valid/len(rows) if rows else None
    output['api_failure_rate']=sum(by_id.get(r['post_id'],{}).get('error_code') in {'api_error','rate_limited','missing_key','client_error'} for r in rows)/len(rows) if rows else None
    latency=[float(p['latency_ms']) for p in predictions if not p.get('cache_hit')]
    output.update(latencies_ms=latency,average_latency_ms=float(np.mean(latency)) if latency else None,p95_latency_ms=float(np.percentile(latency,95)) if latency else None)
    fields=['sentiment','topic','urgency','suggested_severity','language','sarcasm_detected','response_recommended','ambiguous']
    for field in fields:
        actual=[];predicted=[]
        for row in rows:
            ref=references.get(row['post_id'],{})
            if field not in ref: continue
            p=by_id.get(row['post_id'],{})
            guess=(p.get('classification') or {}).get(field,'INVALID')
            a,b=str(ref[field]),str(guess)
            actual.append(a);predicted.append(b)
            if a!=b:
                output['errors'].append(dict(post_id=row['post_id'],text=redact(row['text']),field=field,expected=a,predicted=b,language=ref.get('language','unlabelled'),topic=ref.get('topic','unlabelled'),urgency=ref.get('urgency','unlabelled'),safety_critical=bool(ref.get('safety_flags')),error_code=p.get('error_code')))
        if not actual: continue
        labels=sorted(set(actual));axes=sorted(set(actual+predicted))
        precision,recall,f1,support=precision_recall_fscore_support(actual,predicted,labels=labels,zero_division=0)
        _,_,weighted,_=precision_recall_fscore_support(actual,predicted,labels=labels,average='weighted',zero_division=0)
        output['fields'][field]=dict(n=len(actual),accuracy=accuracy_score(actual,predicted),macro_f1=float(np.mean(f1)),weighted_f1=float(weighted),classes=[dict(label=k,precision=float(p),recall=float(r),f1=float(f),support=int(n)) for k,p,r,f,n in zip(labels,precision,recall,f1,support)],matrix=dict(labels=axes,counts=confusion_matrix(actual,predicted,labels=axes).tolist()))
    safety_rows=[r for r in rows if 'safety_flags' in references.get(r['post_id'],{})]
    tp=fp=fn=0;critical=detected=0;critical_misses=[]
    for flag in SafetyFlag:
        ft=ff=fnn=0
        for row in safety_rows:
            ref=references[row['post_id']];p=by_id.get(row['post_id'],{})
            flags=(p.get('classification') or {}).get('safety_flags',[])
            a,b=flag.value in ref['safety_flags'],flag.value in flags
            ft+=a and b;ff+=not a and b;fnn+=a and not b
        tp+=ft;fp+=ff;fn+=fnn
        output['safety'].append(dict(flag=flag.value,precision=ft/(ft+ff) if ft+ff else 0,recall=ft/(ft+fnn) if ft+fnn else 0,support=ft+fnn,tp=ft,fp=ff,fn=fnn))
    for row in safety_rows:
        ref=references[row['post_id']];p=by_id.get(row['post_id'],{});guess=p.get('classification') or {}
        expected=set(ref['safety_flags']);actual=set(guess.get('safety_flags',[]))
        if expected!=actual:
            output['errors'].append(dict(post_id=row['post_id'],text=redact(row['text']),field='safety_flags',expected=sorted(expected),predicted=sorted(actual),language=ref.get('language','unlabelled'),topic=ref.get('topic','unlabelled'),urgency=ref.get('urgency','unlabelled'),safety_critical=bool(expected),error_code=p.get('error_code')))
        # Critical recall requires critical urgency AND capture of every expected
        # physical danger flag. Privacy and non-imminent cases reported separately.
        if ref.get('urgency')=='critical':
            critical+=1
            hit=guess.get('urgency')=='critical' and expected.issubset(actual)
            detected+=hit
            if not hit: critical_misses.append(row['post_id'])
    output.update(safety_reference_count=len(safety_rows),safety_micro_precision=tp/(tp+fp) if tp+fp else None,safety_micro_recall=tp/(tp+fn) if tp+fn else None,critical_safety_recall=detected/critical if critical else None,critical_case_count=critical,critical_misses=critical_misses)
    for language in ['English','Hindi','Hinglish','other']:
        ids=[r['post_id'] for r in rows if references.get(r['post_id'],{}).get('language')==language and 'sentiment' in references.get(r['post_id'],{})]
        if ids:
            actual=[references[i]['sentiment'] for i in ids];predicted=[(by_id.get(i,{}).get('classification') or {}).get('sentiment','INVALID') for i in ids]
            output['by_language'].append(dict(language=language,n=len(ids),sentiment_accuracy=accuracy_score(actual,predicted)))
    return output

def run(directory=None,split='all',limit=None,delay=0.3):
    candidates=build_candidates()
    rows=[r for r in candidates if split=='all' or r['split']==split]
    if limit: rows=rows[:limit]
    model=classifier_model();prompt,sha=load_prompt()
    directory=Path(directory) if directory else RUNS/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory.mkdir(parents=True,exist_ok=True)
    metadata=dict(model_id=model,prompt_version=PROMPT_VERSION,prompt_sha256=sha,dataset_version='llm_balanced_v3',dataset_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),post_ids=[r['post_id'] for r in rows],split=split,label_status='provisional unless separately human-approved',created_at=datetime.now(timezone.utc).isoformat())
    meta_path=directory/'metadata.json'
    if meta_path.exists():
        previous=json.loads(meta_path.read_text())
        for key in ['model_id','prompt_sha256','dataset_sha256','split']:
            if previous[key]!=metadata[key]: raise ValueError('Resume configuration does not match saved run: '+key)
        metadata=previous
    else: meta_path.write_text(json.dumps(metadata,indent=2),encoding='utf8')
    checkpoint=directory/'predictions.jsonl'
    results={}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding='utf8').splitlines():
            p=json.loads(line);results[p['post_id']]=p
    for i,row in enumerate(rows):
        previous=results.get(row['post_id'],{})
        if previous.get('status')=='success' or previous.get('error_code')=='schema_error': continue
        result=classify_post(row['text'],row['post_id'],model=model).model_dump(mode='json')
        with checkpoint.open('a',encoding='utf8') as handle: handle.write(json.dumps(result,ensure_ascii=False)+'\n');handle.flush()
        results[row['post_id']]=result
        print(f"{i+1}/{len(rows)} {row['post_id']} {result['status']} {result.get('error_code') or ''}",flush=True)
        if result.get('error_code')=='missing_key': break
        if delay: time.sleep(delay)
    return save_report(directory,rows,list(results.values()),metadata)


def save_report(directory,rows,predictions,metadata):
    human,disputes=agreed_references()
    history=[json.loads(line) for line in (directory/'predictions.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
    report={'metadata':metadata,'disputed_annotations':disputes,'attempt_history':dict(record_attempts=len(history),outcomes=dict(Counter(p.get('error_code') or 'success' for p in history))), 'categories':{}}
    for partition in ['all','development','heldout']:
        subset=[r for r in rows if partition=='all' or r['split']==partition]
        preds=[p for p in predictions if p['post_id'] in {r['post_id'] for r in subset}]
        refs={r['post_id']:r['provisional_labels'] for r in subset}
        verified={r['post_id']:human[r['post_id']] for r in subset if r['post_id'] in human}
        report['categories'][partition]=dict(provisional=summarize(subset,preds,refs),human_verified=summarize([r for r in subset if r['post_id'] in verified],[p for p in preds if p['post_id'] in verified],verified) if verified else None)
    (directory/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
    RUNS.mkdir(parents=True,exist_ok=True)
    (RUNS/'latest.json').write_text(json.dumps({'directory':str(directory.resolve())}),encoding='utf8')
    import pandas as pd
    for partition,categories in report['categories'].items():
        for category,data in categories.items():
            if data is None: continue
            prefix=partition+'_'+category
            pd.DataFrame([dict(field=k,**{x:v[x] for x in ['n','accuracy','macro_f1','weighted_f1']}) for k,v in data['fields'].items()]).to_csv(directory/(prefix+'_metrics.csv'),index=False)
            pd.DataFrame(data['errors']).to_csv(directory/(prefix+'_errors.csv'),index=False)
    return report,directory


def refresh_report(directory):
    directory=Path(directory)
    metadata=json.loads((directory/'metadata.json').read_text(encoding='utf8'))
    ids=set(metadata['post_ids'])
    rows=[r for r in build_candidates() if r['post_id'] in ids]
    results={}
    for line in (directory/'predictions.jsonl').read_text(encoding='utf8').splitlines():
        value=json.loads(line);results[value['post_id']]=value
    return save_report(directory,rows,list(results.values()),metadata)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory');parser.add_argument('--split',choices=['all','development','heldout'],default='all');parser.add_argument('--limit',type=int);parser.add_argument('--delay',type=float,default=.3);parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args();report,directory=refresh_report(args.directory) if args.summarize_only else run(args.directory,args.split,args.limit,args.delay)
    print('Saved evaluation:',directory)
