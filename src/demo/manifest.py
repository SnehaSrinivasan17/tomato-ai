import hashlib,json,random
from pathlib import Path
from datetime import datetime,timedelta,timezone
from functools import lru_cache

ROOT=Path(__file__).resolve().parents[2]
DIRECTORY=ROOT/'data/demo'
SOURCE=ROOT/'data/synthetic/food_delivery.jsonl'
SEED=20261029

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prepare(directory=DIRECTORY):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    target=directory/'manifest.json'
    if target.exists():return json.loads(target.read_text(encoding='utf8'))
    rows=[json.loads(x) for x in SOURCE.read_text(encoding='utf8').splitlines() if x.strip()]
    by_id={r['post_key']:r for r in rows}
    from llm.annotation import build_candidates,SUPPLEMENTS
    candidates=build_candidates();known={r['post_id'] for r in candidates if r['source']=='existing_ml_dataset'}
    featured={}
    for name,topic,sentiment in [('praise','late_delivery','positive'),('missing_item','missing_item','negative')]:
        featured[name]=next(r['post_key'] for r in rows if r['post_key'] in known and r['reference_topic']==topic and r['reference_sentiment']==sentiment)
    chosen=set(featured.values());pool=sorted(set(by_id)-chosen)
    chosen.update(random.Random(SEED).sample(pool,min(750-len(chosen),len(pool))))
    ordered=sorted(chosen,key=lambda key:(by_id[key]['date'],key))
    start=datetime(2026,1,1,tzinfo=timezone.utc)
    entries=[dict(post_id=key,original_timestamp=by_id[key]['date'],presentation_timestamp=(start+timedelta(seconds=i*7*86400/max(len(ordered),1))).isoformat(),source='existing_ml_dataset',language_provenance='English source templates; not a model prediction') for i,key in enumerate(ordered)]
    # Reuse already separate language cases; do not manufacture first-hour metadata.
    supplemental=[dict(post_id=r['post_id'],text=r['text'],source=r['source'],language_hint=r['provisional_labels']['language'],human_verified=False,metadata=None,original_timestamp=None,presentation_timestamp=(start+timedelta(days=i%7,hours=12)).isoformat()) for i,r in enumerate(candidates) if r['source']=='llm_language_supplement' and r['post_id'] in {'LLM_V3_Hinglish_11_0','LLM_SMOKE_Hindi_delivery','LLM_SMOKE_allergy'}]
    if not any(r['post_id']=='LLM_SMOKE_allergy' for r in supplemental):
        text=next(r[1] for r in SUPPLEMENTS if r[0]=='allergy')
        supplemental.append(dict(post_id='LLM_SMOKE_allergy',text=text,source='llm_language_supplement',language_hint='English',human_verified=False,metadata=None,original_timestamp=None,presentation_timestamp=(start+timedelta(days=3,hours=12)).isoformat()))
    if not any(r.get('language_hint')=='Hindi' for r in supplemental):
        row=next(r for r in candidates if r['source']=='llm_language_supplement' and r['provisional_labels']['language']=='Hindi')
        supplemental.append(dict(post_id=row['post_id'],text=row['text'],source=row['source'],language_hint='Hindi',human_verified=False,metadata=None,original_timestamp=None,presentation_timestamp=(start+timedelta(days=4)).isoformat()))
    featured.update(allergy='LLM_SMOKE_allergy',hinglish_sarcasm='LLM_V3_Hinglish_11_0')
    config=dict(praise='Positive low-risk monitoring; optional reply is not an obligation.',missing_item='Routine complaint; retrieve policy and draft only if supported.',allergy='Unverified safety allegation; escalate regardless of spread.',hinglish_sarcasm='Hinglish sarcasm, negative intent; policy-grounded complaint handling.')
    (directory/'supplements.json').write_text(json.dumps(supplemental,ensure_ascii=False,indent=2),encoding='utf8')
    (directory/'scenarios.json').write_text(json.dumps({k:dict(post_id=v,expected_behavior=config[k],expectations_are_not_predictions=True) for k,v in featured.items()},indent=2),encoding='utf8')
    manifest=dict(version='1.0.0',seed=SEED,source='data/synthetic/food_delivery.jsonl',source_sha256=sha(SOURCE),source_records=len(rows),selected_originals=len(entries),shortfall=max(0,750-len(entries)),presentation_mapping='Seven presentation days starting 2026-01-01 UTC; original dates retained; mapping never enters ML.',records=entries)
    target.write_text(json.dumps(manifest,indent=2),encoding='utf8');return manifest

@lru_cache(maxsize=4)
def _source_records(source_hash):
    return {r['post_key']:r for r in map(json.loads,SOURCE.read_text(encoding='utf8').splitlines())}

def posts(directory=DIRECTORY):
    directory=Path(directory);m=prepare(directory)
    if sha(SOURCE)!=m['source_sha256']:raise ValueError('Source dataset differs from the demo manifest; no records silently remapped.')
    source=_source_records(m['source_sha256'])
    originals=[dict(e,text=source[e['post_id']]['text'],language_hint='English',metadata=source[e['post_id']]) for e in m['records']]
    return originals+json.loads((directory/'supplements.json').read_text(encoding='utf8'))
