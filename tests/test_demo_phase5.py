"""Fixture-driven integration tests; not measured LLM accuracy."""
import json,hashlib
from pathlib import Path
from unittest.mock import Mock
import pytest
from src.demo.manifest import prepare,posts,SOURCE
from src.demo.runner import analyze,history,feed,analytics,reset,active_path,versions
from src.review.store import review,get_decision
from tests.test_decision_phase4 import eligible_record

def test_manifest_shared_ids_immutable_and_repeatable(tmp_path):
    before=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    a=prepare(tmp_path);b=prepare(tmp_path)
    assert a==b and a['selected_originals']==750 and a['shortfall']==0
    rows=posts(tmp_path);original=[r for r in rows if r['source']=='existing_ml_dataset']
    assert len({r['post_id'] for r in rows})==len(rows)
    assert all(r['post_id'].startswith('syn-post-') for r in original)
    assert len({r['presentation_timestamp'][:10] for r in original})==7
    assert any(r['original_timestamp']!=r['presentation_timestamp'] for r in original)
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==before==a['source_sha256']
    supplements=[r for r in rows if r['source']!='existing_ml_dataset']
    assert {'Hindi','Hinglish'}<={r['language_hint'] for r in supplements}
    assert all(r['metadata'] is None for r in supplements)
    scenarios=json.loads((tmp_path/'scenarios.json').read_text())
    assert len(scenarios)==4
    assert all(s['post_id'] in {r['post_id'] for r in rows} for s in scenarios.values())

def callback():
    def run(text,metadata,post_id,persist):
        r=eligible_record();r['post_id']=post_id;r['decision']['post_id']=post_id
        r.update(status='review_required',errors=[],requires_human_review=True,publish_allowed=False)
        return r
    return Mock(side_effect=run)

def test_pipeline_cache_versions_live_and_unknown_metadata(tmp_path):
    post=dict(post_id='stable',text='Order late.',metadata=None);fn=callback();path=tmp_path/'review.db'
    one=analyze(post,path=path,workflow=fn,version_info={'model':'v1'})
    two=analyze(post,path=path,workflow=fn,version_info={'model':'v1'})
    assert not one['pipeline_cache_hit'] and two['pipeline_cache_hit'] and fn.call_count==1
    assert fn.call_args.args[1]=={} and fn.call_args.kwargs['persist'] is False
    assert two['post_id']=='stable' and two['decision_id']==one['decision_id']
    analyze(post,path=path,workflow=fn,version_info={'model':'v2'})
    analyze(post,path=path,workflow=fn,version_info={'model':'v2'},force=True)
    assert fn.call_count==3 and len(history(path))==3

@pytest.mark.parametrize('component',['classification','generation','retrieval'])
def test_failures_do_not_become_success_cache(tmp_path,component):
    fn=callback();original=fn.side_effect
    def fail(*a,**kw):
        r=original(*a,**kw);r[component]=dict(status='error',error_code='fixture_failure');return r
    fn.side_effect=fail;p=dict(post_id='failure',text='Order late.')
    for _ in range(2):analyze(p,path=tmp_path/'review.db',workflow=fn,version_info={})
    assert fn.call_count==2 and not any(r['complete'] for r in history(tmp_path/'review.db'))

def test_reset_archives_review_and_preserves_manifest(tmp_path):
    root=tmp_path/'state';path=active_path(root);fn=callback();p=dict(post_id='stable',text='Order late.')
    r=analyze(p,path=path,workflow=fn,version_info={})
    review(r['decision_id'],'APPROVED','test-human',human_confirmed=True,expected_revision=0,path=path)
    assert get_decision(r['decision_id'],path)['review_status']=='APPROVED'
    before=path.read_bytes();result=reset(root);new=active_path(root)
    assert new!=path and path.read_bytes()==before and result['deleted_files']==0
    assert not history(new) and len(history(path))==1

def test_actual_analytics_exclude_unprocessed_and_refresh_review(tmp_path):
    path=tmp_path/'review.db';fn=callback()
    available=[dict(post_id=k,text='Order late.') for k in ['processed','unprocessed']]
    r=analyze(available[0],path=path,workflow=fn,version_info={})
    analyze(available[0],path=path,workflow=fn,version_info={})
    rows=feed(available,path);stats=analytics(rows,path)
    assert stats['available']==2 and stats['analyzed']==1 and stats['unprocessed']==1 and stats['cache_hits']==1
    assert rows[1]['sentiment'] is None and rows[1]['action'] is None and rows[1]['high_spread'] is None
    review(r['decision_id'],'REJECTED','test-human',human_confirmed=True,path=path)
    assert analytics(feed(available,path),path)['rejected']==1
    cached=analyze(available[0],path=path,workflow=fn,version_info={})
    assert cached['review_status']=='REJECTED'

def test_presentation_and_targets_never_enter_model_metadata(tmp_path):
    fn=callback();post=dict(post_id='stable',text='Order late.',presentation_timestamp='2099-01-01',metadata=dict(first_hour_likes=4,retweet_count=999,reference_sentiment='negative',presentation_timestamp='2099-01-01'))
    analyze(post,path=tmp_path/'review.db',workflow=fn,version_info={})
    assert fn.call_args.args[1]=={'first_hour_likes':4}

def test_version_contract_includes_actual_components():
    v=versions()
    assert v['classifier_model']=='openai/gpt-oss-120b' and v['generation_model']==v['classifier_model']
    assert v['classifier_prompt_sha256'] and v['generation_prompt_sha256'] and v['ml_manifest'] and v['decision_fingerprint'] and v['files']['rag_results/ingestion.json']
