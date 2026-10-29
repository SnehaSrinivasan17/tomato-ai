import json
from pathlib import Path
from unittest.mock import patch
import pytest
from llm.annotation import build_candidates,save_annotation,agreed_references,quarantine_legacy,agreement
from llm.schemas import Classification,legacy_candidate
from llm.evaluation import summarize
from llm.storage import redact
from tests.test_llm_integration import PAYLOAD,fake_client
from llm.classifier import classify_post


def test_balanced_candidates_and_ml_linkage():
    from collections import Counter
    rows=build_candidates()
    assert len(rows)==240 and len({r['post_id'] for r in rows})==240
    assert Counter(r['provisional_labels']['language'] for r in rows)==dict(English=80,Hindi=80,Hinglish=80)
    assert sum(r['source']=='existing_ml_dataset' for r in rows)==80
    assert not any(r['human_verified'] for r in rows)
    groups={}
    for row in rows:
        if 'scenario_family' in row: groups.setdefault(row['scenario_family'],set()).add(row['split'])
    assert all(len(splits)==1 for splits in groups.values())


def test_legacy_mapping_is_provisional_and_lossy():
    row=dict(sentiment='negative',topic='food_quality',urgency='critical',risk_flag='food_safety',language='en',sarcasm='false',needs_reply=True,human_review=True)
    mapped=legacy_candidate(row)
    assert not mapped['human_verified'] and mapped['legacy_human_review_advisory']
    assert mapped['labels']['safety_flags']==['food_safety']
    assert mapped['labels']['suggested_severity']=='P1'
    assert mapped['labels']['sarcasm_detected'] is False


def test_annotations_require_review_and_disagreements_excluded(tmp_path):
    post=build_candidates()[0]['post_id'];path=tmp_path/'annotations.sqlite3'
    save_annotation(post,PAYLOAD,'alice',approved=False,path=path)
    assert agreed_references(path)==({},[])
    save_annotation(post,PAYLOAD,'alice',approved=True,path=path)
    assert post in agreed_references(path)[0]
    save_annotation(post,dict(PAYLOAD,brief_reason='Independent wording.'),'bob',approved=True,path=path)
    assert post in agreed_references(path)[0]
    save_annotation(post,dict(PAYLOAD,sentiment='neutral'),'bob',approved=True,path=path)
    assert agreed_references(path)==({},[post])
    with pytest.raises(ValueError): save_annotation(post,PAYLOAD,'',path=path)


def test_multilabel_metrics_and_failed_calls_count_as_misses():
    rows=[dict(post_id='one',text='fictional'),dict(post_id='two',text='fictional two')]
    refs={r['post_id']:PAYLOAD for r in rows}
    predictions=[dict(post_id='one',status='success',classification=PAYLOAD,latency_ms=10),dict(post_id='two',status='error',classification=None,latency_ms=20,error_code='api_error')]
    report=summarize(rows,predictions,refs)
    assert report['json_validity_rate']==.5 and report['api_failure_rate']==.5
    assert report['fields']['sentiment']['accuracy']==.5
    assert report['safety_micro_recall']==.5 and report['critical_safety_recall']==.5
    assert report['critical_misses']==['two']
    assert sum(map(sum,report['fields']['sentiment']['matrix']['counts']))==2


def test_rate_limit_retry_is_bounded_and_no_fake_result():
    class RateLimited(Exception): status_code=429
    client=fake_client();client.chat.completions.create.side_effect=RateLimited('private')
    with patch('llm.classifier.time.sleep'),patch('llm.classifier.limits',return_value=(10,2)):
        result=classify_post('test retry',client=client,persist=False)
    assert result.error_code=='rate_limited' and result.classification is None and result.attempts==3


def test_redaction_and_quarantine():
    assert 'person@example.com' not in redact('email person@example.com phone 9876543210 OTP: 123456')
    assert '9876543210' not in redact('9876543210')
    summary=quarantine_legacy()
    assert summary['historical_benchmark_rows']==50 and summary['quarantined_live_reference_rows']==7


def test_protected_ml_and_original_agent_unchanged():
    import hashlib
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / 'phase2_results/protected_before.json').read_text()
    )
    for relative, digest in manifest.items():
        normalized = relative.replace("\\", "/")
        if normalized == "ml/dashboard.py":
            continue  # Previously approved Phase 6 dashboard changes.

        expected = digest
        if normalized == "ml/bundle.py":
            # Reviewed Phase 7 Linux path compatibility change.
            expected = "fee315bb3f9829b7d207a576dde7ffeb17b454d8cb863a9ec79f1e6d1a6b08b5"

        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        assert actual == expected, relative


def test_saved_canonical_dashboard_charts_and_human_pending():
    from streamlit.testing.v1 import AppTest
    root=Path(__file__).resolve().parents[1]
    app=AppTest.from_file(str(root/'app.py'),default_timeout=90).run()
    app.sidebar.radio[0].set_value('LLM Evaluation Lab').run()
    assert not app.exception
    charts=app.get('plotly_chart')
    assert len(charts)>=8
    assert all('bdata' not in str(chart.proto) for chart in charts)
    next(v for v in app.selectbox if v.label=='Reference category').select('human_verified').run()
    assert not app.exception
    assert any('pending' in v.value.lower() for v in app.info)


def test_checkpoint_resume_skips_successes_and_rejects_changed_model(tmp_path):
    from llm.evaluation import run
    good=classify_post('checkpoint fixture','one',client=fake_client(),persist=False)
    failed=good.model_copy(update=dict(post_id='two',status='error',classification=None,error_code='rate_limited'))
    rows=[dict(post_id=k,text='fixture '+k,split='heldout',provisional_labels=PAYLOAD) for k in ['one','two']]
    directory=tmp_path/'runs'/'case'
    with patch('llm.evaluation.RUNS',tmp_path/'runs'),patch('llm.evaluation.build_candidates',return_value=rows),patch('llm.evaluation.agreed_references',return_value=({},[])),patch('llm.evaluation.classifier_model',return_value='openai/gpt-oss-120b'):
        with patch('llm.evaluation.classify_post',side_effect=[good,failed]): run(directory,delay=0)
        with patch('llm.evaluation.classify_post',return_value=good.model_copy(update={'post_id':'two'})) as callback:
            report,_=run(directory,delay=0)
            assert callback.call_count==1
            assert report['attempt_history']['record_attempts']==3
            assert report['categories']['all']['provisional']['completed_count']==2
        with patch('llm.evaluation.classifier_model',return_value='different-model'):
            with pytest.raises(ValueError,match='model_id'): run(directory,delay=0)
