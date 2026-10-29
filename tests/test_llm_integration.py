import hashlib,json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
import pandas as pd
import pytest
from llm.classifier import classify_post
from llm.combined import combined_analysis,review_decision,validate_metadata
from llm.evaluate import BENCHMARK,evaluate_records,load_benchmark,run_evaluation
from ml.inference import SpreadPredictor
ROOT=Path(__file__).resolve().parents[1]
PAYLOAD=dict(sentiment='negative',topic='food_quality',secondary_topics=[],urgency='critical',suggested_severity='P0',language='English',sarcasm_detected=False,safety_flags=['food_safety','medical_emergency'],response_recommended=True,evidence_phrases=[],brief_reason='Fictional safety concern.',ambiguous=False)
def fake_client(payload=None):
    client=Mock()
    client.chat.completions.create.return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(PAYLOAD if payload is None else payload)))],usage=None)
    return client

def test_model_and_original_prompt_preserved():
    assert (ROOT/'llm/prompts/classifier_prompt.txt').read_bytes()==(ROOT/'AIAgent/prompts/classifier_prompt.txt').read_bytes()
    client=fake_client();result=classify_post('Fictional complaint.',client=client,persist=False)
    assert result.status=='success' and result.model_id=='openai/gpt-oss-120b'
    assert result.self_reported_confidence is None
    assert client.chat.completions.create.call_args.kwargs['temperature']==0

@pytest.mark.parametrize('text',['','  ',None,123,'x'*10001])
def test_invalid_post_text(text):
    result=classify_post(text,client=fake_client(),persist=False)
    assert result.status=='error' and result.classification is None

@pytest.mark.parametrize('payload',[[],{'sentiment':'negative'},dict(PAYLOAD,urgency='invalid'),dict(PAYLOAD,sarcasm_detected='false'),dict(PAYLOAD,evidence_phrases=['invented quote'])])
def test_invalid_api_outputs(payload):
    assert classify_post('Fictional complaint.',client=fake_client(payload),persist=False).status=='error'

def test_missing_key_and_safe_api_error():
    with patch('llm.classifier.get_setting',return_value=''):
        result=classify_post('unique missing key smoke',use_cache=False,persist=False)
        assert result.error_code=='missing_key'
    client=fake_client();client.chat.completions.create.side_effect=RuntimeError('sensitive-key-must-not-leak')
    result=classify_post('Fictional complaint.',client=client,persist=False)
    assert 'sensitive-key-must-not-leak' not in result.model_dump_json()

def test_cache_and_reference_isolation(tmp_path):
    before={p.name:p.read_bytes() for p in BENCHMARK.iterdir() if p.is_file()}
    client=fake_client();path=tmp_path/'predictions.sqlite3'
    a=classify_post('unique cache test',client=client,use_cache=True,storage_path=path)
    b=classify_post('unique cache test',client=client,use_cache=True,storage_path=path)
    assert a.status=='success' and b.cache_hit and client.chat.completions.create.call_count==1
    classify_post('unique cache test',client=client,model='different',use_cache=True,storage_path=path)
    assert client.chat.completions.create.call_count==2
    assert before=={p.name:p.read_bytes() for p in BENCHMARK.iterdir() if p.is_file()}

def test_combined_real_ml_and_missing_metadata():
    predictor=SpreadPredictor();metadata={k:int(v) for k,v in predictor.schema['defaults'].items()}
    callback=lambda text:classify_post(text,client=fake_client(),persist=False)
    result=combined_analysis('Fictional complaint.',metadata,classify=callback,predictor=predictor)
    assert result['ml']['probability']==predictor.predict(metadata)['probability']
    assert result['review']['priority_review']
    result=combined_analysis('Fictional complaint.',{},classify=callback,predictor=predictor)
    assert result['ml'] is None and result['metadata_errors'] and result['review']['priority_review']

def test_benchmark_recomputation_and_isolation(tmp_path):
    reference, predictions = load_benchmark()
    assert all(not r['post_id'].startswith('TOM_LIVE_') for r in reference+predictions)
    metrics, comparison, errors, matrices = evaluate_records(reference, predictions)
    saved = pd.read_csv(BENCHMARK/'field_metrics.csv').set_index('field')
    actual = metrics.set_index('field')
    for column in ['accuracy','macro_precision','macro_recall','macro_f1']:
        np.testing.assert_allclose(saved[column], actual[column], atol=.000051)
    assert len(comparison) == len(reference)*8
    assert len(errors) == len(pd.read_csv(BENCHMARK/'error_analysis.csv'))
    assert sum(sum(row.values()) for row in matrices['risk_flag'].values()) == len(reference)
    _, output = run_evaluation(output=tmp_path/'offline')
    assert (output/'confusion_matrices.json').exists()


def test_ml_files_unchanged():
    manifest = json.loads((ROOT/'integration_ml_before.json').read_text())
    for relative, digest in manifest.items():
        # Phase 1.5 explicitly authorizes feature/inference changes; archived data
        # and models must still be unchanged. The active new bundle is hash-bound.
        if relative.replace('\\','/') in {'ml/features.py','ml/inference.py'}:
            continue
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == digest, relative


def test_root_environment_only_no_override(tmp_path, monkeypatch):
    import config
    (tmp_path/'.env').write_text('GROQ_MODEL=test-root-model\n')
    monkeypatch.setattr(config,'PROJECT_ROOT',tmp_path)
    monkeypatch.setenv('GROQ_MODEL','existing-process-model')
    config.load_project_environment()
    assert __import__('os').environ['GROQ_MODEL'] == 'existing-process-model'



def test_dashboard_navigation_and_classification():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
    for page in ['Command Center','LLM Evaluation Lab','ML Spread Lab','RAG & Response Lab','Crisis Monitoring & Reviewer Queue','Demo & Audit']:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception,page
    app.sidebar.radio[0].set_value('LLM Evaluation Lab').run()
    app.text_area[0].set_value('Fictional safety concern.').run()
    with patch('llm.ui.classify_post',side_effect=lambda text,post_id=None:classify_post(text,post_id,client=fake_client(),persist=False)):
        next(b for b in app.button if b.label=='Classify post').click().run()
    assert not app.exception
    assert any(item.label=='Suggested Severity' and item.value=='P0' for item in app.metric)
