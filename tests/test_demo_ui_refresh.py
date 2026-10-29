"""Focused regression checks for the lighter integrated demo UI."""
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from streamlit.testing.v1 import AppTest
from llm.client import AuthFallbackCompletions


def test_fallback_only_authentication_failure():
    class Failure(Exception):
        def __init__(self,status):self.status_code=status
    primary=Mock();standby=Mock()
    primary.chat.completions.create.side_effect=Failure(401)
    standby.chat.completions.create.return_value='actual standby response'
    client=AuthFallbackCompletions(primary,standby)
    assert client.create()=='actual standby response'
    assert client.create()=='actual standby response'
    assert primary.chat.completions.create.call_count==1
    primary=Mock();primary.chat.completions.create.side_effect=Failure(429)
    standby=Mock()
    with pytest.raises(Failure):AuthFallbackCompletions(primary,standby).create()
    standby.chat.completions.create.assert_not_called()


def test_examples_have_valid_actual_spread_scores():
    from src.demo.ui import example_catalog
    from src.workflow.pipeline import predictor
    from llm.combined import validate_metadata
    examples=example_catalog()
    assert len(examples)==6
    scores=[]
    for _,post,_ in examples:
        values,errors=validate_metadata(post['metadata'],predictor().schema)
        assert not errors and post['text'].strip()
        scores.append(predictor().predict(values)['probability'])
    assert all(0<=p<=1 for p in scores)
    assert min(scores)<predictor().selection['thresholds']['operational']<max(scores)


def test_result_renderer_accepts_compact_and_does_not_show_ids():
    app=AppTest.from_string("""
from tests.test_decision_phase4 import eligible_record
from src.decision.ui import show_result
r=eligible_record()
r['customer_post']='Example customer message'
r['review_status']='PENDING_REVIEW'
r['post_id']='INTERNAL_PRIVATE_ID'
show_result(r,compact=True)
""").run(timeout=30)
    assert not app.exception
    assert any(m.label=='Spread probability' for m in app.metric)
    assert not any('INTERNAL_PRIVATE_ID' in str(m.value) for m in app.markdown)


@pytest.mark.parametrize('section',['Command Center','Demo & Audit','Crisis Monitoring & Reviewer Queue','RAG & Response Lab'])
def test_light_pages_load(section,monkeypatch):
    from src.review import postgres
    monkeypatch.setattr(postgres,'configured',lambda:False)
    app=AppTest.from_file('app.py').run(timeout=30)
    app.sidebar.radio[0].set_value(section).run(timeout=30)
    assert not app.exception
    if section=='Demo & Audit':
        next(r for r in app.radio if r.label=='Analyze Post input').set_value('Custom post').run(timeout=30)
        assert not app.exception
        assert len(app.number_input)>0


def test_packaged_vectors_avoid_local_encoder(tmp_path,monkeypatch):
    import json,hashlib,numpy as np
    from pathlib import Path
    from src.rag import embeddings
    directory=Path('data/demo/query_vectors')
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf8'))
    assert manifest['files']
    monkeypatch.setenv('BGE_MODEL_REVISION',manifest['revision'])
    monkeypatch.setenv('BGE_M3_ENDPOINT','')
    def no_model():raise AssertionError('Demo query must not load the large encoder')
    monkeypatch.setattr(embeddings,'local_model',no_model)
    encoder=embeddings.BGEEncoder(cache=tmp_path)
    for name,query in manifest['queries'].items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==manifest['files'][name]
        vector=encoder.encode([query])
        assert vector.shape==(1,1024) and np.isfinite(vector).all()
        assert abs(np.linalg.norm(vector)-1)<1e-5
