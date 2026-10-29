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


def test_demo_initial_view_avoids_database_and_compare_work(monkeypatch):
    from src.demo import ui
    def forbidden(*args,**kwargs):
        raise AssertionError('Initial demo view must not query the remote database')
    monkeypatch.setattr(ui,'active_path',forbidden)
    monkeypatch.setattr(ui,'cached_feed',forbidden)
    app=AppTest.from_string('from src.demo.ui import analyze_view; analyze_view("fast_demo")').run(timeout=30)
    assert not app.exception
    assert app.get('form')
    next(r for r in app.radio if r.label=='Analyze Post input').set_value('Custom post').run(timeout=30)
    assert not app.exception
    assert app.text_area and app.get('form')


def test_overview_statistics_do_not_query_database(monkeypatch):
    from src.demo import runner
    def forbidden(*args,**kwargs):raise AssertionError('Unexpected database request')
    monkeypatch.setattr(runner,'active_path',forbidden)
    monkeypatch.setattr(runner,'db',forbidden)
    assert runner.analytics([],include_cache_hits=False)['analyzed']==0


def test_navigation_order():
    import ast
    from pathlib import Path
    tree=ast.parse(Path('app.py').read_text(encoding='utf8'))
    pages=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PAGES' for t in node.targets))
    assert pages[:3]==['Command Center','Demo & Audit','Crisis Monitoring & Reviewer Queue']


def test_custom_form_submits_text_and_metadata_together(monkeypatch,tmp_path):
    from src.demo import ui
    from tests.test_decision_phase4 import eligible_record
    def run(post,**kwargs):
        result=eligible_record()
        result.update(customer_post=post['text'],input_metadata=post['metadata'],decision_id='fixture',review_status='PENDING_REVIEW')
        return result
    workflow=Mock(side_effect=run)
    monkeypatch.setattr(ui,'analyze',workflow)
    monkeypatch.setattr(ui,'active_path',lambda:tmp_path/'review.db')
    app=AppTest.from_string('from src.demo.ui import analyze_view; analyze_view("form_check")').run(timeout=30)
    next(r for r in app.radio if r.label=='Analyze Post input').set_value('Custom post').run(timeout=30)
    app.text_area[0].set_value('My order is missing an item.')
    next(n for n in app.number_input if n.label=='First Hour Likes').set_value(123)
    next(b for b in app.button if b.label=='Analyze Post').click().run(timeout=30)
    assert not app.exception
    workflow.assert_called_once()
    post=workflow.call_args.args[0]
    assert post['text']=='My order is missing an item.'
    assert post['metadata']['first_hour_likes']==123


def test_evaluation_does_not_render_hidden_annotation(monkeypatch):
    from llm import evaluation_ui
    def forbidden():raise AssertionError('Hidden annotation must not query the database')
    monkeypatch.setattr(evaluation_ui,'annotation_ui',forbidden)
    app=AppTest.from_string('from llm.evaluation_ui import render; render(lambda:None,lambda:None)').run(timeout=30)
    assert not app.exception
