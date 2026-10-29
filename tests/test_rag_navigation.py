"""Regression coverage for the Cloud report-selection crash."""
import json
import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize('view',['Saved RAG evaluation','System overview','Response quality'])
def test_every_rag_view_loads(view):
    app=AppTest.from_string('from src.rag.ui import render; render()').run(timeout=30)
    app.radio[0].set_value(view).run(timeout=30)
    assert not app.exception


@pytest.mark.parametrize('pointer',['windows','missing','invalid'])
def test_pointer_fallbacks(tmp_path,monkeypatch,pointer):
    from src.rag import ui
    reports=tmp_path/'evaluations'/'saved_run';reports.mkdir(parents=True)
    (reports/'report.json').write_text(json.dumps({'metadata':{}}),encoding='utf8')
    content={'windows':json.dumps({'directory':r'C:\project\evaluations\saved_run'}),'missing':json.dumps({'directory':'gone_run'}),'invalid':'not json'}[pointer]
    (tmp_path/'latest_evaluation.json').write_text(content,encoding='utf8')
    monkeypatch.setattr(ui,'RESULTS',tmp_path)
    app=AppTest.from_string('from src.rag.ui import render; render()').run(timeout=30)
    assert not app.exception
    assert any('saved_run' in str(c.value) for c in app.caption)
