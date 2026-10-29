from pathlib import Path
from unittest.mock import patch
import pytest
from streamlit.testing.v1 import AppTest
from src.review.store import record_decision,review,get_decision
from tests.test_decision_phase4 import eligible_record
from tests.test_demo_phase5 import callback

ROOT=Path(__file__).resolve().parents[1]
PAGES=['Command Center','ML Spread Lab','LLM Evaluation Lab','RAG & Response Lab','Crisis Monitoring & Reviewer Queue','Demo & Audit']

def test_all_six_sections_and_plain_array_charts(tmp_path):
    with patch('src.demo.runner.active_path',return_value=tmp_path/'review.db'),patch('src.demo.ui.active_path',return_value=tmp_path/'review.db'),patch('src.command.review.active_path',return_value=tmp_path/'review.db'),patch('src.demo.ui.analyze') as live:
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
        assert app.sidebar.radio[0].options==PAGES
        for page in PAGES:
            app.sidebar.radio[0].set_value(page).run()
            assert not app.exception,page
            assert all('bdata' not in str(c.proto) for c in app.get('plotly_chart')),page
            if page=='ML Spread Lab':assert len(app.get('plotly_chart'))>=7
            if page=='LLM Evaluation Lab':assert len(app.get('plotly_chart'))>=8
        assert not live.called

def test_unified_custom_post_persists_and_widgets_do_not_rerun(tmp_path):
    path=tmp_path/'review.db';fn=callback()
    with patch('src.demo.runner.active_path',return_value=path),patch('src.demo.ui.active_path',return_value=path),patch('src.workflow.pipeline.process_custom_post',side_effect=fn):
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
        next(r for r in app.radio if r.label=='Analyze Post input').set_value('Custom post').run()
        next(b for b in app.button if b.label=='Analyze Post').click().run()
        assert not fn.called and any('characters' in e.value for e in app.error)
        next(t for t in app.text_area if t.label.startswith('Post text')).set_value('My fictional dinner was late.').run()
        next(b for b in app.button if b.label=='Analyze Post').click().run()
        assert not app.exception and fn.call_count==1
        assert fn.call_args.args[1]=={} # no metadata defaults or text-derived spread
        app.run();assert fn.call_count==1
        next(b for b in app.button if b.label=='Analyze Post').click().run()
        assert not app.exception and fn.call_count==1 # persistent, version-aware hit

def test_duplicate_review_is_rejected_and_publication_disabled(tmp_path):
    path=tmp_path/'review.db';key=record_decision(eligible_record(),path)
    r=review(key,'APPROVED','test-human',human_confirmed=True,expected_revision=0,path=path)
    assert r['publish_allowed'] is False
    with pytest.raises(ValueError,match='duplicate'):review(key,'APPROVED','test-human',human_confirmed=True,expected_revision=1,path=path)
    assert len(get_decision(key,path)['events'])==1
