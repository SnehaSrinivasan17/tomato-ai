from pathlib import Path
import json

import numpy as np
import pandas as pd

from ml.bundle import ACTIVE,digest
from ml.charts import categorical_confusion
from ml.comparisons import experiment_table
from llm.evaluate import load_benchmark,evaluate_records

ROOT=Path(__file__).resolve().parents[1]


def test_saved_comparison_data_and_counts():
    for experiment in ['latest','earlier']:
        table,_=experiment_table(experiment)
        assert len(table)>=5
        assert np.allclose(table.accuracy,(table.tp+table.tn)/(table.tp+table.tn+table.fp+table.fn))
        if experiment=='latest':
            assert (table.threshold==.5).all()


def test_all_llm_confusion_matrices_have_plain_arrays_and_correct_counts():
    refs,preds=load_benchmark()
    _,_,_,matrices=evaluate_records(refs,preds)
    for name,values in matrices.items():
        matrix=pd.DataFrame(values).T
        fig=categorical_confusion(matrix,name)
        assert 'bdata' not in fig.to_json()
        assert np.asarray(fig.data[0].z).sum()==len(refs)


def test_comparison_navigation_and_llm_field_selection():
    from streamlit.testing.v1 import AppTest
    before=digest(ACTIVE)
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
    app.sidebar.radio[0].set_value('ML Spread Lab').run()
    next(r for r in app.radio if r.label=='ML lab view').set_value('Other tested models').run()
    assert not app.exception
    assert len(app.dataframe)>=1
    assert len(app.get('plotly_chart'))==2
    next(item for item in app.radio if item.label=='Experiment').set_value('Earlier tuning · held-out').run()
    assert not app.exception
    app.selectbox[0].select('Shifted simulation').run()
    assert not app.exception
    app.sidebar.radio[0].set_value('LLM Evaluation Lab').run()
    assert not app.exception
    app.selectbox[0].select('Historical GPT-OSS provisional benchmark').run()
    for name in ['risk_flag','urgency','sentiment']:
        app.selectbox[1].select(name).run()
        assert not app.exception
        charts=app.get('plotly_chart')
        assert len(charts)==2
        assert 'bdata' not in str(charts[1].proto)
    assert before==digest(ACTIVE)
