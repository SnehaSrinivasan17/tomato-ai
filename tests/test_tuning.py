import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.features import EngagementFeatures
from ml.synthetic import FEATURES
from ml.tune import group_temporal_folds, threshold_selection

ROOT=Path(__file__).resolve().parents[1]


def test_temporal_cv_purges_repeated_authors():
    size=60
    frame=pd.DataFrame({'date':pd.date_range('2026-01-01',periods=size,tz='UTC'),
        'user_key':[str(i) if i%5 else 'repeat' for i in range(size)],'label':[i%2 for i in range(size)]})
    folds,audit=group_temporal_folds(frame)
    for (a,b),info in zip(folds,audit):
        assert not set(frame.iloc[a].user_key)&set(frame.iloc[b].user_key)
        assert frame.iloc[a].date.max() < frame.iloc[b].date.min()
        assert info['overlapping_authors']==0


def test_engineering_zero_counts_and_no_mutation():
    x=np.zeros((3,len(FEATURES)))
    result=EngagementFeatures().fit_transform(x)
    assert result.shape[1]>x.shape[1]
    assert np.isfinite(result).all()
    assert not x.any()


def test_operational_constraints_are_validation_only():
    y=np.array([0,0,0,0,0,0,0,0,1,1])
    p=np.array([.01,.02,.03,.04,.05,.06,.07,.08,.3,.7])
    sweep,choices,feasible=threshold_selection(y,p)
    assert feasible
    selected=sweep[sweep.threshold==choices['operational']].iloc[0]
    assert selected.recall>=.5
    assert selected.alert_rate<=.3


def test_tuned_data_and_targets_preserved_and_fresh_confirmation():
    from ml.train import dataset_digest
    art=ROOT/'artifacts/synthetic_ml'
    old=ROOT/'artifacts/synthetic_previous'
    previous=json.loads((old/'run.json').read_text())
    assert dataset_digest(ROOT/'data/synthetic/food_delivery.jsonl')==previous['data_sha256']
    assert dataset_digest(ROOT/'data/synthetic/shifted_test.jsonl')==previous['shifted_data_sha256']
    assert json.loads((old/'label_definition.json').read_text())==json.loads((art/'label_definition.json').read_text())
    original=pd.read_json(ROOT/'data/synthetic/food_delivery.jsonl',lines=True)
    confirmation=pd.read_json(ROOT/'data/synthetic/confirmation_test.jsonl',lines=True)
    assert set(original.post_key).isdisjoint(confirmation.post_key)
    assert set(original.user_key).isdisjoint(confirmation.user_key)
    assert len(confirmation)==3000
    tuning=json.loads((art/'tuning.json').read_text())
    assert all(fold['overlapping_authors']==0 for fold in tuning['folds'])


def test_heldout_comparison_metrics_and_no_warning_sections():
    art=ROOT/'artifacts/synthetic_ml'
    comparison=pd.read_csv(art/'model_comparison_test.csv')
    assert set(['previous_logistic','majority_baseline','logistic','random_forest','xgboost','lightgbm','catboost']).issubset(comparison.model)
    assert (comparison[comparison.model=='majority_baseline'].recall==0).all()
    assert np.allclose(comparison.accuracy,(comparison.tn+comparison.tp)/(comparison.tn+comparison.tp+comparison.fp+comparison.fn))
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=90).run()
    app.sidebar.radio[0].set_value('ML Spread Lab').run()
    assert not app.exception
    assert len(app.warning)==0
    assert [tab.label for tab in app.tabs]==['Final model performance','Threshold explorer','Try It Yourself','Historical dataset report']
    assert len(app.selectbox)==0
