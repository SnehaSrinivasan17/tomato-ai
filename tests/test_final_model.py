import json
import shutil

import numpy as np
import pandas as pd
import pytest

from ml.bundle import active_model_directory,validate_bundle
from ml.charts import build_figures
from ml.dashboard import load_bundle
from ml.inference import SpreadPredictor
from ml.model import metrics


def test_active_bundle_is_consistent():
    directory=active_model_directory()
    bundle=load_bundle(directory)
    manifest=validate_bundle(directory)
    assert bundle['selection']['model_version']==manifest['version']
    model=SpreadPredictor()
    assert model.directory==directory
    assert model.selection['thresholds']['operational']==bundle['selection']['thresholds']['f1_optimal']
    test=bundle['predictions'].query("split == 'test'")
    actual=metrics(test.label,test.probability,model.selection['thresholds']['operational'])
    for key,value in actual.items():
        assert np.isclose(value,bundle['metrics']['test']['operational'][key])
    assert len(test)==bundle['audit']['partitions']['test']['rows']


def test_graphs_are_nonempty_plain_arrays_with_operating_point():
    bundle=load_bundle(active_model_directory())
    figures=build_figures(bundle)
    assert len(figures)==6
    for figure in figures:
        value=figure.to_plotly_json()
        assert value['data']
        assert 'bdata' not in figure.to_json()
        assert figure.layout.title.text
        for trace in value['data']:
            assert len(trace.get('x',trace.get('z',[])))>0
    point=figures[1].data[1]
    metrics=bundle['metrics']['test']['operational']
    assert point.x[0]==metrics['recall']
    assert point.y[0]==metrics['precision']


def test_corrupt_or_missing_chart_is_reported(tmp_path):
    directory=active_model_directory()
    for path in directory.iterdir():
        if path.is_file():
            shutil.copy2(path,tmp_path/path.name)
    (tmp_path/'roc_curve.csv').unlink()
    with pytest.raises(ValueError,match='roc_curve.csv'):
        load_bundle(tmp_path)


def test_frozen_threshold_explorer_does_not_change_artifacts():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    from ml.bundle import digest
    directory=active_model_directory()
    before=digest(directory/'selection.json')
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'),default_timeout=90).run()
    app.sidebar.radio[0].set_value('ML Spread Lab').run()
    assert not app.exception
    assert len(app.get('plotly_chart'))>=6
    assert not app.selectbox
    next(r for r in app.radio if r.label=='Model view').set_value('Threshold explorer').run()
    app.slider[0].set_value(.5).run()
    assert not app.exception
    bundle=load_bundle(directory)
    test=bundle['predictions'].query("split == 'test'")
    expected=metrics(test.label,test.probability,.5)
    cards={item.label:item.value for item in app.metric}
    assert cards['Exploratory RECALL']==f"{expected['recall']:.3f}"
    assert cards['Exploratory F1']==f"{expected['f1']:.3f}"
    assert before==digest(directory/'selection.json')
