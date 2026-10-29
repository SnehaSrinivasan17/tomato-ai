import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.inference import SpreadPredictor
from ml.bundle import active_model_directory
from ml.model import metrics
from ml.synthetic import FEATURES, generate, simulate

ROOT = Path(__file__).resolve().parents[1]
ART = active_model_directory()


def test_simulation_repeatability_and_schema():
    a, _ = simulate(100, seed=7)
    b, _ = simulate(100, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert {'text', 'reference_topic', 'reference_sentiment', 'retweet_count'}.issubset(a)
    assert (a.retweet_count >= a.first_hour_retweets).all()
    assert set(FEATURES).isdisjoint({'text', 'reference_topic', 'reference_sentiment', 'retweet_count',
                                   'label', 'urgency', 'latent_appeal', 'shock', 'platform_amplification'})


def test_saved_data_splits_and_training_quantile():
    data = pd.read_json(ROOT/'data/synthetic/food_delivery.jsonl', lines=True)
    shifted = pd.read_json(ROOT/'data/synthetic/shifted_test.jsonl', lines=True)
    assert len(data) == 15000
    assert len(shifted) == 3000
    manifest = pd.read_csv(ART/'split_manifest.csv')
    assert manifest.post_key.nunique() == len(manifest)
    assert manifest.user_key.nunique() == len(manifest)
    train_ids = set(manifest[manifest.split == 'train'].post_key)
    train = data[data.post_key.isin(train_ids)]
    target = json.loads((ART/'label_definition.json').read_text())
    assert train.retweet_count.quantile(.9) == target['retweet_threshold']
    assert set(data.post_key).isdisjoint(shifted.post_key)
    assert data.retweet_count.quantile(.99) > data.retweet_count.quantile(.5)*10
    assert ((data.followers_count < 500) & (data.retweet_count >= target['retweet_threshold'])).any()
    latent = pd.read_csv(ROOT/'data/synthetic/food_delivery_latent_audit.csv')
    assert latent.shock.any()
    assert latent.late_noise.std() > 0


def test_frozen_metrics_threshold_and_probability():
    predictor = SpreadPredictor()
    assert predictor.schema['source'] == 'synthetic'
    for values in [{}, {'first_hour_likes': 20, 'first_hour_retweets': 5}, {'first_hour_likes': -1}]:
        result = predictor.predict(values)
        assert 0 <= result['probability'] <= 1
        assert result['model_source'] == 'synthetic'
        assert result['high_spread'] == (result['probability'] >= result['operational_threshold'])
    # Reference labels and final outcomes cannot influence the input matrix.
    np.testing.assert_equal(predictor.prepare({'first_hour_likes': 20}).to_numpy(),
        predictor.prepare({'first_hour_likes': 20, 'text': 'anything', 'reference_sentiment': 'negative',
                           'retweet_count': 999999}).to_numpy())
    held = pd.read_csv(ART/'held_out_predictions.csv')
    report = json.loads((ART/'metrics.json').read_text())
    target = json.loads((ART/'label_definition.json').read_text())['retweet_threshold']
    threshold = predictor.selection['thresholds']['operational']
    for split, part in held.groupby('split'):
        assert ((part.retweet_count >= target).astype(int) == part.label).all()
        measured = metrics(part.label, part.probability, threshold)
        for name, value in measured.items():
            assert np.isclose(value, report[split]['operational'][name])
    sweep = pd.read_csv(ART/'thresholds_validation.csv')
    if predictor.selection.get('phase') == '1.5':
        assert threshold == sweep.sort_values(['f1','threshold'],ascending=[False,False]).iloc[0].threshold
        assert set(held.split) == {'validation','test'}
    elif (ART/'tuning.json').exists():
        if predictor.selection['model'] != 'previous_logistic':
            feasible = sweep[(sweep.recall >= .5)&(sweep.alert_rate <= .3)]
            if not feasible.empty:
                assert threshold == feasible.sort_values(['cost', 'threshold']).iloc[0].threshold
        assert set(held.split) == {'validation', 'test', 'shifted_test', 'confirmation_test'}
    else:
        assert threshold == sweep.sort_values(['cost', 'threshold']).iloc[0].threshold
        assert set(held.split) == {'validation', 'test', 'shifted_test'}
    assert len(pd.read_csv(ART/'model_comparison_validation.csv')) >= 4
