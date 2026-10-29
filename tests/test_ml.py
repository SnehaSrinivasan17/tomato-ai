import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.data import FEATURES, load_data, split_data
from ml.model import metrics, prepare_input

ROOT = Path(__file__).resolve().parents[1] / 'artifacts' / 'repost_diagnostic'


class PipelineTests(unittest.TestCase):
    def test_allowlist_excludes_outcomes_and_content(self):
        self.assertTrue(set(FEATURES).isdisjoint({'retweet_count', 'favorite_count', 'reply_count', 'quote_count', 'text', 'user_key', 'post_key'}))
        self.assertEqual(list(prepare_input({'retweet_count': 100}).columns), FEATURES)

    def test_input_validation(self):
        row = prepare_input({'followers_count': -1, 'hour_utc': 25, 'verified': 'bad'})
        self.assertTrue(row.isna().all().all())

    def test_artifacts_and_probabilities(self):
        model = joblib.load(ROOT / 'model.joblib')
        for value in [{}, {'followers_count': 200}, {'hour_utc': 999}]:
            p = model.predict_proba(prepare_input(value))
            self.assertEqual(p.shape, (1, 2))
            self.assertTrue(np.isfinite(p).all())
            self.assertAlmostEqual(float(p.sum()), 1)
        for path in ROOT.glob('*.json'):
            json.loads(path.read_text())

    def test_splits_and_label_threshold(self):
        manifest = pd.read_csv(ROOT / 'split_manifest.csv')
        sets = {name: set(part.user_key) for name, part in manifest.groupby('split')}
        self.assertFalse(sets['train'] & sets['validation'])
        self.assertFalse(sets['train'] & sets['test'])
        self.assertFalse(sets['validation'] & sets['test'])
        self.assertEqual(manifest.post_key.nunique(), len(manifest))
        dates = {name: pd.to_datetime(part.date) for name, part in manifest.groupby('split')}
        self.assertLess(dates['train'].max(), dates['validation'].min())
        self.assertLess(dates['validation'].max(), dates['test'].min())
        held = pd.read_csv(ROOT / 'held_out_predictions.csv')
        threshold = json.loads((ROOT / 'label_definition.json').read_text())['retweet_threshold']
        self.assertTrue(((held.retweet_count >= threshold).astype(int) == held.label).all())
        report = json.loads((ROOT / 'metrics.json').read_text())
        decision = json.loads((ROOT / 'selection.json').read_text())['thresholds']['operational']
        test = held[held.split == 'test']
        actual = metrics(test.label, test.probability, decision)
        for name, value in actual.items():
            if name == 'accuracy' and name not in report['test']['operational']:
                continue  # Historical artifacts predate explicit accuracy reporting.
            self.assertAlmostEqual(value, report['test']['operational'][name])

    def test_streaming_bad_records_and_deduplication(self):
        row = dict(id=1, created_at='Fri Jul 27 19:14:17 +0000 2018', retweet_count=2,
                   user=dict(id=3, created_at='Fri Jul 27 19:14:17 +0000 2018'))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.jsonl'
            path.write_text('\n'.join([json.dumps(row), json.dumps(row), 'bad', json.dumps(dict(row, id=2, retweet_count=-1))]))
            frame, audit, _ = load_data(path)
        self.assertEqual(len(frame), 1)
        self.assertEqual(audit['duplicate_post'], 1)
        self.assertEqual(audit['invalid_json'], 1)
        self.assertEqual(audit['invalid_target_or_date'], 1)

    def test_primary_is_honestly_unavailable(self):
        root = ROOT.parent / 'ml'
        status = json.loads((root / 'status.json').read_text())
        self.assertEqual(status['status'], 'unavailable')
        report = json.loads((root / 'data_report.json').read_text())
        self.assertEqual(report['splits']['train']['nonzero_retweet_posts'], 0)
        self.assertFalse((root / 'model.joblib').exists())

    def test_reposts_deduplicate_original_source(self):
        row = dict(id=1, created_at='Fri Jul 27 19:14:17 +0000 2018', retweet_count=2,
                   retweeted_status=dict(id=100), user=dict(id=3))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'reposts.jsonl'
            path.write_text('\n'.join([json.dumps(row), json.dumps(dict(row, id=2))]))
            frame, audit, _ = load_data(path, 'reposts')
        self.assertEqual(len(frame), 1)
        self.assertEqual(audit['duplicate_original_source'], 1)

    def test_training_only_target_and_operational_selection(self):
        data, _, _ = load_data(ROOT.parents[1] / 'data' / 'raw' / 'spread_data_india.json', 'reposts')
        training, _, _ = split_data(data)
        target = json.loads((ROOT / 'label_definition.json').read_text())
        self.assertAlmostEqual(float(training.retweet_count.quantile(.9)), target['retweet_threshold'])
        sweep = pd.read_csv(ROOT / 'thresholds_validation.csv')
        selection = json.loads((ROOT / 'selection.json').read_text())
        expected = sweep.sort_values(['cost', 'threshold']).iloc[0].threshold
        self.assertAlmostEqual(expected, selection['thresholds']['operational'])
        self.assertTrue(((5*sweep.fn+sweep.fp) == sweep.cost).all())
        self.assertEqual(len(sweep), 91)

    def test_streamlit_page_and_prediction(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(ROOT.parents[1] / 'app.py'), default_timeout=60).run()
        app.sidebar.radio[0].set_value('ML Spread Lab').run()
        self.assertEqual(len(app.exception), 0)
        app.slider[0].set_value(.5).run()
        self.assertEqual(len(app.exception), 0)
        app.button[-1].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any(item.label == 'Probability of synthetic high-spread class' for item in app.metric))
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)


if __name__ == '__main__':
    unittest.main()
