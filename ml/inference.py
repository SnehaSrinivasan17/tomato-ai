"""Shared integrated prediction entry point; never falls back to historical models."""
import json
from pathlib import Path

import joblib

from ml.model import prepare_input
from ml.bundle import active_model_directory,validate_bundle

# Compatibility alias; active_model_directory() is re-read for every default load.
MODEL_DIRECTORY = active_model_directory()


class SpreadPredictor:
    def __init__(self, directory=None):
        directory = Path(directory) if directory is not None else active_model_directory()
        self.directory=directory
        if (directory/'bundle_manifest.json').exists():
            validate_bundle(directory)
        status = json.loads((directory/'status.json').read_text())
        if status.get('status') != 'ready' or status.get('source') != 'synthetic':
            raise ValueError('A ready synthetic model is required. Run python -m ml.train.')
        self.schema = json.loads((directory/'feature_schema.json').read_text())
        self.selection = json.loads((directory/'selection.json').read_text())
        # Trusted local artifacts only; joblib loading can execute code.
        self.model = joblib.load(directory/'model.joblib')

    def prepare(self, values):
        return prepare_input(values, self.schema['features'], self.schema['bounded'])

    def predict(self, values):
        probability = float(self.model.predict_proba(self.prepare(values))[0, 1])
        threshold = self.selection['thresholds']['operational']
        return dict(probability=probability, high_spread=bool(probability >= threshold),
                    operational_threshold=threshold, model_source='synthetic', observation_window_hours=1)
