"""Run with python -m ml.train. No credentials or external APIs required."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from ml.data import FEATURES, load_data, split_data
from ml.model import metrics, prepare_input

SEED = 42


def excluded_field_reason(name):
    if name in {'retweet_count', 'favorite_count', 'quote_count', 'reply_count', 'favorited', 'retweeted'}:
        return 'Final engagement or collection-state outcome; retweet_count is used only as the label.'
    if name in {'geo', 'coordinates', 'place', 'synthetic_geo', 'synthetic_geo_city', '_original_geo', 'created_at_ist'}:
        return 'Synthetic or unsupported geographic/timezone overlay; not evidence of actual location effects.'
    if name in {'text', 'extended_tweet', 'entities', 'display_text_range', 'truncated', 'lang', 'source'}:
        return 'Content, content-derived metadata or posting client excluded from the metadata-only allowlist.'
    if name in {'id', 'id_str', 'in_reply_to_user_id', 'in_reply_to_user_id_str', 'in_reply_to_screen_name', 'contributors'}:
        return 'Identifier or identity attribute; post ID is used only for deduplication and hashed audit.'
    if name in {'created_at', 'timestamp_ms', 'user', 'is_quote_status', 'in_reply_to_status_id', 'in_reply_to_status_id_str'}:
        return 'Raw field excluded; only allowlisted numeric derivatives or user metadata are used. Timestamp also defines splits.'
    return 'Unvalidated field outside explicit predictor allowlist; excluded conservatively.'


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def dataset_digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def train(data_path, output, cohort='originals', source='historical'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    features = FEATURES
    bounds = {'hour_utc': [0, 23], 'weekday': [0, 6], 'verified': [0, 1], 'is_reply': [0, 1], 'is_quote': [0, 1]}
    ood = None
    if source == 'synthetic':
        from ml.synthetic import generate, FEATURES as SYNTHETIC_FEATURES, BOUNDS, LIMITATIONS
        frame, ood = generate()
        features, bounds = SYNTHETIC_FEATURES, BOUNDS
        data_path = 'data/synthetic/food_delivery.jsonl'
        audit, observed = dict(generated_posts=len(frame), shifted_test_posts=len(ood)), list(frame.columns)
        cohort = 'synthetic_food_delivery'
    else:
        frame, audit, observed = load_data(data_path, cohort)
    parts = dict(zip(['train', 'validation', 'test'], split_data(frame)))
    if ood is not None:
        parts['shifted_test'] = ood
    label_threshold = float(parts['train'].retweet_count.quantile(.9, interpolation='linear'))
    for part in parts.values():
        part['label'] = (part.retweet_count >= label_threshold).astype(int)
    split_info = {name: dict(rows=len(part), users=int(part.user_key.nunique()),
                   start=str(part.date.min()), end=str(part.date.max()), positives=int(part.label.sum()),
                   nonzero_retweet_posts=int((part.retweet_count > 0).sum()),
                   prevalence=float(part.label.mean())) for name, part in parts.items()}
    save_json(output / 'data_report.json', dict(audit=audit, splits=split_info, cohort=cohort,
        excluded_after_temporal_split=len(frame)-sum(len(p) for p in parts.values()),
        split_rule='60/20/20 temporal boundaries; purge prior-partition users from validation and test; no shuffling.',
        duration_hours=(frame.date.max()-frame.date.min()).total_seconds()/3600,
        excluded_top_level_fields={key: excluded_field_reason(key) for key in observed},
        limitations=['Final observed retweets, not a one-hour forecast.', 'Historical Twitter sample is not representative of Indian food-delivery complaints.',
        'Indian geographic overlays are synthetic and excluded.', 'User-disjoint chronological evaluation also serves as grouped-user robustness evaluation.',
        'Profile counters are snapshot metadata; publication-time provenance is unverified and may change after publication.',
        'A narrow historical window cannot establish temporal generalization.', 'Cost ratio is a prototype assumption, not an empirically measured business cost.',
        'Repost retweet_count may describe the source post and include engagement already present at repost publication; repost results are diagnostics only.']))
    if source == 'synthetic':
        save_json(output / 'data_report.json', dict(audit=audit, splits=split_info, cohort=cohort,
            excluded_after_temporal_split=len(frame)-sum(len(parts[n]) for n in ['train', 'validation', 'test']),
            split_rule='Chronological 60/20/20 before labelling; independent fictional authors; shifted test separately generated with seed+1.',
            duration_hours=(frame.date.max()-frame.date.min()).total_seconds()/3600,
            limitations=LIMITATIONS,
            excluded_top_level_fields={name: 'Text, reference labels, outcome, identifier or simulation provenance; never a predictor.'
                                      for name in observed if name not in features},
            characteristics=frame[features+['retweet_count']].describe(percentiles=[.5,.9,.99]).to_dict(),
            small_account_positive_posts=int(((frame.followers_count < 500) & (frame.retweet_count >= label_threshold)).sum()),
            simulation_manifest='data/synthetic/simulation_manifest.json'))
        parts['test'].head(12).to_csv(output / 'fictional_post_examples.csv', index=False)
    manifests = pd.concat([part[['post_key', 'user_key', 'date']].assign(split=name) for name, part in parts.items()])
    manifests.to_csv(output / 'split_manifest.csv', index=False)
    save_json(output / 'label_definition.json', dict(quantile=.9, retweet_threshold=label_threshold,
        rule=('Positive when final simulated retweet_count >= training 90th percentile; all boundary ties positive.' if source == 'synthetic'
              else 'Positive when final observed retweet_count >= training 90th percentile; all boundary ties positive.'),
        prevalence={name: item['prevalence'] for name, item in split_info.items()}))
    save_json(output / 'run.json', dict(seed=SEED, simulation_seed=20261029 if source == 'synthetic' else None,
        source=source, cohort=cohort, python=platform.python_version(),
        data_sha256=dataset_digest(data_path),
        shifted_data_sha256=dataset_digest('data/synthetic/shifted_test.jsonl') if source == 'synthetic' else None,
        versions={name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'joblib', 'streamlit', 'plotly']}))
    invalid = [name for name, part in parts.items() if part.label.nunique() != 2]
    if invalid:
        reason = ('Degenerate labels in ' + ', '.join(invalid) + '. Training 90th percentile = '
                  + str(label_threshold) + '. Boundary ties prevent a meaningful top-decile model. '
                  'No model trained; require original-post data with varied, consistently measured outcomes.')
        save_json(output / 'status.json', dict(status='unavailable', reason=reason, cohort=cohort))
        print(json.dumps(dict(status='unavailable', reason=reason, splits=split_info), indent=2))
        return False
    save_json(output / 'status.json', dict(status='training', cohort=cohort))
    x = {name: prepare_input(part, features, bounds) for name, part in parts.items()}
    candidates = [('dummy', DummyClassifier(strategy='prior'))]
    candidates += [(f'logistic_C{c}', LogisticRegression(C=c, max_iter=1000, random_state=SEED)) for c in [.1, 1.0]]
    candidates += [(f'random_forest_leaf{leaf}', RandomForestClassifier(n_estimators=150, min_samples_leaf=leaf,
                    max_depth=14, n_jobs=-1, random_state=SEED)) for leaf in [5, 20]]
    candidates += [(f'hist_gradient_leaf{leaf}', HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=leaf,
                    l2_regularization=2, random_state=SEED)) for leaf in [7, 15]]
    comparison, fitted = [], {}
    for name, estimator in candidates:
        pipeline = make_pipeline(FunctionTransformer(np.asarray), SimpleImputer(strategy='median', keep_empty_features=True),
                                 FunctionTransformer(np.log1p), StandardScaler(), estimator)
        pipeline.fit(x['train'], parts['train'].label)
        probability = pipeline.predict_proba(x['validation'])[:, 1]
        comparison.append(dict(model=name, **metrics(parts['validation'].label, probability)))
        fitted[name] = pipeline
    table = pd.DataFrame(comparison).sort_values(['average_precision', 'model'], ascending=[False, True])
    selected = table[table.model != 'dummy'].iloc[0].model
    model = fitted[selected]
    probability = model.predict_proba(x['validation'])[:, 1]
    sweep = pd.DataFrame([dict(threshold=float(t), **metrics(parts['validation'].label, probability, t))
                          for t in np.round(np.arange(.05, .951, .01), 2)])
    # One missed high-spread post costs five false alerts. Fixed before test access.
    sweep['cost'] = 5 * sweep.fn + sweep.fp
    operational = float(sweep.sort_values(['cost', 'threshold']).iloc[0].threshold)
    thresholds = {'default': .5, 'f1_optimal': float(sweep.loc[sweep.f1.idxmax(), 'threshold']),
                  'f2_optimal': float(sweep.loc[sweep.f2.idxmax(), 'threshold']), 'operational': operational}
    # Freeze model/threshold prior to calculating any test prediction or metric.
    joblib.dump(model, output / 'model.joblib')
    save_json(output / 'selection.json', dict(model=selected, thresholds=thresholds,
        selection='Highest validation average precision; ties broken by model name.',
        operational_rule='Minimize validation 5*FN + FP; ties choose lower threshold.'))
    table.to_csv(output / 'model_comparison_validation.csv', index=False)
    sweep.to_csv(output / 'thresholds_validation.csv', index=False)
    reports, calibration, predictions = {}, [], []
    for split in (['validation', 'test', 'shifted_test'] if ood is not None else ['validation', 'test']):
        part = parts[split]
        proba = model.predict_proba(x[split])[:, 1]
        reports[split] = {name: metrics(part.label, proba, t) for name, t in thresholds.items()}
        held = part[['post_key', 'user_key', 'date', 'retweet_count', 'label']].copy()
        held['probability'], held['split'] = proba, split
        held['prediction'] = (proba >= operational).astype(int)
        predictions.append(held)
        fraction, mean = calibration_curve(part.label, proba, n_bins=10, strategy='uniform')
        calibration += [dict(split=split, mean_probability=float(a), fraction_positive=float(b)) for a, b in zip(mean, fraction)]
    pd.concat(predictions).to_csv(output / 'held_out_predictions.csv', index=False)
    pd.DataFrame(calibration).to_csv(output / 'calibration.csv', index=False)
    importance = permutation_importance(model, x['validation'], parts['validation'].label,
                                       scoring='average_precision', n_repeats=5, random_state=SEED)
    pd.DataFrame(dict(feature=features, importance=importance.importances_mean,
                      std=importance.importances_std)).to_csv(output / 'feature_importance.csv', index=False)
    save_json(output / 'metrics.json', reports)
    save_json(output / 'feature_schema.json', dict(features=features, source=source,
        defaults={name: float(x['train'][name].median()) if x['train'][name].notna().any() else 0.0 for name in features},
        types={name: 'nonnegative numeric' for name in features},
        bounded=bounds,
        missing='Invalid or missing values are imputed using training medians; wholly missing features use zero.'))
    save_json(output / 'status.json', dict(status='ready', source=source, cohort=cohort, diagnostic_only=cohort == 'reposts'))
    (output / 'requirements-training.txt').write_text('\n'.join(
        f'{name}=={importlib.metadata.version(name)}' for name in
        ['numpy', 'pandas', 'scikit-learn', 'scipy', 'joblib', 'streamlit', 'plotly']) + '\n', encoding='utf-8')
    print(json.dumps(dict(selected=selected, label_threshold=label_threshold, operational_threshold=operational,
                         splits=split_info, test=reports['test']['operational'],
                         shifted_test=reports.get('shifted_test', {}).get('operational')), indent=2))
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/raw/spread_data_india.json')
    parser.add_argument('--output')
    parser.add_argument('--source', choices=['synthetic', 'historical'], default='synthetic')
    parser.add_argument('--cohort', choices=['originals', 'reposts'], default='originals')
    args = parser.parse_args()
    if args.output is None:
        args.output = 'artifacts/synthetic_ml' if args.source == 'synthetic' else 'artifacts/ml'
    if args.source == 'synthetic' and args.output in ['artifacts/ml', 'artifacts/repost_diagnostic']:
        parser.error('Keep historical reports intact: synthetic training requires a separate output directory.')
    if args.source == 'historical' and args.cohort == 'reposts' and args.output == 'artifacts/ml':
        parser.error('Repost diagnostics require a separate --output directory, e.g. artifacts/repost_diagnostic')
    train(args.data, args.output, args.cohort, args.source)
