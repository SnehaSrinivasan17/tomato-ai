"""Offline benchmark evaluation using AIAgent's existing metric implementations.

python -m llm.evaluate
python -m llm.evaluate --live  # explicit new Groq benchmark run; separate outputs
"""
import argparse
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from llm.evaluation_core import (EXPECTED_SCHEMA_KEYS, compute_categorical_metrics,
    compute_boolean_metrics, load_dataset, build_confusion_matrix)

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / 'data' / 'llm' / 'benchmark'


def load_benchmark():
    comparison = pd.read_csv(BENCHMARK/'comparison_results.csv')
    ids = set(comparison.post_id)
    reference = load_dataset(str(BENCHMARK/'tomato_synthetic_posts.csv'))
    reference = [row for row in reference if row['post_id'] in ids]
    predictions = pd.read_csv(BENCHMARK/'groq_predictions.csv').fillna('').to_dict('records')
    predictions = [row for row in predictions if row['post_id'] in ids]
    return reference, predictions


def boolean(value):
    return value is True or str(value).strip().lower() in {'true', '1', 'yes'}


def evaluate_records(reference, predictions):
    by_id = {row['post_id']: row for row in predictions}
    metrics, comparisons, errors, matrices = [], [], [], {}
    for field in EXPECTED_SCHEMA_KEYS:
        refs, preds = [], []
        for ref in reference:
            pred = by_id.get(ref['post_id'])
            if pred is None:
                raise ValueError('Missing benchmark prediction; cannot silently drop records.')
            a, b = ref[field], pred[field]
            if field in {'needs_reply', 'sarcasm', 'human_review'}:
                a, b = boolean(a), boolean(b)
            else:
                a, b = str(a).strip().lower(), str(b).strip().lower()
            refs.append(a); preds.append(b)
            comparisons.append(dict(post_id=ref['post_id'], field=field, reference_value=str(a),
                                    predicted_value=str(b), is_match=a == b))
            if a != b:
                errors.append(dict(post_id=ref['post_id'], field=field, reference_label=str(a),
                                   predicted_label=str(b), post_text=ref['post_text']))
        function = compute_boolean_metrics if field in {'needs_reply', 'sarcasm', 'human_review'} else compute_categorical_metrics
        acc, precision, recall, f1, total, correct = function(refs, preds)
        metrics.append(dict(field=field, accuracy=acc, macro_precision=precision, macro_recall=recall,
                            macro_f1=f1, total_samples=total, correct_count=correct, error_count=total-correct))
        classes = sorted({str(value) for value in refs+preds})
        matrices[field] = build_confusion_matrix([str(v) for v in refs], [str(v) for v in preds], classes)
    return pd.DataFrame(metrics), pd.DataFrame(comparisons), pd.DataFrame(errors), matrices


def run_evaluation(live=False, output=None):
    reference, predictions = load_benchmark()
    if live:
        raise ValueError('Use python -m llm.evaluation for canonical live evaluation; legacy references remain separate.')
    metrics, comparisons, errors, matrices = evaluate_records(reference, predictions)
    directory = Path(output) if output else ROOT/'data/llm/evaluations'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(directory/'field_metrics.csv', index=False)
    comparisons.to_csv(directory/'comparison_results.csv', index=False)
    errors.to_csv(directory/'error_analysis.csv', index=False)
    (directory/'confusion_matrices.json').write_text(json.dumps(matrices, indent=2), encoding='utf-8')
    pd.DataFrame(predictions).to_csv(directory/'groq_predictions.csv', index=False)
    from config import get_setting
    provenance = json.loads((BENCHMARK/'provenance.json').read_text())
    summary = dict(mode='new_live_run' if live else 'saved_benchmark_recomputation',
        benchmark_posts=len(reference), benchmark_source=str(BENCHMARK),
        model=get_setting('GROQ_MODEL', 'openai/gpt-oss-120b') if live else provenance['model'],
        prompt_sha256=hashlib.sha256((ROOT/'llm/prompts/classifier_prompt.txt').read_bytes()).hexdigest())
    (directory/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    (directory/'evaluation_summary.md').write_text(
        '# LLM benchmark evaluation\n\n' + f"Mode: {summary['mode']}\n\nModel: {summary['model']}\n\n"
        + f"Reference-labelled posts: {len(reference)}. Live submissions excluded.\n\n"
        + '```csv\n' + metrics.to_csv(index=False) + '```\n\n'
        + 'Confusion matrices: confusion_matrices.json. Errors: error_analysis.csv. '
        + 'Categorical metrics are macro-averaged; boolean metrics use the positive class. '
        + 'Reference labels are the imported synthetic benchmark labels.\n', encoding='utf8')
    return metrics, directory


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    metrics, directory = run_evaluation(args.live, args.output)
    print(metrics.drop(columns=['correct_count']).to_string(index=False))
    print('Evaluation outputs:', directory)
