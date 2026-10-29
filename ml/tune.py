"""Tune on training CV, select on validation, freeze, then evaluate tests once.

python -m ml.tune --iterations 8
The original simulation, datasets, split membership and target cutoff are preserved.
"""
import argparse
import ast
import importlib.metadata
import importlib.util
import json
import platform
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from ml.features import EngagementFeatures
from ml.model import metrics, prepare_input
from ml.synthetic import FEATURES, BOUNDS, simulate
from ml.train import dataset_digest, save_json

SEED = 42
CONFIRMATION_SEED = 20261031


def group_temporal_folds(frame, n_splits=3):
    """Expanding time folds; purge users and boundary timestamps from validation."""
    folds, audit = [], []
    for train, valid in TimeSeriesSplit(n_splits=n_splits).split(frame):
        validation = frame.iloc[valid]
        keep = (~validation.user_key.isin(frame.iloc[train].user_key)) & (validation.date > frame.iloc[train].date.max())
        valid = valid[keep.to_numpy()]
        if len(valid) == 0 or frame.iloc[valid].label.nunique() != 2:
            raise ValueError('Insufficient independent validation groups/classes for CV.')
        folds.append((train, valid))
        audit.append(dict(train_rows=len(train), validation_rows=len(valid),
            train_end=str(frame.iloc[train].date.max()), validation_start=str(frame.iloc[valid].date.min()),
            overlapping_authors=len(set(frame.iloc[train].user_key)&set(frame.iloc[valid].user_key))))
    return folds, audit


def threshold_selection(y, probability):
    sweep = pd.DataFrame([dict(threshold=float(t), **metrics(y, probability, t))
                          for t in np.round(np.arange(.01, .951, .01), 2)])
    sweep['cost'] = 5*sweep.fn+sweep.fp
    eligible = sweep[(sweep.recall >= .5)&(sweep.alert_rate <= .3)]
    feasible = not eligible.empty
    if not feasible:
        eligible = sweep[sweep.alert_rate <= .3].sort_values(['recall', 'cost'], ascending=[False, True]).head(1)
    operational = float(eligible.sort_values(['cost', 'threshold']).iloc[0].threshold)
    return sweep, dict(default=.5, f1_optimal=float(sweep.loc[sweep.f1.idxmax(), 'threshold']),
        f2_optimal=float(sweep.loc[sweep.f2.idxmax(), 'threshold']), operational=operational), feasible


def families():
    specs = [
        ('logistic', LogisticRegression(max_iter=1500, random_state=SEED),
         dict(C=[.001,.01,.1,1,10,100], class_weight=[None,'balanced',{0:1,1:2},{0:1,1:4}])),
        ('random_forest', RandomForestClassifier(random_state=SEED,n_jobs=2),
         dict(n_estimators=[150,300], max_depth=[5,10,18,None], min_samples_leaf=[5,15,40,80],
              max_features=[.5,1.0,'sqrt'], class_weight=[None,'balanced_subsample',{0:1,1:2}])),
        ('hist_gradient', HistGradientBoostingClassifier(random_state=SEED),
         dict(max_iter=[100,200,300], learning_rate=[.025,.05,.1], max_leaf_nodes=[7,15,31],
              min_samples_leaf=[20,50,100], l2_regularization=[1,5,20])),
    ]
    availability = {}
    for module in ['xgboost','lightgbm','catboost']:
        try:
            if module == 'xgboost':
                from xgboost import XGBClassifier
                specs.append(('xgboost',XGBClassifier(random_state=SEED,n_jobs=2,tree_method='hist',eval_metric='logloss'),
                    dict(n_estimators=[100,200,350],max_depth=[2,3,5],learning_rate=[.025,.05,.1],
                         min_child_weight=[5,20,50],subsample=[.7,1],colsample_bytree=[.7,1],
                         reg_lambda=[1,5,20],scale_pos_weight=[1,2,5])))
            elif module == 'lightgbm':
                from lightgbm import LGBMClassifier
                specs.append(('lightgbm',LGBMClassifier(random_state=SEED,n_jobs=2,verbosity=-1,deterministic=True,force_col_wise=True),
                    dict(n_estimators=[100,200,350],num_leaves=[7,15,31],learning_rate=[.025,.05,.1],
                         min_child_samples=[20,50,100],reg_lambda=[1,5,20],class_weight=[None,'balanced',{0:1,1:2}])))
            else:
                from catboost import CatBoostClassifier
                specs.append(('catboost',CatBoostClassifier(random_seed=SEED,thread_count=2,verbose=False,allow_writing_files=False),
                    dict(iterations=[100,200,350],depth=[3,5,7],learning_rate=[.025,.05,.1],
                         l2_leaf_reg=[3,10,30],auto_class_weights=['None','Balanced','SqrtBalanced'])))
            availability[module] = dict(status='available',version=importlib.metadata.version(module))
        except ImportError as exc:
            availability[module] = dict(status='unavailable', reason=type(exc).__name__)
    return specs, availability


def tune(iterations=8, resume=False):
    root = Path('artifacts/synthetic_ml')
    previous = Path('artifacts/synthetic_previous')
    if not previous.exists():
        shutil.copytree(root, previous)
    # Refuse silently changed data or labels. No simulator regeneration occurs here.
    run = json.loads((previous/'run.json').read_text())
    assert dataset_digest('data/synthetic/food_delivery.jsonl') == run['data_sha256']
    assert dataset_digest('data/synthetic/shifted_test.jsonl') == run['shifted_data_sha256']
    frame = pd.read_json('data/synthetic/food_delivery.jsonl', lines=True)
    frame['date'] = pd.to_datetime(frame.date, utc=True)
    manifest = pd.read_csv(previous/'split_manifest.csv')
    parts = {split:frame[frame.post_key.isin(manifest[manifest.split==split].post_key)].sort_values(['date','post_key']).reset_index(drop=True)
             for split in ['train','validation','test']}
    target = json.loads((previous/'label_definition.json').read_text())
    cutoff = target['retweet_threshold']
    assert parts['train'].retweet_count.quantile(.9) == cutoff
    parts['shifted_test'] = pd.read_json('data/synthetic/shifted_test.jsonl',lines=True)
    for part in parts.values():
        part['label'] = (part.retweet_count >= cutoff).astype(int)
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not set(parts[a].user_key)&set(parts[b].user_key)
        assert parts[a].date.max() < parts[b].date.min()
    x = {split:prepare_input(part,FEATURES,BOUNDS).to_numpy(dtype=float) for split,part in parts.items()}
    folds, fold_audit = group_temporal_folds(parts['train'])
    old_model = joblib.load(previous/'model.joblib')
    models = {'previous_logistic':old_model}
    rows, searches, predictions = [], [], {}
    specs, availability = families()
    for name, estimator, params in specs:
        print('Tuning '+name,flush=True)
        pipeline = Pipeline([('array',FunctionTransformer(np.asarray)),('impute',SimpleImputer(keep_empty_features=True)),
            ('features',EngagementFeatures()),('scale',StandardScaler()),('model',estimator)])
        context = dict(data_sha256=run['data_sha256'],feature_sha256=dataset_digest('ml/features.py'),
            training_code_sha256=dataset_digest('ml/tune.py'),
            iterations=iterations,seed=SEED,folds=fold_audit,sklearn=importlib.metadata.version('scikit-learn'))
        context_path=root/f'cv_{name}_context.json'
        csv_path=root/f'cv_{name}.csv'
        cached = resume and csv_path.exists() and context_path.exists() and json.loads(context_path.read_text())==context
        if cached:
            print('Resuming completed CV '+name,flush=True)
            results=pd.read_csv(csv_path)
            best_index=int(results.mean_test_score.idxmax())
            best_params=ast.literal_eval(results.loc[best_index,'params'])
            best_model=pipeline.set_params(**best_params).fit(x['train'],parts['train'].label.to_numpy())
            best_score=float(results.loc[best_index,'mean_test_score'])
        else:
            search = RandomizedSearchCV(pipeline, {'model__'+k:v for k,v in params.items()},n_iter=iterations,
                scoring='average_precision',cv=folds,random_state=SEED,refit=True,n_jobs=1,
                error_score='raise',return_train_score=True)
            search.fit(x['train'],parts['train'].label.to_numpy())
            best_model,best_score,best_params,best_index=search.best_estimator_,float(search.best_score_),search.best_params_,search.best_index_
            results = pd.DataFrame(search.cv_results_)
        models[name] = best_model
        results['family'] = name
        results.to_csv(root/f'cv_{name}.csv',index=False)
        save_json(context_path,context)
        searches.append(dict(model=name,cv_ap=best_score,parameters=best_params,
            train_cv_gap=float(results.loc[best_index,'mean_train_score']-best_score)))
    thresholds, sweeps = {}, {}
    for name, model in models.items():
        p = model.predict_proba(x['validation'])[:,1]
        sweep, choices, feasible = threshold_selection(parts['validation'].label,p)
        if name == 'previous_logistic':
            choices = json.loads((previous/'selection.json').read_text())['thresholds']
        thresholds[name], sweeps[name] = choices, sweep
        cv_score = next((s['cv_ap'] for s in searches if s['model']==name),None)
        rows.append(dict(model=name,threshold=choices['operational'],cv_average_precision=cv_score,
            threshold_constraints_met=feasible,**metrics(parts['validation'].label,p,choices['operational'])))
    table = pd.DataFrame(rows).sort_values(['average_precision','model'],ascending=[False,True])
    selected = str(table.iloc[0].model)
    selected_model, choices = models[selected], thresholds[selected]
    # Models and all probability thresholds frozen before either old test or new confirmation evaluation.
    selection = dict(model=selected,thresholds=choices,selection='Highest validation average precision; tie by name; prior model eligible.',
        operational_rule='Validation only: minimize 5*FN+FP with recall >=0.50 and alert rate <=0.30; lower threshold breaks ties. If infeasible, maximize recall within alert cap. Previous model retains original threshold.',
        tuning='RandomizedSearchCV on three chronological author-purged training folds.',
        confirmation_seed=CONFIRMATION_SEED,iterations_per_family=iterations)
    save_json(root/'selection.json',selection)
    joblib.dump(selected_model,root/'model.joblib')
    table.to_csv(root/'model_comparison_validation.csv',index=False)
    sweeps[selected].to_csv(root/'thresholds_validation.csv',index=False)
    save_json(root/'tuning.json',dict(families=availability,searches=searches,folds=fold_audit,
        engineered_features='log1p counts; smoothed engagement/impression rates, reach-normalized impressions and retweets, engagement shares, list/follower ratio, cyclic UTC hour and weekday.',
        authors_are_unique=True, cv_note='All synthetic authors are unique, so author grouping is a defensive check rather than an independent repeated-author robustness test.',
        data_unchanged=True,target_unchanged=True,seed=SEED))
    confirmation_path = Path('data/synthetic/confirmation_test.jsonl')
    confirmation, _ = simulate(3000,CONFIRMATION_SEED)
    confirmation['post_key'] = confirmation.post_key.str.replace('syn-', 'confirm-',regex=False)
    confirmation['user_key'] = confirmation.user_key.str.replace('syn-', 'confirm-',regex=False)
    confirmation.to_json(confirmation_path,orient='records',lines=True,date_format='iso')
    confirmation['label'] = (confirmation.retweet_count >= cutoff).astype(int)
    parts['confirmation_test'] = confirmation
    x['confirmation_test'] = prepare_input(confirmation,FEATURES,BOUNDS).to_numpy(dtype=float)
    test_rows, reports, calibration, held = [], {}, [], []
    for split in ['validation','test','shifted_test','confirmation_test']:
        part = parts[split]
        probability = selected_model.predict_proba(x[split])[:,1]
        reports[split] = {key:metrics(part.label,probability,threshold) for key,threshold in choices.items()}
        output = part[['post_key','user_key','date','retweet_count','label']].copy()
        output['split'],output['probability'] = split,probability
        output['prediction'] = (probability >= choices['operational']).astype(int)
        held.append(output)
        fraction,mean = calibration_curve(part.label,probability,n_bins=10)
        calibration += [dict(split=split,mean_probability=float(a),fraction_positive=float(b)) for a,b in zip(mean,fraction)]
        if split != 'validation':
            for name,model in models.items():
                p = model.predict_proba(x[split])[:,1]
                test_rows.append(dict(split=split,model=name,threshold=thresholds[name]['operational'],
                    **metrics(part.label,p,thresholds[name]['operational'])))
            majority = np.full(len(part),float(parts['train'].label.mean()))
            test_rows.append(dict(split=split,model='majority_baseline',threshold=.5,**metrics(part.label,majority,.5)))
    pd.DataFrame(test_rows).to_csv(root/'model_comparison_test.csv',index=False)
    pd.concat(held).to_csv(root/'held_out_predictions.csv',index=False)
    pd.DataFrame(calibration).to_csv(root/'calibration.csv',index=False)
    save_json(root/'metrics.json',reports)
    importance = permutation_importance(selected_model,x['validation'],parts['validation'].label,
        scoring='average_precision',n_repeats=5,random_state=SEED,n_jobs=1)
    pd.DataFrame(dict(feature=FEATURES,importance=importance.importances_mean,std=importance.importances_std)).to_csv(root/'feature_importance.csv',index=False)
    # Error slices are descriptive only; they do not trigger retraining or model selection.
    error_rows = []
    for split in ['test','shifted_test','confirmation_test']:
        part = parts[split]
        p = selected_model.predict_proba(x[split])[:,1]
        for dimension, masks in {'followers':{'small':part.followers_count<500,'larger':part.followers_count>=500},
            'early_retweets':{'zero':part.first_hour_retweets==0,'nonzero':part.first_hour_retweets>0}}.items():
            for group,mask in masks.items():
                if mask.any():
                    error_rows.append(dict(split=split,dimension=dimension,group=group,rows=int(mask.sum()),
                        **metrics(part.loc[mask,'label'],p[mask],choices['operational'])))
    pd.DataFrame(error_rows).to_csv(root/'error_analysis.csv',index=False)
    train_metrics = metrics(parts['train'].label,selected_model.predict_proba(x['train'])[:,1],choices['operational'])
    save_json(root/'overfitting.json',dict(train=train_metrics,validation=reports['validation']['operational']))
    run.update(tuning_seed=SEED,confirmation_seed=CONFIRMATION_SEED,confirmation_sha256=dataset_digest(confirmation_path),
        python=platform.python_version(),search_iterations=iterations,
        versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn','joblib','xgboost','lightgbm','catboost'] if importlib.util.find_spec(name.replace('scikit-learn','sklearn'))})
    save_json(root/'run.json',run)
    (root/'requirements-training.txt').write_text('\n'.join(
        f'{name}=={importlib.metadata.version(name)}' for name in
        ['numpy','pandas','scipy','scikit-learn','joblib','streamlit','plotly','pyarrow','xgboost','lightgbm','catboost'])+'\n',encoding='utf8')
    # Historical and synthetic data reports remain accessible as documents, not dashboard warning panels.
    comparison = pd.DataFrame(test_rows)
    lines=['# Tuned synthetic spread evaluation','','The simulator, base datasets, outer splits and training label cutoff are unchanged.',
        f"The saved prior model scored {comparison[(comparison.model=='previous_logistic')&(comparison.split=='test')].iloc[0].accuracy:.2%} accuracy on the reused test; accuracy depends on the probability threshold.",
        'The majority baseline scores highly by missing every viral case. Models are not selected by test accuracy.',
        'The original test was previously inspected: its comparisons are reused-test results, not a fresh unbiased confirmation.',
        f'The confirmation set (seed {CONFIRMATION_SEED}) was generated only after model/threshold freezing; no test-driven reselection occurred.',
        '',f'Selected: {selected}; operational threshold {choices["operational"]}.',selection['operational_rule'],'',
        '## Actual held-out results','',comparison.to_csv(index=False),'','## Method and audits','',
        'RandomizedSearchCV uses average precision on three expanding chronological author-purged folds inside the original training split. Imputation, feature engineering and scaling are inside each fitted pipeline. No resampling occurs before or after splitting; imbalance is handled through searched class weights/positive weights.',
        'Candidates are selected by validation AP, with viral recall/alert-volume constraints on operational thresholds. The confirmation set is evaluated once after freezing. Shifted-test results remain separate.',
        'Authors are unique in this simulator, so the grouped CV check cannot establish repeated-author robustness. Fold dates and group-overlap counts are saved in tuning.json.',
        'Feature importance uses validation permutation AP; local explanations use median replacement. Error slices are reported after freezing and are not used for further tuning.',
        'Training scores and train-minus-CV AP gaps expose overfitting in overfitting.json and tuning.json. Final outcomes, text, topic, sentiment, urgency and simulator latent draws are never features.',
        'All results are synthetic; unobserved amplification, appeal and shocks limit discrimination. No 90–95% accuracy claim is warranted without meaningful viral recall. A single new simulation draw is not evidence of real-world validity.',
        'Raw historical reports remain in artifacts/ml and previous simulation details in docs/SYNTHETIC_SIMULATION.md. Deployed inference never falls back to repost diagnostics.',
        'Current Python and dependency versions are in run.json. Exact search results per family are in cv_*.csv.',
        '', '## Validation model comparison','',table.to_csv(index=False),'','## Error slices','',pd.DataFrame(error_rows).to_csv(index=False)]
    current_confirmation=reports['confirmation_test']['operational']
    majority_confirmation=comparison[(comparison.model=='majority_baseline')&(comparison.split=='confirmation_test')].iloc[0]
    lines += ['', '## Interpretation', '',
        f"The validation-selected model is {selected}. Confirmation accuracy is {current_confirmation['accuracy']:.2%}, "
        f"viral recall {current_confirmation['recall']:.2%}, precision {current_confirmation['precision']:.2%}, "
        f"F1 {current_confirmation['f1']:.4f}, AP {current_confirmation['average_precision']:.4f} and ROC-AUC {current_confirmation['roc_auc']:.4f}.",
        f"The majority baseline reaches {majority_confirmation.accuracy:.2%} accuracy with zero viral recall. "
        'The requested 90–95% accuracy with meaningful viral detection was not achieved. No test-driven model reselection was performed.',
        f"Selected-model training AP {train_metrics['average_precision']:.4f} versus validation AP "
        f"{reports['validation']['operational']['average_precision']:.4f}; the per-family train-to-CV gaps are recorded separately.",
        'First-hour retweets and likes dominate validation permutation importance. ' 
        'Small-account and zero-early-retweet false negatives are visible in the error slices; late shocks remain unobserved at prediction time.']
    (root/'EVALUATION_REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    save_json(root/'status.json',dict(status='ready',source='synthetic',cohort='synthetic_food_delivery',tuned=True))
    print(json.dumps(dict(selected=selected,threshold=choices['operational'],confirmation=reports['confirmation_test']['operational'],
        reused_test=reports['test']['operational']),indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--iterations',type=int,default=8)
    parser.add_argument('--resume',action='store_true',help='Reuse matching completed CV searches; refit family winners.')
    args=parser.parse_args()
    tune(args.iterations,args.resume)
