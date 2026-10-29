"""Phase 1.5: fixed-data training, validation-F1 threshold and immutable bundle.

python -m ml.retrain --iterations 8
No simulator regeneration, label redefinition, or test-driven selection.
"""
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
import platform
import uuid

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,precision_recall_curve,roc_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer,StandardScaler

from ml.bundle import ROOT,ACTIVE,digest,active_model_directory
from ml.features import EngagementFeaturesV15
from ml.model import metrics,prepare_input
from ml.synthetic import FEATURES,BOUNDS
from ml.tune import families,group_temporal_folds
from ml.train import save_json

SEED=42


def split_inputs():
    old=ROOT/'artifacts/synthetic_previous'
    dataset=ROOT/'data/synthetic/food_delivery.jsonl'
    frame=pd.read_json(dataset,lines=True)
    frame['date']=pd.to_datetime(frame.date,utc=True)
    manifest=pd.read_csv(old/'split_manifest.csv')
    parts={name:frame[frame.post_key.isin(manifest[manifest.split==name].post_key)].sort_values(['date','post_key']).reset_index(drop=True)
           for name in ['train','validation','test']}
    target=json.loads((old/'label_definition.json').read_text())
    if float(parts['train'].retweet_count.quantile(.9)) != target['retweet_threshold']:
        raise ValueError('Training target differs from the preserved definition.')
    for part in parts.values():
        part['label']=(part.retweet_count>=target['retweet_threshold']).astype(int)
    return frame,parts,target


def audit(directory,frame,parts,target):
    train=parts['train']
    latent=pd.read_csv(ROOT/'data/synthetic/food_delivery_latent_audit.csv').set_index('post_key').loc[train.post_key]
    noise_terms={name:float(np.var(np.log(latent[name].clip(lower=1e-12)))) for name in
                 ['platform_amplification','late_noise','shock_multiplier']}
    logrt=np.log1p(train.retweet_count)
    correlations={name:float(np.corrcoef(np.log1p(train[name]),logrt)[0,1]) for name in
                  ['followers_count','first_hour_impressions','first_hour_likes','first_hour_retweets']}
    overlaps={a+'_'+b:dict(posts=len(set(parts[a].post_key)&set(parts[b].post_key)),
        authors=len(set(parts[a].user_key)&set(parts[b].user_key))) for a,b in [('train','validation'),('train','test'),('validation','test')]}
    if any(item['posts'] or item['authors'] for item in overlaps.values()):
        raise ValueError('Partition overlap found.')
    for a,b in [('train','validation'),('validation','test')]:
        if parts[a].date.max()>=parts[b].date.min():
            raise ValueError('Chronology is not strictly separated.')
    forbidden={'retweet_count','text','reference_topic','reference_sentiment','label'}
    assert not forbidden&set(FEATURES)
    source_hashes=json.loads((ROOT/'artifacts/synthetic_previous/run.json').read_text())
    assert digest(ROOT/'data/synthetic/food_delivery.jsonl')==source_hashes['data_sha256']
    data=dict(dataset='data/synthetic/food_delivery.jsonl',dataset_rows=len(frame),
        dataset_sha256=digest(ROOT/'data/synthetic/food_delivery.jsonl'),generator_sha256=digest(ROOT/'ml/synthetic.py'),
        generator_manifest=json.loads((ROOT/'data/synthetic/simulation_manifest.json').read_text()),
        duplicate_post_ids=int(frame.post_key.duplicated().sum()),duplicate_authors=int(frame.user_key.duplicated().sum()),
        exact_duplicate_predictor_rows=int(frame[FEATURES].duplicated().sum()),partition_overlap=overlaps,
        final_less_than_first_hour=int((frame.retweet_count<frame.first_hour_retweets).sum()),
        target_threshold=target['retweet_threshold'],tie_rule='>=; include all boundary ties',features=FEATURES,
        partitions={name:dict(rows=len(part),positives=int(part.label.sum()),negatives=int((part.label==0).sum()),
            prevalence=float(part.label.mean()),start=str(part.date.min()),end=str(part.date.max())) for name,part in parts.items()},
        noise_analysis_training_only=dict(log_noise_variance=noise_terms,latent_appeal_log_variance=float(np.var(.5*latent.latent_appeal)),
            shock_prevalence=float(latent.shock.mean()),log_feature_final_correlations=correlations,
            retweet_quantiles={str(q):float(train.retweet_count.quantile(q)) for q in [.5,.9,.95,.99,.999]},
            predictor_ap={feature:float(average_precision_score(train.label,train[feature])) for feature in ['followers_count','first_hour_retweets','first_hour_likes']}),
        findings=['No final outcome, reference labels or raw text in the predictor allowlist.',
          'All authors are unique: author-purged temporal CV is defensive, not evidence of repeated-author robustness.',
          'Post text/reference labels are sampled after engagement: text length/hashtags/mentions are deliberately weak controls, not semantic spread drivers.',
          'Unobserved appeal, amplification, heavy-tailed late noise, rare shocks and Gamma-Poisson counts create overlap; no measured Bayes performance bound is claimed.',
          'No verified historical-author engagement field exists; none is invented.',
          'Prior cost-sensitive probability threshold is not F1-optimal. New threshold maximizes validation F1.',
          'Original and confirmation tests were inspected in prior phases; this run cannot honestly call them untouched.',
          'Generator and dataset are retained: changing noise after seeing prior tests would compromise evaluation.'])
    save_json(directory/'audit.json',data)
    distributions=[]
    for feature in ['followers_count','first_hour_impressions','first_hour_likes','first_hour_retweets','account_age_days','retweet_count']:
        for label,part in train.groupby('label'):
            for q in [.1,.25,.5,.75,.9,.99]:
                distributions.append(dict(feature=feature,label=int(label),quantile=q,value=float(part[feature].quantile(q))))
    pd.DataFrame(distributions).to_csv(directory/'class_distributions.csv',index=False)
    return data


def calibration(y,p):
    labels=np.asarray(y)
    bins=np.minimum((p*10).astype(int),9)
    rows=[]
    for index in range(10):
        mask=bins==index
        if mask.any():
            rows.append(dict(bin=index,count=int(mask.sum()),mean_probability=float(p[mask].mean()),
                             fraction_positive=float(labels[mask].mean())))
    ece=sum(row['count']*abs(row['mean_probability']-row['fraction_positive']) for row in rows)/len(p)
    return rows,float(ece)


def train(iterations=8):
    timestamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    version='spread_v1_5_'+timestamp+'_'+uuid.uuid4().hex[:6]
    directory=ROOT/'artifacts/ml_versions'/version
    directory.mkdir(parents=True,exist_ok=False)
    prior=active_model_directory()
    prior_model=joblib.load(prior/'model.joblib')
    prior_threshold=json.loads((prior/'selection.json').read_text())['thresholds']['operational']
    frame,parts,target=split_inputs()
    audit_data=audit(directory,frame,parts,target)
    x={name:prepare_input(part,FEATURES,BOUNDS).to_numpy(dtype=float) for name,part in parts.items()}
    folds,fold_metadata=group_temporal_folds(parts['train'])
    candidates,comparison,searches={},[],[]
    # Retrain original-feature logistic baselines, not just reuse an old pickle.
    for c in [.1,1.0]:
        name=f'logistic_original_C{c}'
        model=Pipeline([('array',FunctionTransformer(np.asarray)),('impute',SimpleImputer(keep_empty_features=True)),
            ('log',FunctionTransformer(np.log1p)),('scale',StandardScaler()),('model',LogisticRegression(C=c,max_iter=1500,random_state=SEED))])
        model.fit(x['train'],parts['train'].label)
        candidates[name]=model
    specs,compatibility=families()
    for name,estimator,parameters in specs:
        if name=='catboost':
            continue  # Phase 1.5 requests the five compatible families, not another search.
        print('Training/searching '+name,flush=True)
        pipeline=Pipeline([('array',FunctionTransformer(np.asarray)),('impute',SimpleImputer(keep_empty_features=True)),
            ('features',EngagementFeaturesV15()),('scale',StandardScaler()),('model',estimator)])
        search=RandomizedSearchCV(pipeline,{'model__'+key:value for key,value in parameters.items()},
            n_iter=iterations,scoring='average_precision',cv=folds,random_state=SEED,n_jobs=1,
            refit=True,error_score='raise',return_train_score=True)
        search.fit(x['train'],parts['train'].label)
        candidates[name]=search.best_estimator_
        results=pd.DataFrame(search.cv_results_)
        results.to_csv(directory/f'cv_{name}.csv',index=False)
        searches.append(dict(family=name,cv_ap=float(search.best_score_),params=search.best_params_,
            train_cv_gap=float(results.loc[search.best_index_,'mean_train_score']-search.best_score_)))
    for name,model in candidates.items():
        p=model.predict_proba(x['validation'])[:,1]
        comparison.append(dict(model=name,**metrics(parts['validation'].label,p)))
    table=pd.DataFrame(comparison).sort_values(['average_precision','model'],ascending=[False,True])
    selected=str(table.iloc[0].model)
    model=candidates[selected]
    p=model.predict_proba(x['validation'])[:,1]
    sweep=pd.DataFrame([dict(threshold=float(t),**metrics(parts['validation'].label,p,t)) for t in np.round(np.arange(.01,1,.01),2)])
    threshold=float(sweep.sort_values(['f1','threshold'],ascending=[False,False]).iloc[0].threshold)
    thresholds=dict(default=.5,f1_optimal=threshold,
        f2_optimal=float(sweep.sort_values(['f2','threshold'],ascending=[False,False]).iloc[0].threshold),operational=threshold)
    selection=dict(model=selected,model_version=version,thresholds=thresholds,
        selection='Largest validation average precision; exact ties by model name ascending.',
        operational_rule='Maximum validation F1 on 0.01–0.99 grid; exact ties choose higher threshold.',
        trained_at_utc=datetime.now(timezone.utc).isoformat(),phase='1.5')
    save_json(directory/'selection.json',selection)
    joblib.dump(model,directory/'model.joblib')
    save_json(directory/'freeze.json',dict(model_sha256=digest(directory/'model.joblib'),threshold=threshold,
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),test_evaluation_started=False))
    table.to_csv(directory/'model_comparison_validation.csv',index=False)
    sweep.to_csv(directory/'thresholds_validation.csv',index=False)
    save_json(directory/'training_search.json',dict(seed=SEED,iterations=iterations,folds=fold_metadata,searches=searches,compatibility=compatibility))
    # Final selected-model test prediction is computed once, after the freeze.
    reports,held,cal_rows={},[],[]
    for split in ['train','validation','test']:
        part=parts[split]
        probability=model.predict_proba(x[split])[:,1]
        reports[split]={key:metrics(part.label,probability,value) for key,value in thresholds.items()}
        bins,ece=calibration(part.label,probability)
        reports[split]['calibration_ece']=ece
        cal_rows += [dict(split=split,**row) for row in bins]
        if split!='train':
            output=part[['post_key','user_key','date','retweet_count','label']].copy()
            output['split']=split; output['probability']=probability;output['prediction']=(probability>=threshold).astype(int)
            held.append(output)
        if split=='test':
            precision,recall,pr_thresholds=precision_recall_curve(part.label,probability)
            pd.DataFrame(dict(precision=precision,recall=recall)).to_csv(directory/'precision_recall_curve.csv',index=False)
            fpr,tpr,_=roc_curve(part.label,probability)
            pd.DataFrame(dict(false_positive_rate=fpr,true_positive_rate=tpr)).to_csv(directory/'roc_curve.csv',index=False)
            test_probability=probability
    pd.concat(held).to_csv(directory/'held_out_predictions.csv',index=False)
    pd.DataFrame(cal_rows).to_csv(directory/'calibration.csv',index=False)
    importance=permutation_importance(model,x['validation'],parts['validation'].label,scoring='average_precision',n_repeats=5,random_state=SEED)
    pd.DataFrame(dict(feature=FEATURES,importance=importance.importances_mean,std=importance.importances_std)).to_csv(directory/'feature_importance.csv',index=False)
    save_json(directory/'metrics.json',reports)
    save_json(directory/'confusion_matrix.json',{key:reports['test']['operational'][key] for key in ['tn','fp','fn','tp']})
    schema=dict(source='synthetic',features=FEATURES,bounded=BOUNDS,
        defaults={name:float(parts['train'][name].median()) for name in FEATURES},
        feature_order=FEATURES,engineering_version='EngagementFeaturesV15 or original log-only baseline; serialized in model pipeline')
    save_json(directory/'feature_schema.json',schema)
    save_json(directory/'label_definition.json',target)
    save_json(directory/'status.json',dict(status='ready',source='synthetic',phase='1.5',model_version=version))
    pd.concat([part[['post_key','user_key','date']].assign(split=name) for name,part in parts.items()]).to_csv(directory/'split_manifest.csv',index=False)
    previous_p=prior_model.predict_proba(x['test'])[:,1]
    previous_metrics=metrics(parts['test'].label,previous_p,prior_threshold)
    save_json(directory/'previous_vs_new.json',dict(previous=dict(directory=str(prior.relative_to(ROOT)),threshold=prior_threshold,metrics=previous_metrics),
        new=dict(version=version,threshold=threshold,metrics=reports['test']['operational']),comparable=True,
        conditions='Same original test posts, labels and target; operational thresholds may differ. Test was already inspected in earlier phases.'))
    test=parts['test'].copy();test['probability']=test_probability
    test['error']=np.where((test_probability>=threshold)&(test.label==0),'false_positive',
        np.where((test_probability<threshold)&(test.label==1),'false_negative','correct'))
    test[['post_key','followers_count','first_hour_retweets','first_hour_likes','label','probability','error']].to_csv(directory/'error_analysis.csv',index=False)
    run=dict(version=version,seed=SEED,python=platform.python_version(),dataset_version='simulation_v1',dataset_sha256=audit_data['dataset_sha256'],
        versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn','joblib','plotly','streamlit']},
        evaluation_timestamp_utc=datetime.now(timezone.utc).isoformat(),final_test_reused_from_earlier_phases=True,
        final_test_evaluations_this_run=1,generator_changed=False,phase3_executed=False)
    save_json(directory/'run.json',run)
    lines=['# Phase 1.5 final model report','',json.dumps(selection,indent=2),'','## Audit findings','']+audit_data['findings']
    lines += ['','## Training-only noise analysis','',json.dumps(audit_data['noise_analysis_training_only'],indent=2),
        '', '## Validation candidate comparison (offline only)','',table.to_csv(index=False),
        '', '## Final test metrics', '',json.dumps(reports['test'],indent=2),
        '', '## Previous versus new: same original test', '',json.dumps(json.loads((directory/'previous_vs_new.json').read_text()),indent=2),
        '', '## Interpretation', '',
        'No Bayes-optimal performance ceiling has been established. Predictive early engagement exists, but unobserved late noise, amplification and rare shocks create unavoidable overlap for these inputs. The generator was not edited.',
        'Search train-to-CV AP gaps diagnose overfitting; calibration ECE and Brier quantify probability quality. The final test is reused from earlier phases, so these results are not a claim of a newly untouched evaluation.',
        'F1 threshold selection is validation-only. Dashboard test-threshold exploration is descriptive and never writes model thresholds.',
        'Exact counts, distributions, feature order, data hashes, fitting parameters and curve arrays are in this immutable bundle.']
    (directory/'EVALUATION_REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    manifest=dict(version=version,files={path.name:digest(path) for path in directory.iterdir() if path.is_file()})
    save_json(directory/'bundle_manifest.json',manifest)
    # Publish only a fully built version; old artifacts remain untouched.
    pointer=dict(directory=str(directory.relative_to(ROOT)),version=version,manifest_sha256=digest(directory/'bundle_manifest.json'))
    temporary=ACTIVE.with_suffix('.tmp')
    save_json(temporary,pointer);temporary.replace(ACTIVE)
    print(json.dumps(dict(directory=str(directory),selected=selected,threshold=threshold,metrics=reports['test']['operational']),indent=2),flush=True)
    return directory


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--iterations',type=int,default=8)
    args=parser.parse_args()
    train(args.iterations)
