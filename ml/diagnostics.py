"""Training-only distribution/noise/learning diagnostics; no reselection."""
import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.base import clone
from sklearn.metrics import average_precision_score

from ml.bundle import ROOT,active_model_directory
from ml.charts import layout
from ml.model import prepare_input
from ml.synthetic import FEATURES,BOUNDS
from ml.tune import group_temporal_folds


def write_diagnostics():
    directory=active_model_directory()
    audit=json.loads((directory/'audit.json').read_text())
    frame=pd.read_json(ROOT/'data/synthetic/food_delivery.jsonl',lines=True)
    manifest=pd.read_csv(directory/'split_manifest.csv')
    train=frame[frame.post_key.isin(manifest[manifest.split=='train'].post_key)].sort_values(['date','post_key']).reset_index(drop=True)
    train['label']=(train.retweet_count>=audit['target_threshold']).astype(int)
    latent=pd.read_csv(ROOT/'data/synthetic/food_delivery_latent_audit.csv').set_index('post_key').loc[train.post_key]
    visible=np.log(2+1.8*train.first_hour_retweets+.25*(train.first_hour_likes+1)**.8+.15*(train.followers_count+1)**.35)+.35*train.has_media
    noise=np.log(latent.platform_amplification)+np.log(latent.late_noise)+np.log(latent.shock_multiplier)
    noise_analysis=dict(visible_log_mean_variance=float(np.var(visible)),independent_late_log_noise_variance=float(np.var(noise)),
        caveat='Variance components concern the specified log-mean mechanism, not an estimated irreducible classification error bound.')
    plots=[]
    for feature in ['followers_count','first_hour_retweets','first_hour_likes','retweet_count']:
        fig=go.Figure()
        for label,name,color in [(0,'Ordinary','#4da6ff'),(1,'High spread','#ffad66')]:
            values=np.log1p(train.loc[train.label==label,feature]).astype(float).tolist()
            fig.add_trace(go.Histogram(x=values,name=name,histnorm='probability',opacity=.6,marker_color=color,nbinsx=35))
        fig.update_layout(barmode='overlay')
        plots.append(layout(fig,'Training class overlap: '+feature,'log(1 + '+feature+')','Fraction within class'))
    model=joblib.load(directory/'model.joblib')
    x=prepare_input(train,FEATURES,BOUNDS).to_numpy(dtype=float)
    folds,_=group_temporal_folds(train)
    fit,hold=folds[-1]
    learning=[]
    for size in [1000,3000,len(fit)]:
        indices=fit[:size]
        fitted=clone(model).fit(x[indices],train.iloc[indices].label)
        learning.append(dict(training_rows=len(indices),training_ap=float(average_precision_score(train.iloc[indices].label,fitted.predict_proba(x[indices])[:,1])),
            held_training_fold_ap=float(average_precision_score(train.iloc[hold].label,fitted.predict_proba(x[hold])[:,1]))))
    fig=go.Figure()
    for key in ['training_ap','held_training_fold_ap']:
        fig.add_trace(go.Scatter(x=[row['training_rows'] for row in learning],y=[row[key] for row in learning],mode='lines+markers',name=key))
    plots.append(layout(fig,'Training-only sample-size diagnostic','Training posts','Average precision'))
    out=ROOT/'phase15_results'
    out.mkdir(exist_ok=True)
    result=dict(model_version=directory.name,noise_analysis=noise_analysis,learning_curve=learning,
        selection_changed=False,all_diagnostics_use_training_only=True)
    (out/'noise_and_learning.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    html='<html><head><meta charset="utf-8"><title>Training-only ML diagnostics</title></head><body>'
    html+='<h1>Training-only diagnostics · '+directory.name+'</h1>'
    for index,figure in enumerate(plots):
        html+=figure.to_html(full_html=False,include_plotlyjs=index==0)
    html+='</body></html>'
    (out/'diagnostic_plots.html').write_text(html,encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    write_diagnostics()
