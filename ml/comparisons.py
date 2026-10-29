"""Separate, read-only comparisons of actual saved model experiments."""
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ml.bundle import ROOT,active_model_directory,validate_bundle
from ml.charts import confusion,layout
from ml.display import display_table

LABELS={'accuracy':'Accuracy','precision':'Precision','recall':'Recall','f1':'F1',
    'f2':'F2','average_precision':'PR-AUC (AP)','roc_auc':'ROC-AUC','brier':'Brier score',
    'balanced_accuracy':'Balanced accuracy'}


def experiment_table(experiment,split='test'):
    if experiment=='latest':
        directory=active_model_directory()
        validate_bundle(directory)
        frame=pd.read_csv(directory/'model_comparison_validation.csv')
        frame['threshold']=.5  # The training report recorded metrics at the default threshold.
        return frame,'Latest retraining · validation · all candidate metrics at threshold 0.50'
    path=ROOT/'artifacts/synthetic_ml/model_comparison_test.csv'
    frame=pd.read_csv(path)
    return frame[frame.split==split].copy(),'Earlier tuning · '+split+' · each candidate’s saved operational threshold'


def render():
    st.title('ML Model Comparisons')
    st.caption('Saved experiment results · Read-only · The deployed model is unchanged')
    source=st.radio('Experiment',['Latest retraining · validation','Earlier tuning · held-out'])
    experiment='latest' if source.startswith('Latest') else 'earlier'
    split='test'
    if experiment=='earlier':
        labels={'Original test':'test','Confirmation test':'confirmation_test','Shifted simulation':'shifted_test'}
        choice=st.selectbox('Evaluation set',list(labels))
        split=labels[choice]
    try:
        table,description=experiment_table(experiment,split)
    except (OSError,ValueError,KeyError) as error:
        st.error('Unable to load model comparison artifacts: '+str(error))
        return
    st.write(description)
    st.caption('The latest validation experiment and earlier held-out experiment are separate runs. Compare models within the same run/split; their thresholds and feature pipelines can differ.')
    table=table.sort_values(['average_precision','model'],ascending=[False,True])
    columns=['model','threshold',*LABELS,'tn','fp','fn','tp']
    st.dataframe(display_table(table[[column for column in columns if column in table]]),use_container_width=True)
    metric_name=st.selectbox('Comparison metric',list(LABELS.values()))
    metric=next(key for key,value in LABELS.items() if value==metric_name)
    order=table.sort_values(metric,ascending=metric!='brier')
    figure=go.Figure(go.Bar(x=order[metric].astype(float).tolist(),y=order.model.astype(str).tolist(),
        orientation='h',text=[f'{value:.3f}' for value in order[metric]],textposition='auto'))
    st.plotly_chart(layout(figure,metric_name+' across tested models',metric_name,'Model'),
        use_container_width=True,key='model_metric_comparison')
    selected=st.selectbox('Inspect model',table.model.astype(str).tolist())
    row=table[table.model==selected].iloc[0]
    st.subheader(selected.replace('_',' ').title())
    cards=st.columns(3)
    for column,metric in zip(cards,['precision','recall','f1']):
        column.metric(LABELS[metric],f'{row[metric]:.3f}')
    st.plotly_chart(confusion(row.to_dict(),selected+' · confusion matrix'),
        use_container_width=True,key='candidate_confusion')
    if experiment=='latest':
        path=active_model_directory()/'training_search.json'
        if path.exists():
            details=json.loads(path.read_text())
            st.subheader('Training-only cross-validation')
            scores=pd.DataFrame([dict(model=item['family'],cv_average_precision=item['cv_ap'],
                train_cv_gap=item['train_cv_gap'],best_parameters=json.dumps(item['params'])) for item in details['searches']])
            st.dataframe(display_table(scores),use_container_width=True)
    st.caption('Opening this page never retrains a model, changes the active version, or changes deployment thresholds.')
