"""Final selected model only; all statistics/curves share a verified bundle."""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ml.bundle import ROOT,active_model_directory,validate_bundle
from ml.inference import SpreadPredictor
from ml.model import metrics
from ml.display import display_table
from ml.charts import build_figures,confusion,layout


def load_bundle(directory):
    directory=Path(directory)
    manifest=validate_bundle(directory)
    def read(name):
        return json.loads((directory/name).read_text(encoding='utf8'))
    result=dict(manifest=manifest,selection=read('selection.json'),schema=read('feature_schema.json'),
        metrics=read('metrics.json'),audit=read('audit.json'),target=read('label_definition.json'),
        pr=pd.read_csv(directory/'precision_recall_curve.csv'),roc=pd.read_csv(directory/'roc_curve.csv'),
        importance=pd.read_csv(directory/'feature_importance.csv'),calibration=pd.read_csv(directory/'calibration.csv'),
        predictions=pd.read_csv(directory/'held_out_predictions.csv'))
    for name,frame,columns in [('precision-recall',result['pr'],['recall','precision']),
        ('ROC',result['roc'],['false_positive_rate','true_positive_rate']),
        ('calibration',result['calibration'],['mean_probability','fraction_positive'])]:
        if frame.empty or not all(column in frame for column in columns) or frame[columns].isna().any().any():
            raise ValueError('Missing or invalid '+name+' chart data in final bundle.')
    return result


@st.cache_resource
def predictor_for_version(directory,manifest_hash):
    return SpreadPredictor(directory)


@st.cache_data(show_spinner=False)
def cached_bundle(directory,manifest_hash):
    return load_bundle(directory)


def render():
    st.title('Tomato AI · ML Spread Prediction Lab')
    st.caption('Final selected model · Synthetic food-delivery data · Prediction after the first hour')
    try:
        directory=active_model_directory()
        from ml.bundle import digest
        bundle=cached_bundle(str(directory),digest(directory/'bundle_manifest.json'))
        predictor=predictor_for_version(str(directory),bundle['manifest']['version'])
    except (OSError,ValueError,KeyError) as error:
        st.error('Unable to load the final-model dashboard: '+str(error))
        st.info('Build a complete version with: python -m ml.retrain --iterations 8')
        return
    selection,schema,audit=bundle['selection'],bundle['schema'],bundle['audit']
    final=bundle['metrics']['test']['operational']
    threshold=selection['thresholds']['operational']
    name=selection['model']
    display='Logistic Regression' if name.startswith('logistic') else name.replace('_',' ').title()
    st.subheader(display)
    st.caption('Model version: '+selection['model_version'])
    left,middle,right=st.columns(3)
    left.metric('Dataset posts',f"{audit['dataset_rows']:,}")
    middle.metric('Training positive class',f"{audit['partitions']['train']['prevalence']:.2%}")
    right.metric('Frozen operational threshold',f'{threshold:.2f}')
    st.write(f"High spread = final simulated retweets ≥ {bundle['target']['retweet_threshold']:g}, "
        'using the training-only 90th percentile with boundary ties included.')
    st.caption(f"Training {audit['partitions']['train']['rows']:,} · Validation {audit['partitions']['validation']['rows']:,} · Test {audit['partitions']['test']['rows']:,}")
    view=st.radio('Model view',['Final model performance','Threshold explorer','Try It Yourself','Historical dataset report'],horizontal=True)
    if view=='Final model performance':
        st.subheader('Frozen final-test results')
        for columns,fields in [(st.columns(3),['precision','recall','f1']),(st.columns(3),['f2','average_precision','roc_auc'])]:
            names=dict(precision='Precision',recall='Recall',f1='F1 Score',f2='F2 Score',average_precision='PR-AUC (AP)',roc_auc='ROC-AUC')
            for column,field in zip(columns,fields):
                column.metric(names[field],f'{final[field]:.3f}')
        st.caption(f"Balanced accuracy {final['balanced_accuracy']:.3f} · Brier {final['brier']:.3f} · Calibration ECE {bundle['metrics']['test']['calibration_ece']:.3f} · Accuracy {final['accuracy']:.1%}")
        try:
            figures=build_figures(bundle)
            # Each figure uses Python lists rather than Plotly 6 binary arrays.
            for index in range(0,len(figures),2):
                left,right=st.columns(2)
                with left:
                    st.plotly_chart(figures[index],use_container_width=True,key=f'final_{index}')
                with right:
                    st.plotly_chart(figures[index+1],use_container_width=True,key=f'final_{index+1}')
        except (ValueError,KeyError,TypeError) as error:
            st.error('Final-model chart rendering failed: '+str(error))
            raise
    if view=='Threshold explorer':
        st.subheader('Explore the fixed test predictions')
        st.write(f'The deployed threshold stays at {threshold:.2f}. This slider is descriptive test-set exploration; it does not change the saved model or establish a new validated threshold.')
        selected=st.slider('Exploratory probability threshold',.01,.99,float(threshold),.01)
        test=bundle['predictions'].query("split == 'test'")
        measured=metrics(test.label,test.probability,selected)
        columns=st.columns(4)
        for column,field in zip(columns,['precision','recall','f1','f2']):
            column.metric('Exploratory '+field.upper(),f'{measured[field]:.3f}')
        st.plotly_chart(confusion(measured,'Exploratory test confusion matrix'),use_container_width=True,key='explore_confusion')
    if view=='Try It Yourself':
        st.subheader('Author, post and observed first-hour metadata')
        features=schema['features']
        def reset():
            for feature in features:
                st.session_state['ml15_'+feature]=int(schema['defaults'][feature])
        st.button('Reset to training defaults',on_click=reset)
        with st.form('ml_prediction_inputs'):
            values={}
            columns=st.columns(3)
            for index,feature in enumerate(features):
                key='ml15_'+feature
                if key not in st.session_state:
                    st.session_state[key]=int(schema['defaults'][feature])
                bounds=schema['bounded'].get(feature)
                with columns[index%3]:
                    values[feature]=st.number_input(feature.replace('_',' ').title(),min_value=0,
                        max_value=int(bounds[1]) if bounds else None,step=1,key=key)
            submitted=st.form_submit_button('Predict high-spread probability',type='primary')
        if submitted:
            result=predictor.predict(values)
            st.metric('Probability of synthetic high-spread class',f"{result['probability']:.1%}")
            st.write('Classification:', 'High spread' if result['high_spread'] else 'Ordinary spread')
            st.write('Operational threshold:',result['operational_threshold'])
            effects=[]
            for feature in features:
                replacement=dict(values,**{feature:schema['defaults'][feature]})
                alternate=predictor.predict(replacement)['probability']
                effects.append(dict(feature=feature,difference=result['probability']-alternate))
            effects=pd.DataFrame(effects)
            effects=effects.loc[effects.difference.abs().nlargest(5).index].sort_values('difference')
            fig=go.Figure(go.Bar(x=effects.difference.astype(float).tolist(),y=effects.feature.tolist(),orientation='h'))
            st.plotly_chart(layout(fig,'Local sensitivity: replace one input with its training median','Probability difference','Feature'),use_container_width=True,key='local_sensitivity')
            st.caption('One-feature median replacements are sensitivity comparisons, not additive or causal contributions.')
    if view=='Historical dataset report':
        st.subheader('Historical original-post data · retained report')
        path=ROOT/'artifacts/ml/data_report.json'
        if path.exists():
            report=json.loads(path.read_text())
            st.write('The historical original-post training records had only zero retweets. The resulting percentile labels could not train a meaningful binary classifier. The archived repost diagnostic is not the primary model.')
            st.dataframe(display_table(pd.DataFrame(report['splits']).T),use_container_width=True)
            for limitation in report['limitations']:
                st.write('• '+limitation)
        else:
            st.error('Historical report is missing: artifacts/ml/data_report.json')
