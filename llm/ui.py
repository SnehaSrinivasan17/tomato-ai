"""Incremental Streamlit sections; existing ML dashboard remains in app.py."""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import get_setting
from llm.classifier import classify_post
from llm.config import classifier_model
from llm.storage import store_live_text
from llm.combined import combined_analysis, review_decision
from llm.evaluate import BENCHMARK, load_benchmark, evaluate_records
from ml.display import display_table
from ml.inference import SpreadPredictor
from ml.bundle import active_model_directory
from ml.charts import categorical_confusion,layout


def classification_output(result):
    if hasattr(result,'classification'):
        if result.status != 'success':
            st.error(result.error_message)
            return
        metadata={key:result.model_dump(mode='json')[key] for key in ['post_id','model_id','prompt_version','latency_ms','cache_hit','token_usage']}
        result=result.classification.model_dump(mode='json')
        st.caption(f"{metadata['model_id']} · prompt {metadata['prompt_version']} · {metadata['latency_ms']:.0f} ms · cache {metadata['cache_hit']}")
    columns=st.columns(3)
    for index,field in enumerate(['sentiment','topic','urgency','suggested_severity','language','ambiguous']):
        columns[index%3].metric(field.replace('_',' ').title(),str(result.get(field)))
    st.write('Secondary topics:', ', '.join(result.get('secondary_topics',[])) or 'None')
    st.write('Safety flags:', ', '.join(result.get('safety_flags',[])) or 'None')
    st.write('Sarcasm detected:',result.get('sarcasm_detected'))
    st.write('Response recommended:',result.get('response_recommended'))
    st.write('Evidence phrases:',result.get('evidence_phrases',[]))
    st.write(result.get('brief_reason',''))
    with st.expander('Validated classification JSON'):
        st.json(result)


@st.cache_data(show_spinner=False)
def existing_ml_posts():
    path=Path(__file__).resolve().parents[1]/'data/synthetic/food_delivery.jsonl'
    return pd.read_json(path,lines=True)[['post_key','text']].rename(columns={'post_key':'post_id'}).to_dict('records')


def classification_page():
    st.title('LLM Classification')
    st.caption('Existing AIAgent · Groq · '+classifier_model()+' · English / Hindi / Hinglish')
    source=st.radio('Input method',['Enter a new post','Select existing ML post'],horizontal=True)
    post_id=None
    if source=='Select existing ML post':
        rows=existing_ml_posts()
        post_id=st.selectbox('Existing post ID',[r['post_id'] for r in rows])
        post=next(r['text'] for r in rows if r['post_id']==post_id)
        st.write(post)
    else:
        post=st.text_area('Social media post',key='llm_post',height=150)
    save=st.checkbox('Save redacted live post text locally',value=False)
    if st.button('Classify post',type='primary'):
        with st.spinner('Running the existing Groq classifier…'):
            result=classify_post(post,post_id)
        st.session_state['llm_last_v2']=(post,result)
        if result.status=='success' and save and post_id is None:
            store_live_text(result.post_id,post)
    last=st.session_state.get('llm_last_v2')
    if last and last[0]==post:
        classification_output(last[1])


def legacy_evaluation_page():
    st.title('LLM Evaluation')
    reference, predictions = load_benchmark()
    # Imported results remain authoritative files, not constants in the UI.
    metrics = pd.read_csv(BENCHMARK/'field_metrics.csv')
    errors = pd.read_csv(BENCHMARK/'error_analysis.csv')
    _, _, _, matrices = evaluate_records(reference, predictions)
    st.caption(f'Saved Groq benchmark · {len(reference)} reference-labelled synthetic posts · live submissions excluded')
    st.dataframe(display_table(metrics), use_container_width=True)
    figure=go.Figure()
    for column,name in [('accuracy','Accuracy'),('macro_f1','F1')]:
        figure.add_trace(go.Bar(x=metrics[column].astype(float).tolist(),y=metrics.field.astype(str).tolist(),
            name=name,orientation='h'))
    figure.update_layout(barmode='group')
    st.plotly_chart(layout(figure,'Benchmark accuracy and F1 by field','Score','Classification field'),
        use_container_width=True,key='llm_field_metrics')
    field = st.selectbox('Classification field', list(matrices))
    matrix = pd.DataFrame(matrices[field]).T
    st.plotly_chart(categorical_confusion(matrix,field.replace('_',' ').title()+' confusion matrix'),
        use_container_width=True,key='llm_confusion')
    st.caption(f'Rows: reference labels · Columns: predictions · {int(matrix.to_numpy().sum())} benchmark posts')
    st.subheader('Saved benchmark error analysis')
    st.dataframe(display_table(errors), use_container_width=True)
    with st.expander('Original evaluation report'):
        st.markdown((BENCHMARK/'evaluation_summary.md').read_text(encoding='utf-8'))


def evaluation_page():
    from llm.evaluation_ui import render
    render(legacy_evaluation_page,classification_page)


def combined_page():
    st.title('Combined Analysis')
    st.caption('Independent LLM triage and existing synthetic spread model · observed first-hour inputs')
    post = st.text_area('Social media post', key='combined_post', height=120)
    try:
        directory=active_model_directory()
        schema = json.loads((directory/'feature_schema.json').read_text())
        definition = json.loads((directory/'label_definition.json').read_text())
    except (OSError, ValueError):
        st.error('The existing ML feature schema is unavailable. Standalone LLM Classification remains available.')
        return
    st.write('Enter measured metadata. Blank fields leave ML prediction unavailable; no training-median defaults are substituted here.')
    metadata = {}
    columns = st.columns(3)
    for index, feature in enumerate(schema['features']):
        bounds = schema['bounded'].get(feature)
        suffix = f' ({bounds[0]}–{bounds[1]})' if bounds else ''
        with columns[index % 3]:
            metadata[feature] = st.text_input(feature.replace('_', ' ').title()+suffix,
                key='combined_'+feature, placeholder='Required; enter 0 if observed zero')
    if st.button('Run combined analysis', type='primary'):
        with st.spinner('Running independent classifications…'):
            result = combined_analysis(post, metadata)
        st.session_state['combined_last'] = dict(result=result, text=post, metadata=metadata.copy())
    last = st.session_state.get('combined_last')
    if not last or last['text'] != post or last['metadata'] != metadata:
        return
    result = last['result']
    left, right = st.columns(2)
    with left:
        st.subheader('LLM output')
        if result['llm']:
            classification_output(result['llm'])
        else:
            st.error(result['llm_error'])
    with right:
        st.subheader('ML spread output')
        if result['ml']:
            ml = result['ml']
            st.metric('Viral probability · synthetic model', f"{ml['probability']:.1%}")
            st.write('Prediction:', 'Viral' if ml['high_spread'] else 'Non-viral')
            st.write('Operational probability threshold:', ml['operational_threshold'])
        else:
            st.write('ML prediction unavailable: provide valid values for every required field.')
            with st.expander('Metadata validation details'):
                for error in result['metadata_errors']:
                    st.write(error)
        st.write(f"Virality definition: final simulated retweets ≥ {definition['retweet_threshold']:g}; training-only 90th percentile, ties included.")
    st.subheader('Deterministic decision')
    st.caption('The same versioned Python engine used by the end-to-end workflow.')
    st.write('Safety escalation remains independent of spread; missing values remain unknown.')
    st.json(result['review'])
    st.caption('Python selects the action through the shared decision engine. Missing predictions remain unknown; publication is disabled.')


def render_page(page):
    {'LLM Classification': classification_page, 'LLM Evaluation': evaluation_page,
     'Combined Analysis': combined_page}[page]()
