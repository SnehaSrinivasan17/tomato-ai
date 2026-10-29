"""Demo feed, actual analytics and the single integrated Analyze Post workflow."""
import json
from collections import Counter
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ml.charts import layout
from ml.display import display_table
from src.demo.manifest import posts,prepare,DIRECTORY
from src.demo.runner import analyze,reset,all_feed,analytics,history,versions,fingerprint
from src.demo import runner
from src.review.store import get_decision
from src.decision.ui import show_result

def active_path():return runner.active_path()

def chart(counts,title,key):
    if counts:st.plotly_chart(layout(go.Figure(go.Bar(x=list(counts),y=list(counts.values()))),title,'Category','Posts'),use_container_width=True,key=key)
    else:st.info(title+': no processed labels yet.')

def analyze_view(key='unified'):
    available=posts();lookup={p['post_id']:p for p in available}
    mode=st.radio('Analyze Post input',['Existing demo post','Custom post'],horizontal=True,key=key+'_mode')
    if mode=='Existing demo post':
        selected=st.selectbox('Shared post ID',list(lookup),key=key+'_selected');post=lookup[selected]
        st.write(post['text']);st.caption(post['source']+' · original ID preserved')
    else:
        text=st.text_area('Post text · English, Hindi or Hinglish',key=key+'_text')
        metadata={}
        if st.checkbox('I have measured first-hour metadata',key=key+'_metadata_enabled'):
            from src.workflow.pipeline import predictor
            with st.expander('Enter every required feature; blanks remain missing',expanded=True):
                for feature in predictor().schema['features']:
                    value=st.text_input(feature,key=key+'_meta_'+feature)
                    if value.strip():metadata[feature]=value
        post=dict(post_id='CUSTOM_'+fingerprint(dict(text=text,metadata=metadata))[:24],text=text,metadata=metadata,source='custom submission')
    force=st.checkbox('Run live again · bypass cached pipeline and classifier output',key=key+'_force')
    if st.button('Analyze Post',type='primary',key=key+'_run'):
        if not post['text'].strip() or len(post['text'])>10000:st.error('Enter 1–10,000 characters.');return
        with st.spinner('Classifying → predicting spread where available → applying rules → retrieving and validating eligible drafts…'):
            try:st.session_state[key+'_last']=(fingerprint(post),str(active_path()),analyze(post,force=force))
            except Exception:st.error('Analysis could not complete. Check model artifacts, credentials or service availability; no outputs substituted.')
    last=st.session_state.get(key+'_last')
    if last and last[0]==fingerprint(post) and last[1]==str(active_path()):
        r=last[2];st.caption('Cached real pipeline result' if r['pipeline_cache_hit'] else 'New pipeline attempt · classifier cache status shown below')
        stored=get_decision(r['decision_id'],active_path());r=dict(r,review_status=stored['review_status'])
        show_result(r);st.write('Review status:',r['review_status'])
        if r.get('ml'):st.metric('High-spread probability',f"{r['ml']['probability']:.1%}");st.caption('Frozen threshold: '+str(r['ml']['operational_threshold']))
        else:st.write('Spread prediction unavailable · missing/invalid first-hour metadata is not replaced.')
        if (r.get('classification') or {}).get('status')!='success':st.error('Classification failed: '+str((r.get('classification') or {}).get('error_code','unavailable')))
        if (r.get('generation') or {}).get('status') in {'error','blocked'}:st.error('Draft unavailable: '+str(r['generation'].get('error_code') or r['generation']['status']))
        with st.expander('Versions, source references and processing details'):st.json(r.get('version_info'));st.json(r.get('errors',[]))

def overview(rows,key='overview'):
    a=analytics(rows)
    for labels in [[('Available posts','available'),('Analyzed posts','analyzed'),('Unprocessed','unprocessed'),('Failed / incomplete','failed_or_incomplete')],[('High spread','high_spread'),('Safety escalations','safety_escalations'),('Pending reviews','pending_review'),('Pipeline cache hits','cache_hits')]]:
        for col,(label,field) in zip(st.columns(4),labels):col.metric(label,a[field])
    st.caption('Counts use the latest stored attempt per post in the active demo session, including custom submissions. Earlier attempts remain in the audit history. Cache hits count reuse events, not new predictions.')
    st.caption(f"ML scores available: {a['ml_available']} of {a['analyzed']} analyzed posts · open review queue: {a['review_queue']}. Unavailable scores remain unknown.")
    chart(a['sentiment'],'Sentiment distribution',key+'_sentiment')
    return a

def filtered_feed(rows,key='feed'):
    frame=pd.DataFrame(rows)
    if frame.empty:st.info('No feed records.');return
    with st.expander('Filter feed'):
        columns=st.columns(4)
        for i,field in enumerate(['date','sentiment','language','topic','urgency','high_spread','action','review_status']):
            values=sorted({str(x)[:10] if field=='date' else str(x) for x in frame[field].dropna()})
            with columns[i%4]:selected=st.multiselect(field.replace('_',' ').title(),values,key=key+'_'+field)
            if selected:frame=frame[frame[field].map(lambda x:str(x)[:10] if field=='date' else str(x)).isin(selected)]
    st.dataframe(display_table(frame.drop(columns=['complete'])),use_container_width=True)

def render():
    st.title('Demo & Audit');st.caption('Fictional food-delivery posts · seven presentation days · original timestamps retained')
    manifest=prepare();available=posts();rows=all_feed(available)
    feed_tab,featured,analytics_tab,audit,reset_tab=st.tabs(['Seven-day feed','Featured scenarios','Demo analytics','Processing audit','Reset demo'])
    with feed_tab:
        st.write(f"{manifest['selected_originals']} original records + {len(available)-manifest['selected_originals']} separate language/safety examples")
        st.caption(manifest['presentation_mapping']);filtered_feed(rows)
    with featured:
        scenarios=json.loads((DIRECTORY/'scenarios.json').read_text());name=st.selectbox('Featured scenario',list(scenarios))
        scenario=scenarios[name];post=next(p for p in available if p['post_id']==scenario['post_id'])
        st.write(post['text']);st.caption('Expected behavior, not a prediction: '+scenario['expected_behavior']);st.write('Source:',post['source'],'· ID:',post['post_id'])
        live=st.checkbox('Rerun this scenario live',key='scenario_live')
        if st.button('Run selected scenario'):
            with st.spinner('Running the actual integrated pipeline…'):
                try:st.session_state['scenario_last']=(name,str(active_path()),analyze(post,force=live))
                except Exception:st.error('Scenario processing failed; no predictions fabricated.')
        last=st.session_state.get('scenario_last')
        if last and last[:2]==(name,str(active_path())):
            st.caption('Cached pipeline output' if last[2]['pipeline_cache_hit'] else 'New pipeline attempt');show_result(last[2])
    with analytics_tab:
        a=overview(rows,'demo');chart(a['topic'],'Topic distribution','demo_topic');chart(a['urgency'],'Urgency distribution','demo_urgency')
        for col,(label,k) in zip(st.columns(3),[('Approved drafts','approved'),('Rejected drafts','rejected'),('Edited drafts','edited')]):col.metric(label,a[k])
        st.write('Mean processing latency (ms):',round(a['mean_latency_ms'],1) if a['mean_latency_ms'] is not None else 'Unavailable')
    with audit:
        h=history();st.write('Actual processing attempts:',len(h))
        if h:
            st.dataframe(display_table(pd.DataFrame(h)),use_container_width=True)
            selected=st.selectbox('Inspect pipeline audit ID',[r['decision_id'] for r in h]);record=get_decision(selected,active_path());show_result(dict(record['pipeline'],decision_id=selected));st.json(record['events'])
        with st.expander('Current version contract'):
            try:st.json(versions())
            except (OSError,ValueError,KeyError):st.error('Version details unavailable. Check the active ML bundle, policy ingestion report and prompt files.')
    with reset_tab:
        st.write('Start a fresh demo session. Previous processing and review records are archived intact; source data, annotations, evaluations, model artifacts and Qdrant are untouched.')
        confirmed=st.checkbox('Start a new empty demo session')
        if st.button('Reset demo state',disabled=not confirmed):reset();st.success('New demo session started; previous history retained.');st.rerun()
