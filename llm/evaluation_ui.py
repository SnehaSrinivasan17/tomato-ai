"""Canonical saved-run views and independent human annotation in the existing app."""
import json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from llm.annotation import build_candidates,save_annotation,agreement,annotations
from llm.evaluation import RUNS
from llm.schemas import Classification,Sentiment,Topic,Urgency,Severity,Language,SafetyFlag
from ml.charts import categorical_confusion,layout
from ml.display import display_table


def plot_classes(data,title,key):
    if not data:
        st.info(title+': no reference labels available.');return
    fig=go.Figure()
    for field in ['precision','recall']:
        fig.add_trace(go.Bar(x=[str(d.get('label',d.get('flag'))) for d in data],y=[float(d[field]) for d in data],name=field.title()))
    fig.update_layout(barmode='group')
    st.plotly_chart(layout(fig,title,'Reference class','Score'),use_container_width=True,key=key)


def annotation_ui():
    st.subheader('Human reference annotation')
    rows=build_candidates()
    reviewer=st.text_input('Reviewer identifier')
    independent=st.checkbox('Independent review: hide provisional labels',value=True)
    post_id=st.selectbox('Annotation post ID',[r['post_id'] for r in rows])
    row=next(r for r in rows if r['post_id']==post_id)
    st.write(row['text']);st.caption('Split: '+row['split']+' · '+row['source'])
    if not independent:
        st.json(row['provisional_labels'])
    # Never prepopulate with model predictions or another reviewer's annotation.
    with st.form('annotation_'+post_id):
        values={}
        for field,enum in [('sentiment',Sentiment),('topic',Topic),('urgency',Urgency),('suggested_severity',Severity),('language',Language)]:
            values[field]=st.selectbox('Reference '+field,[e.value for e in enum],key='ann_'+post_id+field)
        values['secondary_topics']=st.multiselect('Reference secondary topics',[e.value for e in Topic])
        values['safety_flags']=st.multiselect('Reference safety flags',[e.value for e in SafetyFlag])
        for field in ['sarcasm_detected','response_recommended','ambiguous']:
            values[field]=st.checkbox('Reference '+field,key='ann_'+post_id+field)
        evidence=st.text_area('Exact evidence phrases (one per line)')
        values['brief_reason']=st.text_area('Reference reasoning')
        notes=st.text_area('Reviewer notes')
        approved=st.checkbox('I reviewed every field and approve this reference',value=False)
        if st.form_submit_button('Save annotation revision'):
            values['evidence_phrases']=[v.strip() for v in evidence.splitlines() if v.strip()]
            try:
                if any(v not in row['text'] for v in values['evidence_phrases']): raise ValueError('Evidence must occur in the post.')
                if values['topic'] in values['secondary_topics']: raise ValueError('Primary topic cannot be repeated.')
                version=save_annotation(post_id,values,reviewer,notes,approved)
                st.success(f'Saved revision {version}; human_verified={approved}.')
            except ValueError as error:
                st.error(str(error))
    reviews=annotations()
    st.write('Approved reviewer annotations:',sum(r['human_verified'] for r in reviews))
    pairs=agreement()
    if pairs: st.dataframe(display_table(pd.DataFrame(pairs)),use_container_width=True)
    else: st.caption('Inter-annotator agreement pending independent overlapping reviews.')


def render(legacy,try_it):
    st.title('LLM Evaluation')
    mode=st.selectbox('Evaluation source',['Current canonical runs','Historical GPT-OSS provisional benchmark'])
    if mode.startswith('Historical'):
        st.caption('Historical prompt and schema · provisional synthetic references · not human-verified')
        legacy();return
    sections=st.tabs(['Model performance','Evaluation graphs','Error analysis','Try It Yourself','Human annotation'])
    with sections[4]: annotation_ui()
    with sections[3]: try_it()
    paths=sorted(RUNS.glob('*/report.json'),reverse=True) if RUNS.exists() else []
    if not paths:
        with sections[0]: st.info('Canonical evaluation pending. No completed evaluation report found. Run python -m llm.evaluation.')
        return
    with sections[0]:
        run_name=st.selectbox('Saved evaluation run',[p.parent.name for p in paths],key='canonical_run_name_v3')
        path=next((p for p in paths if p.parent.name==run_name),paths[0])
        try: report=json.loads(path.read_text(encoding='utf8'))
        except (OSError,ValueError): st.error('Saved evaluation artifact is missing or invalid.');return
        meta=report['metadata']
        st.caption(f"{meta['model_id']} · prompt {meta['prompt_version']} · dataset {meta['dataset_version']}")
        with st.expander('Attempt history and resume audit'):
            st.json(report.get('attempt_history',{}))
            st.caption('Metrics use latest record outcomes. Transport failures can be retried; schema failures remain frozen. Prior attempts remain in the checkpoint.')
        split=st.selectbox('Evaluation partition',['heldout','development','all'])
        category=st.selectbox('Reference category',['provisional','human_verified'])
        data=report['categories'][split][category]
        if data is None:
            st.info('Human-verified evaluation pending approved reference annotations and a new evaluation report.')
        else:
            st.write('Evaluation samples:',data['sample_count'],'· completed:',data.get('completed_count',data['sample_count']),'· reference-labelled:',data['reference_count'],'· status:',category)
            st.write('Call outcomes:',data.get('error_counts',{}))
            if data.get('pending_count',0): st.info(f"Evaluation in progress: {data['pending_count']} calls pending; displayed metrics cover attempted calls only.")
            fields=data['fields'];cards=[('Sentiment accuracy',fields.get('sentiment',{}).get('accuracy')),('Sentiment macro F1',fields.get('sentiment',{}).get('macro_f1')),('Topic macro F1',fields.get('topic',{}).get('macro_f1')),('Urgency macro F1',fields.get('urgency',{}).get('macro_f1')),('Critical safety recall',data.get('critical_safety_recall')),('Valid JSON + schema',data.get('json_validity_rate'))]
            columns=st.columns(3)
            for i,(label,value) in enumerate(cards): columns[i%3].metric(label,'Pending' if value is None else f'{value:.1%}')
            summary=[dict(field=k,**{x:v[x] for x in ['n','accuracy','macro_f1','weighted_f1']}) for k,v in fields.items()]
            st.dataframe(display_table(pd.DataFrame(summary)),use_container_width=True)
            st.write('Safety micro precision / recall:',data.get('safety_micro_precision'),data.get('safety_micro_recall'))
            st.write('API failure rate:',data.get('api_failure_rate'),'· average / P95 latency (ms):',data.get('average_latency_ms'),data.get('p95_latency_ms'))
            st.caption('Field denominators differ where existing English posts have only partial reference labels. Critical recall requires critical urgency and every expected safety flag; synthetic provisional scores are not verified accuracy.')
    if data is None: return
    with sections[1]:
        for field in ['sentiment','urgency']:
            saved=data['fields'].get(field)
            if not saved: st.info(field+' references unavailable.');continue
            matrix=saved['matrix'];frame=pd.DataFrame(matrix['counts'],index=matrix['labels'],columns=matrix['labels'])
            st.plotly_chart(categorical_confusion(frame,field.title()+' confusion matrix'),use_container_width=True,key='canonical_'+field+'_cm')
            plot_classes(saved['classes'],field.title()+' precision and recall','canonical_'+field+'_pr')
        plot_classes(data['fields'].get('topic',{}).get('classes',[]),'Topic performance','topic_pr')
        plot_classes(data['safety'],'Multi-label safety flag performance','safety_pr')
        language=data['by_language']
        fig=go.Figure(go.Bar(x=[r['language'] for r in language],y=[r['sentiment_accuracy'] for r in language],text=[str(r['n'])+' posts' for r in language]))
        st.plotly_chart(layout(fig,'Sentiment accuracy by language','Language','Accuracy'),use_container_width=True,key='language_accuracy')
        plot_classes(data['fields'].get('sarcasm_detected',{}).get('classes',[]),'Sarcasm performance','sarcasm_pr')
        fig=go.Figure(go.Histogram(x=data['latencies_ms'],nbinsx=20))
        st.plotly_chart(layout(fig,'Observed inference latency · uncached calls','Milliseconds','Calls'),use_container_width=True,key='latency')
    with sections[2]:
        errors=pd.DataFrame(data['errors'])
        st.caption(meta['model_id']+' · prompt '+meta['prompt_version'])
        if errors.empty: st.info('No errors against available references in this category.')
        else:
            for field in ['language','topic','urgency']:
                selected=st.multiselect('Filter '+field,sorted(errors[field].astype(str).unique().tolist()))
                if selected: errors=errors[errors[field].isin(selected)]
            if st.checkbox('Safety-critical errors only'): errors=errors[errors.safety_critical]
            st.dataframe(display_table(errors.astype(str)),use_container_width=True)
        st.write('Critical cases missed:',data['critical_misses'])
