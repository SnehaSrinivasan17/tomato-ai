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

@st.cache_resource(ttl=10,max_entries=1,show_spinner=False)
def active_path():return runner.active_path()

@st.cache_data(ttl=15,max_entries=4,show_spinner=False)
def cached_feed(scope_key):
    return all_feed(posts())

def chart(counts,title,key):
    if counts:st.plotly_chart(layout(go.Figure(go.Bar(x=list(counts),y=list(counts.values()))),title,'Category','Posts'),use_container_width=True,key=key)
    else:st.info(title+': no processed labels yet.')

def public_frame(frame):
    return frame.drop(columns=[c for c in frame if c in {'source','post_id','decision_id','complete','language_source','original_timestamp','cache_key','version_fingerprint','processing_status','latency_ms','processed_at'}],errors='ignore')


def cell_color(value):
    text=str(value).upper()
    if text in {'P0','P1','URGENT_ESCALATION','ESCALATED','HIGH','NEGATIVE'}:return 'background-color:#fce8e6;color:#803932'
    if text in {'P2','PRIORITY_REVIEW','PENDING_REVIEW','MEDIUM','EDITED'}:return 'background-color:#fff2d8;color:#785c24'
    if text in {'P3','MONITOR','APPROVED','RESOLVED','POSITIVE','LOW'}:return 'background-color:#e6f3ed;color:#285f4d'
    return ''


def styled_table(frame):
    """Small escaped HTML tables avoid Styler dependencies and Arrow string errors."""
    from html import escape
    count=len(frame);visible=frame.head(50)
    header=''.join('<th>'+escape(str(c).replace('_',' ').title())+'</th>' for c in visible.columns)
    rows=[]
    for values in visible.itertuples(index=False,name=None):
        cells=[]
        for value in values:
            label='' if value is None else str(value)
            cells.append('<td style="'+cell_color(value)+'">'+escape(label)+'</td>')
        rows.append('<tr>'+''.join(cells)+'</tr>')
    st.markdown('<div style="overflow:auto;max-height:520px"><table class="tomato-table"><thead><tr>'+header+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>',unsafe_allow_html=True)
    if count>50:st.caption(f'Showing 50 of {count} messages. Use the feed filters to narrow the view.')


def result_view(result):
    from src.decision.ui import show_result
    show_result(result,compact=True)


def metadata_inputs(values,key):
    from src.workflow.pipeline import predictor
    schema=predictor().schema;metadata={}
    with st.expander('Virality predictor · author and first-hour engagement',expanded=True):
        with st.form(key+'_metadata_form'):
            columns=st.columns(3)
            for i,feature in enumerate(schema['features']):
                bounds=schema.get('bounded',{}).get(feature)
                with columns[i%3]:
                    metadata[feature]=st.number_input(feature.replace('_',' ').title(),min_value=0,max_value=int(bounds[1]) if bounds else None,value=int(values.get(feature,0)),step=1,key=key+'_'+feature)
            st.form_submit_button('Update spread prediction')
        from llm.combined import validate_metadata
        cleaned,errors=validate_metadata(metadata,schema)
        if errors:st.warning('; '.join(errors))
        else:
            spread=predictor().predict(cleaned)
            st.metric('Predicted high-spread probability',f"{spread['probability']:.1%}")
            st.caption(f"{'High spread' if spread['high_spread'] else 'Lower spread'} · operational threshold {spread['operational_threshold']:.0%} · synthetic model")
    return metadata


@st.cache_data(show_spinner=False,max_entries=1)
def example_catalog():
    from src.workflow.pipeline import predictor
    available=posts();lookup={p['post_id']:p for p in available};examples=[]
    scenarios=json.loads((DIRECTORY/'scenarios.json').read_text())
    for name,scenario in scenarios.items():
        post=dict(lookup[scenario['post_id']])
        post['metadata']=post.get('metadata') or available[0]['metadata']
        examples.append((name.replace('_',' ').title(),post,scenario['expected_behavior']))
    pool=[p for p in available if p.get('metadata')][:60]
    model=predictor()
    scores=model.model.predict_proba(model.prepare([p['metadata'] for p in pool]))[:,1].tolist()
    for label,index in [('Lower spread example',min(range(len(pool)),key=lambda i:scores[i])),('Higher spread example',max(range(len(pool)),key=lambda i:scores[i]))]:
        post=dict(pool[index]);examples.append((label,post,'Reference sentiment: '+str(post['metadata'].get('reference_sentiment','unknown'))))
    return examples


def analyze_view(key='unified'):
    available=posts()
    mode=st.radio('Analyze Post input',['Demo examples','Custom post'],horizontal=True,key=key+'_mode')
    if mode=='Demo examples':
        catalog=example_catalog()
        labels=[item[0]+' - '+item[1]['text'][:85] for item in catalog]
        chosen=st.selectbox('Choose an example',labels,key=key+'_selected')
        selected=labels.index(chosen)
        label,example,intent=catalog[selected];post=dict(example);st.write(post['text'])
        st.caption('Reference scenario intent: '+intent+' Priority is determined by analysis.')
        with st.expander('Compare demo examples'):
            from src.workflow.pipeline import predictor
            rows=[]
            saved={r['post_id']:r for r in cached_feed(str(active_path()))}
            for name,item,intent in catalog:
                score=predictor().predict(item['metadata'])['probability']
                rows.append({'Example':name,'Message':item['text'],'Reference intent':intent,'Sentiment':saved.get(item['post_id'],{}).get('sentiment') or 'Awaiting analysis','Priority':saved.get(item['post_id'],{}).get('priority') or 'Awaiting analysis','Spread probability':f"{score:.1%}"})
            styled_table(pd.DataFrame(rows))
        if not example.get('original_timestamp'):st.caption('Fictional demonstration metadata; not measured engagement for this language/safety example.')
        defaults=post.get('metadata')
        if not defaults:
            st.caption('Simulated demonstration metadata supplied for this language/safety example; these are not measured engagement.')
            defaults=available[0]['metadata']
        metadata=metadata_inputs(defaults,key+'_'+str(selected))
        post['metadata']=metadata
    else:
        text=st.text_area('Post text · English, Hindi or Hinglish',key=key+'_text')
        st.caption('Metadata is prefilled with a fictional example. Edit it to explore spread; values are simulated, not inferred from your text.')
        metadata=metadata_inputs(available[0]['metadata'],key+'_custom')
        post=dict(post_id='CUSTOM_'+fingerprint(dict(text=text,metadata=metadata))[:24],text=text,metadata=metadata,source='custom submission')
    force=st.checkbox('Refresh live analysis',key=key+'_force',help='Uses external services again. Otherwise reuse a matching successful analysis.')
    if st.button('Analyze Post',type='primary',key=key+'_run'):
        if not post['text'].strip() or len(post['text'])>10000:st.error('Enter 1–10,000 characters.');return
        with st.spinner('Analyzing the post and checking policy evidence…'):
            try:
                st.session_state[key+'_last']=(fingerprint(post),str(active_path()),analyze(post,force=force,path=active_path()))
                cached_feed.clear()
            except Exception as error:
                st.error('Analysis could not be saved or completed. Check database connectivity and service configuration, then retry. Error category: '+type(error).__name__)
    last=st.session_state.get(key+'_last')
    if last and last[0]==fingerprint(post) and last[1]==str(active_path()):
        r=last[2];stored=get_decision(r['decision_id'],active_path())
        result_view(dict(r,review_status=stored['review_status'] if stored else r.get('review_status')))
        if (r.get('classification') or {}).get('status')!='success':st.warning('Classification is unavailable. Retry after checking Groq credentials or quota.')
        if (r.get('retrieval') or {}).get('status')=='error':st.warning('Policy retrieval is unavailable. Check Qdrant and embedding service resources. Safety review remains active.')
        if (r.get('generation') or {}).get('status') in {'error','blocked'}:st.warning('A validated draft is unavailable. The review decision remains visible; retry if the service was temporarily unavailable.')


def overview(rows,key='overview'):
    a=analytics(rows)
    for labels in [[('Analyzed posts','analyzed'),('Safety escalations','safety_escalations'),('Pending reviews','pending_review')]]:
        for col,(label,field) in zip(st.columns(3),labels):col.metric(label,a[field])
    st.caption('Counts use the latest stored attempt per post in the active demo session, including custom submissions. Earlier attempts remain in the audit history.')
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
    styled_table(public_frame(frame))

def render():
    st.title('Demo & Audit')
    view=st.radio('Demo view',['Analyze Post','Examples & feed','Demo analytics','Processing audit','Reset demo'],horizontal=True)
    if view=='Analyze Post':
        analyze_view('demo');return
    if view=='Reset demo':
        st.write('Start a fresh demo session. Previous processing and reviews remain archived.')
        confirmed=st.checkbox('Start a new empty demo session')
        if st.button('Reset demo state',disabled=not confirmed):
            reset();active_path.clear();cached_feed.clear();st.rerun()
        return
    if view=='Processing audit':
        h=history()
        if h:
            records=[get_decision(r['decision_id'],active_path()) for r in h[:20]]
            labels=[r['pipeline'].get('customer_post','')[:100] for r in records]
            selected=st.selectbox('Processed message',range(len(records)),format_func=lambda i:labels[i])
            record=records[selected];result_view(dict(record['pipeline'],review_status=record['review_status']))
        else:st.info('Analyze a post to populate this audit.')
        return
    rows=cached_feed(str(active_path()))
    if view=='Examples & feed':
        st.caption('Sentiment, priority and spread values appear after actual analysis. Select an example in Analyze Post to run it.')
        filtered_feed(rows)
    else:
        a=overview(rows,'demo');chart(a['topic'],'Topic distribution','demo_topic');chart(a['urgency'],'Urgency distribution','demo_urgency')
