"""Decision audit, live pipeline playground and explicit human review in the one app."""
import json
from collections import Counter
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ml.charts import layout
from ml.display import display_table
from src.decision.rules import RULES,PRECEDENCE,VERSION
from src.decision.schemas import Action
from src.review.store import audit_rows,get_decision,get_decisions,review,ReviewStatus


def show_result(result, compact=False):
    from html import escape
    st.write(result.get('customer_post',''))
    decision=result.get('decision') or {}
    priority=decision.get('priority','Unknown')
    color={'P0':'#fce8e6','P1':'#fce8e6','P2':'#fff2d8','P3':'#e6f3ed'}.get(priority,'#eef2f8')
    st.markdown('<div style="background:'+color+';padding:18px;border-radius:12px;color:#263445">'
                '<b>'+escape(str(decision.get('action','Unavailable')).replace('_',' '))+'</b> · '+escape(priority)+'</div>',unsafe_allow_html=True)
    ml=result.get('ml') or {}
    left,right=st.columns(2)
    left.metric('Spread probability',f"{ml['probability']:.1%}" if ml.get('probability') is not None else 'Unavailable')
    right.metric('Review status',str(result.get('review_status') or 'Not required').replace('_',' ').title())
    if ml.get('probability') is not None:
        st.caption(f"{'High spread' if ml.get('high_spread') else 'Lower spread'} · operational threshold {ml['operational_threshold']:.0%} · synthetic model")
    g=result.get('generation') or {}
    st.subheader('Response draft')
    if g.get('status')=='review_required' and g.get('draft') and decision.get('requires_response_draft'):
        st.info(g['draft']['draft_response'])
        st.caption('Awaiting human approval · automatic publication disabled')
    elif decision.get('action')=='MONITOR':
        st.write('Monitoring only. A response is not needed.')
    elif decision.get('action')=='URGENT_ESCALATION':
        st.write('Safety escalation: a human must review this message. Automated drafting is disabled.')
    else:
        reason=g.get('error_code') or ', '.join((g.get('validation') or {}).get('issues',[])) or g.get('status','not requested')
        st.write('No validated draft available: '+str(reason).replace('_',' '))
    classification=(result.get('classification') or {}).get('classification') or {}
    columns=st.columns(3)
    for col,field in zip(columns,['sentiment','topic','urgency']):col.metric(field.title(),str(classification.get(field,'Unavailable')).replace('_',' ').title())
    with st.expander('Analysis details and metadata'):
        for reason in decision.get('decision_reasons',[]):st.write(reason)
        if decision.get('missing_inputs'):st.write('Missing inputs:',decision['missing_inputs'])
        st.write('Author and first-hour metadata');st.json(result.get('input_metadata',{}))
        st.write('Classification');st.json(classification)
        st.write('Policy evidence')
        for evidence in (result.get('retrieval') or {}).get('evidence',[]):
            st.write(evidence.get('policy_id'));st.text(evidence.get('chunk_text',''))
        st.write('Stage timing (ms)');st.json(result.get('latencies_ms',{}))
        st.write('Draft validation');st.json(g.get('validation'))


def render(review_path=None):
    st.title('Crisis Monitoring & Reviewer Queue')
    st.caption('One deterministic Python engine · rules '+VERSION+' · allegations investigated, not confirmed · no publishing')
    records=audit_rows(review_path)
    records.sort(key=lambda r:({'P0':0,'P1':1,'P2':2,'P3':3}.get(r['priority'],4),r['created_at']))
    eligible=[r for r in records if r['review_status'] is not None]
    if not eligible:st.info('No matching review records.');return
    labels={r['decision_id']:r.get('customer_post','Message')[:160] for r in eligible}
    chosen=st.selectbox('Message to review',list(labels),format_func=labels.get,key='review_decision_id')
    record=get_decision(chosen,review_path);show_result(dict(record['pipeline'],review_status=record['review_status']))
    st.write('Current review status:',record['review_status'],'· revision:',record['revision'])
    st.caption('Draft approval records a human decision only. It never posts or publishes anything.')
    with st.form('human_review_form_'+chosen):
        reviewer=st.text_input('Human reviewer identifier')
        status=st.selectbox('New review status',[s.value for s in ReviewStatus])
        proposed=record['decision']['action'];actions=[a.value for a in Action]
        final_action=st.selectbox('Human final action',actions,index=actions.index(proposed))
        notes=st.text_area('Reviewer notes / action-change rationale')
        g=record['pipeline'].get('generation') or {};candidate=g.get('draft') or {}
        latest_edit=next((e for e in reversed(record['events']) if e['edited_draft']),None)
        if latest_edit:
            candidate=json.loads(latest_edit['edited_draft'])
            st.write('Latest edited revision:',candidate.get('draft_response',''))
            st.caption('Revision validation: '+str(json.loads(latest_edit['validation_json']).get('status')))
        edited_text=None;support=None
        with st.expander('Draft editing (used only for EDITED status)'):
            edited_text=st.text_area('Final edited draft',value=candidate.get('draft_response',''))
            support=st.text_area('Edited supporting evidence JSON (exact draft spans and policy quotes)',value=json.dumps(candidate.get('supporting_evidence',[]),ensure_ascii=False,indent=2))
        confirmed=st.checkbox('I am the identified human reviewer and have checked this decision and draft',value=False)
        submitted=st.form_submit_button('Record human review')
    if submitted:
        try:
            result=review(chosen,status,reviewer,notes=notes,final_action=final_action,edited_text=edited_text if status=='EDITED' else None,edited_support=json.loads(support) if status=='EDITED' and support is not None else None,human_confirmed=confirmed,expected_revision=record['revision'],path=review_path)
            st.success('Review recorded; no publication occurred.')
            from src.demo.ui import cached_feed
            cached_feed.clear()
        except (ValueError,TypeError) as error:st.error(str(error))
    with st.expander('Append-only review history'):st.json(record['events'])
