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
from src.review.store import audit_rows,get_decision,review,ReviewStatus
from src.workflow.pipeline import dataset,predictor,process_post,process_custom_post


def show_result(result):
    st.write('Post ID:',result['post_id']);st.write(result.get('customer_post',''))
    decision=result.get('decision')
    if decision:
        left,right=st.columns(2);left.metric('Deterministic action',decision['action']);right.metric('Priority',decision['priority'])
        st.write('Applied rule IDs:',decision['rule_ids']);st.write('Decision rule version:',decision['decision_rule_version'])
        for reason in decision['decision_reasons']:st.write(reason)
        st.write('Missing inputs:',decision['missing_inputs'])
    c=result.get('classification') or {}
    with st.expander('LLM classification and independent ML prediction'):
        st.json(c);st.json(result.get('ml'))
    with st.expander('Retrieved policy evidence'):
        r=result.get('retrieval') or {};st.write('Retrieval status:',r.get('status','not requested'))
        for e in r.get('evidence',[]):
            st.write(e['policy_id'],e.get('source_filename'),e.get('version'));st.text(e.get('chunk_text',''))
    g=result.get('generation') or {}
    if g.get('status')=='review_required' and g.get('draft') and decision and decision['requires_response_draft']:
        st.subheader('Draft awaiting human review');st.code(g['draft']['draft_response'],language=None)
    else:st.write('No eligible copyable draft:',g.get('error_code') or g.get('status','not requested'))
    st.write('Human review required:',result.get('requires_human_review',True),'· Automatic publishing: disabled')
    if result.get('decision_id'):st.caption('Persistent audit ID: '+result['decision_id'])
    with st.expander('Decision stages and validators'):
        st.json(result.get('decision_history',[]));st.json(g.get('validation'));st.json(result.get('latencies_ms',{}))


def render(review_path=None):
    st.title('Crisis Monitoring & Reviewer Queue')
    st.caption('One deterministic Python engine · rules '+VERSION+' · allegations investigated, not confirmed · no publishing')
    overview,explanations,playground,queue=st.tabs(['Stored decisions','Rule precedence','Try It Yourself','Human review'])
    records=audit_rows(review_path)
    records.sort(key=lambda r:({'P0':0,'P1':1,'P2':2,'P3':3}.get(r['priority'],4),r['created_at']))
    with overview:
        st.caption('Counts cover actual stored workflow runs. Re-running a post adds a new audit record; no predictions are generated on page load.')
        cols=st.columns(4)
        for col,(label,value) in zip(cols,[('Stored decisions',len(records)),('Unique posts',len({r['post_id'] for r in records})),('Urgent escalations',sum(r['proposed_action']=='URGENT_ESCALATION' for r in records)),('Pending review',sum(r['review_status']=='PENDING_REVIEW' for r in records))]):col.metric(label,value)
        if records:
            for field,title in [('final_action','Final action distribution'),('priority','Engine priority distribution')]:
                counts=Counter(r[field] for r in records)
                fig=go.Figure(go.Bar(x=list(counts),y=list(counts.values())))
                st.plotly_chart(layout(fig,title,field.replace('_',' ').title(),'Stored decision runs'),use_container_width=True,key='decision_'+field)
            st.write('Safety overrides:',sum('A_CRITICAL' in r['rule_ids'] for r in records))
            st.write('High-spread priorities:',sum('C_HIGH_SPREAD' in r['rule_ids'] for r in records))
            st.write('Missing-input runs:',sum(bool(r['missing_inputs']) for r in records))
            st.write('Abstentions:',sum(r['proposed_action']=='ABSTAIN_FOR_HUMAN' for r in records))
            st.dataframe(display_table(pd.DataFrame(records)),use_container_width=True)
        else:st.info('No stored decisions yet. Run one post in Try It Yourself.')
    with explanations:
        for rank,key in enumerate(PRECEDENCE,1):st.write(f'{rank}. {key}: {RULES[key]}')
        st.write('Additional gates:')
        for key in ['F_ML_UNAVAILABLE','F_POLICY','F_DRAFT']:st.write(key+': '+RULES[key])
        st.caption('Missing ML never becomes zero. Safety and priority cannot be downgraded by unavailable evidence. Critical/serious cases go to human investigation without an automated public draft.')
    with playground:
        from src.demo.ui import analyze_view
        st.caption('New analyses use the active demo session; select its review source to inspect them.')
        analyze_view('decision_unified')
    with queue:
        statuses=st.multiselect('Review queue status',[e.value for e in ReviewStatus],default=['PENDING_REVIEW','EDITED'])
        eligible=[r for r in records if r['review_status'] is not None and (not statuses or r['review_status'] in statuses)]
        if not eligible:st.info('No matching review records.');return
        chosen=st.selectbox('Decision audit ID',[r['decision_id'] for r in eligible],key='review_decision_id')
        record=get_decision(chosen,review_path);show_result(dict(record['pipeline'],decision_id=chosen))
        st.write('Current review status:',record['review_status'],'· revision:',record['revision'])
        st.caption('Draft approval records a human decision only. It never posts or publishes anything.')
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
        if status=='EDITED':
            edited_text=st.text_area('Final edited draft',value=candidate.get('draft_response',''))
            support=st.text_area('Edited supporting evidence JSON (exact draft spans and policy quotes)',value=json.dumps(candidate.get('supporting_evidence',[]),ensure_ascii=False,indent=2))
        confirmed=st.checkbox('I am the identified human reviewer and have checked this decision and draft',value=False)
        if st.button('Record human review'):
            try:
                result=review(chosen,status,reviewer,notes=notes,final_action=final_action,edited_text=edited_text,edited_support=json.loads(support) if support is not None else None,human_confirmed=confirmed,expected_revision=record['revision'],path=review_path)
                st.success('Review recorded; no publication occurred.');st.json(result)
            except (ValueError,TypeError) as error:st.error(str(error))
        with st.expander('Append-only review history'):st.json(record['events'])
