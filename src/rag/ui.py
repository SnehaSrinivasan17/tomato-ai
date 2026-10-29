"""One existing Streamlit application: retrieval, drafts, review and evaluation."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from src.rag.settings import RESULTS,EMBEDDING_MODEL,collection,generation_model
from src.rag.ingest import qdrant_client,load_chunks
from src.rag.evaluate import candidates,save_review,approved_references
from src.workflow.pipeline import dataset,process_post,process_custom_post,predictor
from llm.ui import classification_output
from ml.display import display_table


def render():
    st.title('RAG & Response Lab')
    st.caption('Fictional proposed Tomato policies · internal drafts · human approval required')
    overview,evaluation,quality,playground,review=st.tabs(['System overview','Retrieval evaluation','Response quality','Try It Yourself','Reference review'])
    with overview:
        st.write('Embedding model:',EMBEDDING_MODEL)
        st.write('Qdrant collection:',collection())
        st.write('Groq drafting model:',generation_model())
        st.write('Dense cosine retrieval · top 5 · relevance threshold 0.35 · high-risk corroboration ≥ 2 passages')
        if (RESULTS/'ingestion.json').exists():
            data=json.loads((RESULTS/'ingestion.json').read_text(encoding='utf8'))
            st.caption('Last successful ingestion · saved snapshot')
            left,right=st.columns(2)
            left.metric('Policy source documents',data['documents'])
            right.metric('Indexed policy chunks',data['records'])
            with st.expander('Ingestion provenance'):st.json(data)
        else:st.info('No successful ingestion report available yet.')
        if st.button('Check live Qdrant collection'):
            try:
                client=qdrant_client();info=client.get_collection(collection())
                points,_=client.scroll(collection(),limit=10000,with_vectors=False)
                st.metric('Live indexed chunks',client.count(collection(),exact=True).count)
                st.metric('Live source documents',len({p.payload.get('source_filename') for p in points if p.payload}))
                st.write('Vector dimension:',info.config.params.vectors.size)
            except Exception:st.error('Qdrant unavailable or collection not indexed. No counts fabricated.')
        if (RESULTS/'policy_audit.json').exists():
            audit=json.loads((RESULTS/'policy_audit.json').read_text(encoding='utf8'))
            with st.expander('Policy provenance, exceptions and conflicts'):
                st.write('Documents parsed:',len(audit['documents']),'· policy sections:',audit['policy_records'])
                st.json(audit['conflicts']);st.markdown(audit['hierarchy'])
    report=None
    reports=list((RESULTS/'evaluations').glob('*/report.json'))
    if reports:
        reports.sort(key=lambda p:p.stat().st_mtime,reverse=True)
        with evaluation:
            names=[p.parent.name for p in reports]
            latest=RESULTS/'latest_evaluation.json'
            preferred=Path(json.loads(latest.read_text())['directory']).name if latest.exists() else names[0]
            name=st.selectbox('Saved RAG evaluation',names,index=names.index(preferred) if preferred in names else 0,key='rag_evaluation_name')
        directory=next(p.parent for p in reports if p.parent.name==name)
        try: report=json.loads((directory/'report.json').read_text(encoding='utf8'))
        except (OSError,ValueError):st.error('Saved RAG evaluation report unavailable.')
    with evaluation:
        if report:
            st.caption('Evaluation model: '+str(report.get('metadata',{}).get('model','not recorded'))+' · generation prompt '+str(report.get('metadata',{}).get('generation_prompt_version','not recorded'))+' · saved run, not a live score')
            category=st.selectbox('Retrieval reference category',['human_verified_retrieval','provisional_retrieval'])
            metrics=report.get(category)
            if metrics:
                st.caption(f"Reference status: {category} · {metrics['n']} evaluated mappings")
                columns=st.columns(4)
                for col,(name,key) in zip(columns,[('Recall@3','recall_at_3'),('Recall@5','recall_at_5'),('Precision@5','precision_at_5'),('MRR','mrr')]):col.metric(name,f'{metrics[key]:.3f}')
                st.write('Mean retrieval latency (ms):',metrics['mean_retrieval_latency_ms'])
                st.write('Retrieval failure rate:',metrics['retrieval_failure_rate'])
            else:st.info('Reviewed reference metrics pending approved mappings and an evaluation run.')
        else:st.info('No completed live retrieval evaluation artifact available.')
    with quality:
        if report:
            st.write('Saved-run validation results:');st.json(report['generation']);st.json(report.get('validation_issue_counts',{}));st.write('Independent safety escalations:',report.get('essential_safety_escalations'));st.write('Policy-conflict blocks:',report.get('policy_conflict_blocks'));st.write(report['subjective_quality'])
            if report.get('current_guardrail_revalidation'):
                st.write('Post-run validation with current guardrails:');st.json(report['current_guardrail_revalidation'])
        else:st.info('Response quality evaluation pending a live run.')
        st.markdown('Human rubric: verify each material claim against its cited policy and exceptions; assess empathy, English/Hindi/Hinglish appropriateness, allegation handling, privacy, escalation and injection resistance. Record pass/fail with evidence; deterministic checks do not prove semantic entailment.')
        if (RESULTS/'pipeline_runs.jsonl').exists():
            with st.expander('Recent real workflow outcomes'):
                records=[json.loads(x) for x in (RESULTS/'pipeline_runs.jsonl').read_text(encoding='utf8').splitlines()[-20:]]
                st.dataframe(display_table(pd.DataFrame([dict(post_id=r['post_id'],status=r['status'],human_review=r['requires_human_review'],total_ms=r['latencies_ms']['total']) for r in records])),use_container_width=True)
    with playground:
        from src.demo.ui import analyze_view
        analyze_view('rag_unified')
    with review:
        rows=candidates();ids=[r['post_id'] for r in rows]
        selected=st.selectbox('Relevance review post ID',ids)
        row=next(r for r in rows if r['post_id']==selected);st.write(row['text'])
        st.caption('Provisional suggestion; model retrieval is not shown as ground truth.')
        st.write(row['expected_policy_ids'])
        policy_ids=sorted({r['policy_id'] for r in load_chunks() if r['kind']=='policy'}) if (RESULTS/'rag_chunks.jsonl').exists() else []
        chosen=st.multiselect('Reviewed relevant policy IDs',policy_ids)
        reviewer=st.text_input('Policy relevance reviewer')
        notes=st.text_area('Relevance rationale and subjective response review notes')
        approved=st.checkbox('I checked the source policies and approve this reference mapping',value=False)
        if st.button('Save reviewed mapping'):
            try:save_review(selected,chosen,reviewer,approved,notes);st.success('Stored a separate reference revision; rerun evaluation to recompute metrics.')
            except ValueError as error:st.error(str(error))
        st.write('Human-verified mappings:',len(approved_references()))
