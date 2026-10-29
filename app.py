"""One Tomato AI Command Center; labs reuse existing implementations."""
import streamlit as st

st.set_page_config(page_title='Tomato AI · Command Center',layout='wide')
st.markdown('''<style>
.block-container {padding-top:2rem; max-width:1500px;}
[data-testid="stMetric"] {border:1px solid #dbe3ec; border-radius:8px; padding:14px;}
[data-testid="stMetric"] {background:#f0f5fa;}
[data-testid="stAlert"] {border-radius:10px;}
[data-testid="stMetricValue"] {font-size:1.65rem; white-space:normal;}
[data-testid="stSidebar"] {background:#f0f5f6;}
[data-testid="stExpander"] {border-radius:10px;background:#f7f9fc;}
button[kind="primary"] {background:#357d73;border-color:#357d73;}
.tomato-table {width:100%;border-collapse:separate;border-spacing:3px;font-size:.9rem;}
.tomato-table th {background:#edf2f8;text-align:left;padding:10px;color:#34465a;}
.tomato-table td {padding:10px;border-radius:5px;min-width:80px;max-width:460px;white-space:normal;}
h1 {font-size:2rem;} h2 {font-size:1.45rem;} h3 {font-size:1.15rem;}
@media (prefers-reduced-motion: reduce) {* {animation:none!important;transition:none!important;}}
</style>''',unsafe_allow_html=True)
PAGES=['Command Center','Demo & Audit','ML Spread Lab','LLM Evaluation Lab','RAG & Response Lab','Crisis Monitoring & Reviewer Queue']
st.sidebar.title('Tomato AI')
page=st.sidebar.radio('Workspace',PAGES)
st.sidebar.caption('Fictional demonstration · human approval · publication disabled')
if page=='Command Center':
    from src.command.ui import render
    render()
elif page=='ML Spread Lab':
    view=st.radio('ML lab view',['Final selected model','Other tested models','Combined Analysis'],horizontal=True)
    if view=='Final selected model':
        from ml.dashboard import render
    elif view=='Other tested models':
        from ml.comparisons import render
    else:
        from llm.ui import combined_page as render
    render()
elif page=='LLM Evaluation Lab':
    from llm.ui import evaluation_page
    evaluation_page()
elif page=='Crisis Monitoring & Reviewer Queue':
    from src.command.review import render
    render()
elif page=='RAG & Response Lab':
    from src.rag.ui import render
    render()
elif page=='Demo & Audit':
    from src.demo.ui import render
    render()
