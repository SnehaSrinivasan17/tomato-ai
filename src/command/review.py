import streamlit as st
from src.demo.runner import STATE
from src.demo import runner
from src.review.store import DATABASE

def active_path():return runner.active_path()

def render():
    from src.review import postgres
    current=active_path()
    sources={'Active demo session':current,'Existing application audit':postgres.scope() if postgres.configured() else DATABASE}
    if postgres.configured():
        for target in postgres.epochs():
            if target!=current:sources['Archived demo · '+target.epoch[:12]]=target
        st.caption('Review storage: PostgreSQL · shared across app restarts and sessions')
    else:
        for path in sorted(STATE.glob('*/review.sqlite3')):
            if path!=current:sources['Archived demo · '+path.parent.name[:12]]=path
        st.caption('Review storage: local SQLite')
    selected=st.selectbox('Review history source',list(sources))
    from src.decision.ui import render as review_page
    review_page(sources[selected])
