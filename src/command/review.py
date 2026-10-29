import streamlit as st
from src.demo.runner import STATE
from src.demo import runner
from src.review.store import DATABASE

def active_path():return runner.active_path()

def render():
    current=active_path()
    sources={'Active demo session':current,'Existing application audit':DATABASE}
    for path in sorted(STATE.glob('*/review.sqlite3')):
        if path!=current:sources['Archived demo · '+path.parent.name[:12]]=path
    selected=st.selectbox('Review history source',list(sources))
    from src.decision.ui import render as review_page
    review_page(sources[selected])
