import streamlit as st
import pandas as pd
from src.demo.manifest import posts
from src.demo.runner import all_feed
from src.demo.ui import overview, styled_table, public_frame, cached_feed, active_path


def render():
    st.title('Tomato AI · Command Center')
    st.subheader('Operational overview')
    rows=cached_feed(str(active_path()));overview(rows,'command')
    st.subheader('Recent activity')
    actual=[r for r in rows if r['decision_id']]
    if actual:styled_table(public_frame(pd.DataFrame(actual).sort_values('processed_at',ascending=False).head(20)))
    else:st.info('Open Demo & Audit to analyze a post or run a featured scenario.')
