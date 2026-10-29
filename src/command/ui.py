import streamlit as st
import pandas as pd
from ml.display import display_table
from src.demo.manifest import posts
from src.demo.runner import all_feed
from src.demo.ui import overview,analyze_view

def render():
    st.title('Tomato AI · Command Center')
    st.caption('Synthetic demonstration · classification, spread prediction and human decisions connected by shared post IDs')
    metrics,analyze=st.tabs(['Operational overview','Analyze Post'])
    with metrics:
        rows=all_feed(posts());overview(rows,'command')
        st.subheader('Recent activity')
        actual=[r for r in rows if r['decision_id']]
        if actual:st.dataframe(display_table(pd.DataFrame(actual).drop(columns=['complete']).sort_values('processed_at',ascending=False).head(20)),use_container_width=True)
        else:st.info('No posts analyzed in this demo session. Open Analyze Post or run a featured scenario.')
    with analyze:analyze_view()
