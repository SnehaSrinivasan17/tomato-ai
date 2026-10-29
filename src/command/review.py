"""Message-first review queue using the active durable demo session."""
import streamlit as st

def render():
    from src.demo.ui import active_path
    from src.decision.ui import render as review_page
    review_page(active_path())
