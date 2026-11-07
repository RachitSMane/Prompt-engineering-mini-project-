"""Streamlit entry point:  streamlit run app.py"""

from __future__ import annotations

import streamlit as st

from ui.common import render_sidebar
from utils.logging_config import setup_logging

st.set_page_config(page_title="Aspect-Based Review Analytics", page_icon="📊", layout="wide")
setup_logging()
render_sidebar()

pages = [
    st.Page("ui/home.py", title="Home / Dashboard", icon="🏠", default=True),
    st.Page("ui/review_analysis.py", title="Review Analysis", icon="🔎"),
    st.Page("ui/prompt_comparison.py", title="Prompt Comparison", icon="⚖️"),
    st.Page("ui/product_insights.py", title="Product Insights", icon="📦"),
    st.Page("ui/summary_evaluation.py", title="Summary Evaluation", icon="🧑‍⚖️"),
    st.Page("ui/annotation_page.py", title="Data Annotation", icon="✍️"),
]
st.navigation(pages).run()
