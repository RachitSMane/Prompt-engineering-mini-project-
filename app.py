"""Streamlit entry point:  streamlit run app.py"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ui.common import render_sidebar
from ui.components import inject_css
from utils.logging_config import setup_logging

ASSETS = Path(__file__).parent / "assets"

st.set_page_config(page_title="ReviewIQ", page_icon=str(ASSETS / "logo_mark.svg"), layout="wide", initial_sidebar_state="auto")
setup_logging()
inject_css()
st.logo(str(ASSETS / "logo.svg"), size="large", icon_image=str(ASSETS / "logo_mark.svg"))
render_sidebar()

pages = [
    st.Page("ui/home.py", title="Overview", icon=":material/space_dashboard:", default=True),
    st.Page("ui/review_analysis.py", title="Review Analysis", icon=":material/rate_review:"),
    st.Page("ui/prompt_comparison.py", title="Prompt Experiments", icon=":material/science:"),
    st.Page("ui/product_insights.py", title="Product Intelligence", icon=":material/inventory_2:"),
    st.Page("ui/summary_evaluation.py", title="Summary Evaluation", icon=":material/fact_check:"),
    st.Page("ui/annotation_page.py", title="Human Annotation", icon=":material/edit_note:"),
]
st.navigation(pages).run()
