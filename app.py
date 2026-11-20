"""Minimal Streamlit shell for Phase 2 question answering.

Owner: Aliyan. Starts offline: all node bodies are stubs, so no API key is read.
"""
from __future__ import annotations

import streamlit as st

from src.graph import run_question

st.set_page_config(page_title="Research Paper Discovery", layout="wide")
st.title("Research Paper Discovery and Trend Analysis")
st.caption("Phase 2 skeleton - stub outputs until teammates implement the nodes.")

question = st.text_input("Ask about cs.LG / cs.CL / cs.CV papers or their trends")
if st.button("Ask") and question.strip():
    state = run_question(question.strip(), thread_id="streamlit")
    st.subheader("Route")
    st.write(state.get("route", "unknown"))
    st.subheader("Answer")
    st.write(state.get("answer", ""))
    st.subheader("Citations")
    st.write(state.get("citations", []))
    trend = state.get("trend_data")
    if trend:
        st.subheader("Trend")
        st.write(f"{trend['label']}: {trend['slope']:+.4f} share/quarter")
        st.write(trend.get("chart_path") or "No chart produced by the stub.")
