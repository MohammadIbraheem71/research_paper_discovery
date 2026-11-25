"""Minimal Streamlit shell for Phase 2 question answering.

Owner: Aliyan. Starts offline: all node bodies are stubs, so no API key is read.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.graph import run_question

st.set_page_config(page_title="Research Paper Discovery", layout="wide")
st.title("Research Paper Discovery and Trend Analysis")
st.caption("Retrieve arXiv literature and summarize topic trends.")

question = st.text_input("Ask about cs.LG / cs.CL / cs.CV papers or their trends")
if st.button("Ask") and question.strip():
    try:
        with st.spinner("Searching and generating an answer..."):
            state = run_question(question.strip(), thread_id="streamlit")
    except Exception as exc:
        st.error(f"Unable to answer the question: {type(exc).__name__}: {exc}")
        st.stop()
    st.subheader("Route")
    st.write(state.get("route", "unknown"))
    st.subheader("Answer")
    st.write(state.get("answer", ""))
    if state.get("low_confidence"):
        st.warning("Low confidence: the retrieved evidence may not support this answer.")
    st.subheader("Citations")
    citations = state.get("citations", [])
    if citations:
        for citation in citations:
            arxiv_id = citation["arxiv_id"]
            url = f"https://arxiv.org/abs/{arxiv_id}"
            st.markdown(f"- [arXiv:{arxiv_id}]({url}) — {citation['title']}")
    else:
        st.write("No citations.")
    trend = state.get("trend_data")
    if trend:
        st.subheader("Trend")
        st.write(f"{trend['label']}: {trend['slope']:+.4f} share/quarter")
        chart_path = trend.get("chart_path")
        if chart_path and Path(chart_path).exists():
            st.image(chart_path, caption="Topic share over time")
        else:
            st.write(chart_path or "No chart produced.")
