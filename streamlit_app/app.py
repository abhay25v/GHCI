from __future__ import annotations

import os
import requests
import streamlit as st
from typing import List

API_BASE = os.getenv("ATI_API_BASE", "http://localhost:8000")

st.set_page_config(page_title="ATI Engine Demo", layout="wide")
st.title("Autonomous Transaction Intelligence (ATI) — Demo UI")

with st.sidebar:
    st.header("Settings")
    api_base = st.text_input("API Base URL", API_BASE)
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5)
    explain = st.checkbox("Compute SHAP explanation", value=False)

text = st.text_area("Transaction text", height=150, value="Paid $23.45 at Starbucks Seattle")

col1, col2 = st.columns([1, 1])

if st.button("Predict"):
    try:
        resp = requests.post(f"{api_base}/v1/infer", json={"text": text, "top_k": top_k}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        with col1:
            st.subheader("Predictions")
            st.json(data)
        if explain:
            eresp = requests.post(f"{api_base}/v1/explain", json={"text": text, "max_tokens": 50}, timeout=120)
            eresp.raise_for_status()
            edata = eresp.json()
            with col2:
                st.subheader("Token Attributions")
                # Simple highlight: tokens colored by value sign & magnitude
                tokens = edata.get("attributions", [])
                if tokens:
                    html = []
                    for t in tokens:
                        val = float(t.get("value", 0.0))
                        color = "#ffdddd" if val < 0 else "#ddffdd"
                        weight = min(1.0, abs(val))
                        html.append(
                            f'<span style="background-color:{color}; opacity:{0.4 + 0.6*weight}; padding:2px; margin:2px; border-radius:3px; display:inline-block">{t.get("token")}</span>'
                        )
                    st.markdown(" ".join(html), unsafe_allow_html=True)
                else:
                    st.info("No token attributions available.")
    except Exception as e:
        st.error(f"Request failed: {e}")
