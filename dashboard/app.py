"""
Streamlit dashboard for SRT Next-Day Predictor.
Run with: streamlit run dashboard/app.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Make project root importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.hybrid_predictor import HybridPredictor
from backtest.engine import run_backtest
from data.fetchers import get_latest_price

st.set_page_config(
    page_title="SRT Next-Day Predictor",
    page_icon="📈",
    layout="wide",
)

st.title("SRT Next-Day Predictor")
st.caption("CSOP iEdge S-REIT Leaders Index ETF (SGX:SRT) · Multi-factor + ML + Sequence · Research tool only")

# Sidebar
with st.sidebar:
    st.header("Controls")
    conf_thresh = st.slider("High-conviction threshold", 0.50, 0.90, 0.60, 0.01)
    use_synth = st.checkbox("Force synthetic data (bypass yfinance rate limits)", value=True)
    st.markdown("---")
    st.markdown("**Data notes**")
    st.markdown("- Live SRT.SI may be rate-limited by Yahoo; synthetic is calibrated to observed vol & levels.")
    st.markdown("- Economic series (STI, US10Y, USDSGD) attempt live fetch.")
    st.markdown("---")
    st.markdown("Not financial advice. See README.")

# Main action
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    run_btn = st.button("▶ Run Next-Day Predictor", type="primary", use_container_width=True)
with col2:
    bt_btn = st.button("Run Backtest (2024–2026)", use_container_width=True)

@st.cache_resource
def get_predictor(force_synth: bool):
    p = HybridPredictor()
    p.train(force_synthetic=force_synth)
    return p

if run_btn:
    with st.spinner("Scoring factors · running ensemble & sequence models…"):
        predictor = get_predictor(use_synth)
        result = predictor.predict_next(confidence_threshold=conf_thresh)

    st.success("Prediction complete")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Latest Price (SGD)", f"{result['price']:.4f}", help=f"As of {result['as_of']}")
    dir_label = "▲ UP" if result["direction"] > 0 else "▼ DOWN"
    m2.metric("Direction Bias", dir_label, f"P(up)={result['probability_up']:.1%}")
    m3.metric("Confidence", f"{result['confidence']:.1%}")
    signal_txt = "HIGH CONVICTION" if result["high_conviction"] else "No signal (wait)"
    m4.metric("Actionable Signal", signal_txt)

    st.info(result["interpretation"])

    # Factor breakdown
    st.subheader("Factor Scores (−100 … +100)")
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("Composite", f"{result['composite_score']:+.1f}")
    f2.metric("Technical", f"{result['tech_score']:+.1f}")
    f3.metric("Fundamental", f"{result['fund_score']:+.1f}")
    f4.metric("Economic", f"{result['econ_score']:+.1f}")

    # Gauge-like bar
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=result["probability_up"] * 100,
        title={"text": "P(Next Day Up) %"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 40], "color": "#ffcccc"},
                {"range": [40, 60], "color": "#ffffcc"},
                {"range": [60, 100], "color": "#ccffcc"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 50,
            },
        },
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Model probabilities detail"):
        st.write({
            "ML ensemble P(up)": round(result["ml_prob"], 4),
            "Sequence / DL-style P(up)": round(result["seq_prob"], 4),
            "Blended P(up)": round(result["probability_up"], 4),
            "High-conviction threshold used": conf_thresh,
        })

if bt_btn:
    with st.spinner("Running walk-forward style backtest on 2024-01-01 → 2026-08-08…"):
        bt = run_backtest(
            start="2024-01-01",
            end="2026-08-08",
            confidence_threshold=conf_thresh,
            force_synthetic=use_synth,
        )

    st.subheader("Backtest Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Full-sample Accuracy", f"{bt['full_sample_accuracy']*100:.1f}%")
    hc_acc = bt["high_confidence_accuracy"]
    c2.metric("High-Conviction Accuracy", f"{hc_acc*100:.1f}%" if hc_acc is not None else "N/A")
    c3.metric("Coverage (high-conv days)", f"{bt['high_confidence_coverage']*100:.1f}%")
    c4.metric("# High-Conviction Signals", bt["n_high_conviction_signals"])

    st.caption(bt["note"])

    if bt["details"] is not None and len(bt["details"]) > 0:
        details = bt["details"]
        details = details.copy()
        details["position"] = 0
        details.loc[details["high_conv"] & (details["pred"] == 1), "position"] = 1
        details.loc[details["high_conv"] & (details["pred"] == 0), "position"] = -1
        details["ret"] = np.where(details["actual"] == 1, 0.004, -0.004)
        details["strat_ret"] = details["position"].shift(1).fillna(0) * details["ret"]
        details["cum"] = (1 + details["strat_ret"]).cumprod()

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=details["date"], y=details["cum"], name="High-conv strategy (illustrative)"))
        fig2.update_layout(title="Illustrative cumulative equity (high-conviction signals only)", height=360)
        st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(details.tail(20), use_container_width=True)

st.markdown("---")
st.markdown(
    "**Source context**: FSM Global screenshot + CSOP factsheet (30 Jun 2026). "
    "Top holdings CapitaLand Integrated, Ascendas, Keppel DC, Mapletree Logistics etc. "
    "~98.6% Real Estate, semi-annual dividend, LTV 70%, mgmt fee 0.50%."
)
