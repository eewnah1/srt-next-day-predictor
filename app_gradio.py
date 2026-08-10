"""Gradio live app for SRT Next-Day Predictor."""
from __future__ import annotations

import json

import gradio as gr
from models.hybrid_predictor import HybridPredictor


def run_prediction():
    p = HybridPredictor()
    p.train(force_synthetic=False)
    pred = p.predict_next(confidence_threshold=0.60)
    direction = "UP" if pred["direction"] == 1 else "DOWN"
    signal = "HIGH CONVICTION" if pred["high_conviction"] else "No signal (wait)"
    out = {
        "ticker": "SRT.SI",
        "as_of": pred["as_of"],
        "price": pred["price"],
        "direction": direction,
        "signal": signal,
        "probability_up": round(pred["probability_up"], 4),
        "confidence": round(pred["confidence"], 4),
        "composite_score": round(pred["composite_score"], 3),
        "tech_score": round(pred["tech_score"], 3),
        "fund_score": round(pred["fund_score"], 3),
        "econ_score": round(pred["econ_score"], 3),
        "ml_prob": round(pred["ml_prob"], 4),
        "seq_prob": round(pred["seq_prob"], 4),
        "interpretation": pred["interpretation"],
    }
    return json.dumps(out, indent=2, default=str)


with gr.Blocks(title="SRT Next-Day Predictor") as demo:
    gr.Markdown("# SRT Next-Day Predictor")
    gr.Markdown("CSOP iEdge S-REIT Leaders Index ETF (SRT.SI). Fetches live data, runs the hybrid ML + sequence model, and returns a directional forecast.")
    run_btn = gr.Button("Run prediction", variant="primary")
    output = gr.Code(label="Forecast", language="json")
    run_btn.click(fn=run_prediction, outputs=output)

if __name__ == "__main__":
    demo.launch(share=True, server_name="0.0.0.0", server_port=8052)
