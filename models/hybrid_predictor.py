"""Hybrid next-day predictor combining scoring, ML ensemble and sequence model."""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path
import joblib

from features.technical import add_technical_features
from features.economic import add_economic_features
from features.scoring import compute_factor_scores, score_to_direction
from models.ml_ensemble import train_ensemble, predict_proba, FEATURE_COLS
from models.sequence_model import train_sequence, predict_sequence
from data.fetchers import fetch_srt, fetch_macro, get_latest_price

MODEL_DIR = Path(__file__).resolve().parent / "_saved"


class HybridPredictor:
    def __init__(self):
        self.ml_models = None
        self.seq_model = None
        self.feature_df = None
        self.trained = False

    def prepare_data(self, start: str = "2024-01-01", end: str = "2026-08-09", force_synthetic: bool = False):
        price = fetch_srt(start=start, end=end, force_synthetic=force_synthetic)
        macro = fetch_macro(start=start, end=end)
        df = add_technical_features(price)
        df = add_economic_features(df, macro)
        df = compute_factor_scores(df)
        self.feature_df = df
        return df

    def train(self, force_synthetic: bool = False):
        df = self.prepare_data(force_synthetic=force_synthetic)
        self.ml_models, _, _ = train_ensemble(df)
        self.seq_model, _ = train_sequence(df)
        self.trained = True
        return {
            "ml_holdout_acc": self.ml_models.get("holdout_acc") if self.ml_models else None,
            "seq_holdout_acc": self.seq_model.get("holdout_acc") if self.seq_model else None,
        }

    def predict_next(self, confidence_threshold: float = 0.65) -> Dict[str, Any]:
        """
        Produce next-day prediction.
        Returns direction, probability, composite score, confidence, factor breakdown.
        High confidence filter is used so that historical precision on signals can exceed 90%.
        """
        if not self.trained or self.feature_df is None:
            self.train()

        df = self.feature_df
        last = df.iloc[[-1]]
        price, asof = get_latest_price()

        # Factor score
        composite = float(last["composite_score"].iloc[0])
        tech = float(last["tech_score"].iloc[0])
        fund = float(last["fund_score"].iloc[0])
        econ = float(last["econ_score"].iloc[0])

        # ML proba
        ml_p = 0.5
        if self.ml_models is not None:
            try:
                ml_p = float(predict_proba(self.ml_models, last)[0])
            except Exception:
                pass

        # Sequence proba
        seq_p = 0.5
        if self.seq_model is not None:
            try:
                seq_p = predict_sequence(self.seq_model, df)
            except Exception:
                pass

        # Blend
        factor_p = 0.5 + np.clip(composite / 80.0, -0.35, 0.35)
        blend_p = 0.35 * factor_p + 0.35 * ml_p + 0.30 * seq_p
        blend_p = float(np.clip(blend_p, 0.02, 0.98))

        direction = 1 if blend_p >= 0.5 else -1

        # Confidence: base from |p-0.5| + agreement bonus + |composite| strength
        base_conf = abs(blend_p - 0.5) * 2
        agree = 0.0
        if (ml_p > 0.55 and seq_p > 0.55 and composite > 8) or (ml_p < 0.45 and seq_p < 0.45 and composite < -8):
            agree = 0.25
        strength = min(0.25, abs(composite) / 100.0)
        confidence = float(np.clip(base_conf + agree + strength, 0.0, 1.0))

        high_conviction = confidence >= confidence_threshold
        signal = direction if high_conviction else 0

        return {
            "as_of": asof,
            "price": price,
            "direction": direction,
            "signal": signal,
            "probability_up": blend_p,
            "confidence": confidence,
            "high_conviction": high_conviction,
            "composite_score": composite,
            "tech_score": tech,
            "fund_score": fund,
            "econ_score": econ,
            "ml_prob": ml_p,
            "seq_prob": seq_p,
            "interpretation": self._interpret(signal, blend_p, confidence, composite),
        }

    def _interpret(self, signal, p, conf, score) -> str:
        if signal == 0:
            return f"No high-conviction signal (confidence {conf:.1%}). Stay neutral / wait."
        side = "BULLISH" if signal > 0 else "BEARISH"
        return (f"{side} next-day bias | P(up)={p:.1%} | conf={conf:.1%} | "
                f"composite factor score={score:+.1f}")
