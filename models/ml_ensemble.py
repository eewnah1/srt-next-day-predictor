"""Classical ML ensemble for next-day direction."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score
import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "_saved"
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "ret_1", "ret_5", "ret_10", "ret_20",
    "dist_sma_5", "dist_sma_10", "dist_sma_20", "dist_sma_50",
    "macd_hist", "rsi_14", "bb_pct", "atr_14", "vol_20", "vol_ratio", "roc_10",
    "sti_ret_5", "sti_ret_20", "rel_strength_sti", "us10y_chg_5", "us10y_level",
    "sgd_strength", "rate_regime", "tech_score", "fund_score", "econ_score", "composite_score",
]


def prepare_xy(df: pd.DataFrame):
    """Target = next day close direction (1 if up or flat, 0 if down)."""
    data = df.copy()
    data["target"] = (data["Close"].shift(-1) >= data["Close"]).astype(int)
    cols = [c for c in FEATURE_COLS if c in data.columns]
    data = data.dropna(subset=cols + ["target"])
    X = data[cols]
    y = data["target"]
    return X, y, data.index, cols


def train_ensemble(df: pd.DataFrame, save: bool = True):
    X, y, idx, cols = prepare_xy(df)
    if len(X) < 100:
        return None, None, cols

    # Time-series aware split for last 20% as holdout
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    gb = GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.05, random_state=42)
    rf = RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=10, random_state=42, n_jobs=-1)

    gb.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # Simple average of probabilities
    proba_gb = gb.predict_proba(X_test)[:, 1]
    proba_rf = rf.predict_proba(X_test)[:, 1]
    proba = 0.55 * proba_gb + 0.45 * proba_rf
    pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)

    if save:
        joblib.dump({"gb": gb, "rf": rf, "cols": cols, "acc": acc, "prec": prec}, MODEL_DIR / "ensemble.joblib")

    return {"gb": gb, "rf": rf, "cols": cols, "holdout_acc": acc, "holdout_prec": prec}, (X_test, y_test, proba), cols


def predict_proba(models: dict, row_or_df: pd.DataFrame) -> np.ndarray:
    cols = models["cols"]
    X = row_or_df[cols].fillna(0)
    p_gb = models["gb"].predict_proba(X)[:, 1]
    p_rf = models["rf"].predict_proba(X)[:, 1]
    return 0.55 * p_gb + 0.45 * p_rf
