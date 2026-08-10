"""Deep-learning style sequential predictor using sklearn MLP on lagged feature windows.
(Lightweight alternative to full LSTM/TF for portability and speed.)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "_saved"
MODEL_DIR.mkdir(exist_ok=True)

LAGS = 5
CORE_FEATS = ["ret_1", "rsi_14", "macd_hist", "bb_pct", "vol_ratio", "composite_score"]


def make_sequences(df: pd.DataFrame, feature_cols=None, lags: int = LAGS):
    if feature_cols is None:
        feature_cols = [c for c in CORE_FEATS if c in df.columns]
    data = df[feature_cols + ["Close"]].copy()
    data["target"] = (data["Close"].shift(-1) >= data["Close"]).astype(int)
    data = data.dropna()

    Xs, ys, idxs = [], [], []
    vals = data[feature_cols].values
    targets = data["target"].values
    index = data.index

    for i in range(lags, len(data)):
        window = vals[i - lags : i].flatten()
        Xs.append(window)
        ys.append(targets[i])
        idxs.append(index[i])

    return np.array(Xs), np.array(ys), idxs, feature_cols


def train_sequence(df: pd.DataFrame, save: bool = True):
    X, y, idxs, cols = make_sequences(df)
    if len(X) < 120:
        return None, None

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                              max_iter=400, early_stopping=True, random_state=42,
                              learning_rate_init=0.001))
    ])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = (pred == y_test).mean()

    if save:
        joblib.dump({"pipe": pipe, "cols": cols, "lags": LAGS, "acc": acc}, MODEL_DIR / "sequence.joblib")

    return {"pipe": pipe, "cols": cols, "lags": LAGS, "holdout_acc": acc}, (X_test, y_test, proba)


def predict_sequence(model: dict, df: pd.DataFrame) -> float:
    """Return P(up) for the most recent complete window."""
    cols = model["cols"]
    lags = model["lags"]
    if len(df) < lags + 1:
        return 0.5
    recent = df[cols].iloc[-lags:].values.flatten().reshape(1, -1)
    return float(model["pipe"].predict_proba(recent)[0, 1])
