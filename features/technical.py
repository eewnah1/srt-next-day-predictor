"""Technical indicators for SRT."""
from __future__ import annotations
import pandas as pd
import numpy as np

try:
    import ta
except ImportError:
    ta = None


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged technical features. All features use only past data."""
    out = df.copy()
    c = out["Close"]
    h, l, v = out["High"], out["Low"], out["Volume"]

    # Returns
    out["ret_1"] = c.pct_change(1)
    out["ret_5"] = c.pct_change(5)
    out["ret_10"] = c.pct_change(10)
    out["ret_20"] = c.pct_change(20)

    # Moving averages & distance
    for w in (5, 10, 20, 50):
        out[f"sma_{w}"] = c.rolling(w).mean()
        out[f"dist_sma_{w}"] = (c - out[f"sma_{w}"]) / out[f"sma_{w}"]

    out["ema_12"] = c.ewm(span=12, adjust=False).mean()
    out["ema_26"] = c.ewm(span=26, adjust=False).mean()
    out["macd"] = out["ema_12"] - out["ema_26"]
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    # Bollinger
    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    out["bb_upper"] = mid + 2 * std
    out["bb_lower"] = mid - 2 * std
    out["bb_pct"] = (c - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"] + 1e-9)

    # ATR & volatility
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    out["atr_14"] = tr.rolling(14).mean()
    out["vol_20"] = out["ret_1"].rolling(20).std()

    # Volume
    out["vol_sma_20"] = v.rolling(20).mean()
    out["vol_ratio"] = v / (out["vol_sma_20"] + 1)

    # Momentum / ROC
    out["roc_10"] = c.pct_change(10)

    # Lag all features so they are known at open of next day
    feature_cols = [col for col in out.columns if col not in df.columns]
    out[feature_cols] = out[feature_cols].shift(1)

    return out.dropna()


def technical_score(row: pd.Series) -> float:
    """Map technical features to a -100 .. +100 score."""
    score = 0.0
    # RSI
    rsi = row.get("rsi_14", 50)
    if rsi < 30:
        score += 25
    elif rsi < 40:
        score += 12
    elif rsi > 70:
        score -= 25
    elif rsi > 60:
        score -= 12

    # MACD hist
    mh = row.get("macd_hist", 0)
    score += np.clip(mh * 800, -20, 20)

    # BB position
    bb = row.get("bb_pct", 0.5)
    if bb < 0.15:
        score += 15
    elif bb > 0.85:
        score -= 15

    # Trend vs SMA20 / SMA50
    d20 = row.get("dist_sma_20", 0)
    d50 = row.get("dist_sma_50", 0)
    score += np.clip(d20 * 300, -15, 15)
    score += np.clip(d50 * 200, -10, 10)

    # Short momentum
    r5 = row.get("ret_5", 0)
    score += np.clip(r5 * 200, -15, 15)

    return float(np.clip(score, -100, 100))
