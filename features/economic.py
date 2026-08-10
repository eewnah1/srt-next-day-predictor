"""Economic / macro factor scoring for Singapore REIT context."""
from __future__ import annotations
import pandas as pd
import numpy as np


def add_economic_features(price_df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Align macro and create lagged economic features."""
    out = price_df.copy()
    m = macro.reindex(out.index).ffill()

    if "STI" in m.columns:
        out["sti"] = m["STI"]
        out["sti_ret_5"] = out["sti"].pct_change(5).shift(1)
        out["sti_ret_20"] = out["sti"].pct_change(20).shift(1)
        out["rel_strength_sti"] = (out["Close"].pct_change(20) - out["sti"].pct_change(20)).shift(1)

    if "US10Y" in m.columns:
        out["us10y"] = m["US10Y"]
        out["us10y_chg_5"] = out["us10y"].diff(5).shift(1)
        out["us10y_level"] = out["us10y"].shift(1)

    if "USDSGD" in m.columns:
        out["usdsgd"] = m["USDSGD"]
        out["sgd_strength"] = -out["usdsgd"].pct_change(10).shift(1)  # rising SGD = positive

    # Simple regime: based on US10Y trend (proxy for global rate pressure on REITs)
    if "us10y_chg_5" in out.columns:
        out["rate_regime"] = np.where(out["us10y_chg_5"] > 0.08, 1,  # hawkish
                                      np.where(out["us10y_chg_5"] < -0.08, -1, 0))  # easing / neutral
    else:
        out["rate_regime"] = 0

    return out


def economic_score(row: pd.Series) -> float:
    """-100 .. +100 based on rate regime, relative strength, FX."""
    score = 0.0

    regime = row.get("rate_regime", 0)
    # REITs dislike rising rates
    score -= regime * 18

    us10y = row.get("us10y_level", 4.2)
    # Higher absolute rates compress valuations
    if us10y > 4.5:
        score -= 12
    elif us10y < 3.8:
        score += 10

    rs = row.get("rel_strength_sti", 0)
    score += np.clip(rs * 150, -15, 15)

    sgd = row.get("sgd_strength", 0)
    score += np.clip(sgd * 80, -10, 10)

    # Mild positive for Singapore growth resilience (static bias from analyst notes ~3% GDP)
    score += 6

    return float(np.clip(score, -100, 100))
