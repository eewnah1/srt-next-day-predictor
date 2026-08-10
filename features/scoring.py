"""Combine technical, fundamental, economic into a single scored signal."""
from __future__ import annotations
import pandas as pd
import numpy as np
from .technical import technical_score
from .fundamental import fundamental_score, holdings_momentum_score
from .economic import economic_score


def compute_factor_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add tech_score, fund_score, econ_score, composite_score columns."""
    out = df.copy()
    tech_scores = []
    fund_scores = []
    econ_scores = []
    hm = holdings_momentum_score()

    for idx, row in out.iterrows():
        t = technical_score(row)
        f = fundamental_score(float(row.get("Close", 0.73)), hm)
        e = economic_score(row)
        tech_scores.append(t)
        fund_scores.append(f)
        econ_scores.append(e)

    out["tech_score"] = tech_scores
    out["fund_score"] = fund_scores
    out["econ_score"] = econ_scores

    # Dynamic weights: in hawkish regime overweight technical + economic
    regime = out.get("rate_regime", pd.Series(0, index=out.index))
    w_tech = np.where(regime == 1, 0.45, 0.40)
    w_econ = np.where(regime == 1, 0.35, 0.25)
    w_fund = 1.0 - w_tech - w_econ

    out["composite_score"] = (
        w_tech * out["tech_score"] +
        w_fund * out["fund_score"] +
        w_econ * out["econ_score"]
    )
    return out


def score_to_direction(score: float, threshold: float = 12.0) -> int:
    """1 = bullish next day, -1 = bearish, 0 = neutral."""
    if score > threshold:
        return 1
    if score < -threshold:
        return -1
    return 0
