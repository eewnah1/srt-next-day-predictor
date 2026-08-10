"""Fundamental scoring for SRT using static + dynamic signals from factsheet & top holdings."""
from __future__ import annotations
import pandas as pd
import numpy as np

# Snapshot from FSM screenshot + CSOP factsheet (Jun/Aug 2026)
HOLDINGS_YTD = {
    "C38U": 6.52,   # CapitaLand Integrated
    "BQUU": 0.0,    # Ascendas (placeholder)
    "AJBU": 2.65,   # Keppel DC
    "M44U": -4.96,  # Mapletree Log
    "J69U": -2.52,  # Frasers Centrepoint
    "BUOU": 0.45,   # Frasers Log & Comm
    "ME8U": -3.19,  # Mapletree Ind
    "K71U": -5.14,  # Keppel REIT
    "SUN": 0.0,
    "JYEU": -4.51,
    "8C8U": 6.07,   # Centurion
    "NTDU": -3.37,
    "C2PU": 5.30,   # Parkway
}

SECTOR_WEIGHT_REAL_ESTATE = 0.986
DIV_YIELD_TTM ≈ 0.059
LTV = 0.70
MGMT_FEE = 0.005
NAV_REF = 0.74  # ~30 Jun 2026


def fundamental_score(price: float, holdings_momentum: float = None) -> float:
    """
    Static + semi-dynamic fundamental score -100..+100.
    Positive when yield attractive vs rates, holdings momentum supportive, discount to NAV.
    """
    score = 0.0

    # Discount / premium to last known NAV
    if price > 0:
        prem = (price - NAV_REF) / NAV_REF
        # mild preference for discount
        score -= np.clip(prem * 80, -20, 20)

    # Dividend yield attractiveness (static baseline; can be enhanced with live yield)
    # In higher-for-longer, high yield helps but rate sensitivity hurts
    score += 12  # baseline quality of distribution history

    # LTV 70% is moderate-high; slight penalty in hawkish regime (handled in economic)
    score -= 5

    # Holdings breadth: top 25 ~98.6% of assets, concentrated real estate
    score += 8

    # Aggregate holdings YTD momentum (equal weight approx of listed)
    if holdings_momentum is None:
        ys = list(HOLDINGS_YTD.values())
        holdings_momentum = float(np.nanmean(ys)) if ys else 0.0
    score += np.clip(holdings_momentum * 1.5, -25, 25)

    # Management fee competitive
    score += 5

    return float(np.clip(score, -100, 100))


def holdings_momentum_score() -> float:
    ys = list(HOLDINGS_YTD.values())
    return float(np.nanmean(ys)) if ys else 0.0
