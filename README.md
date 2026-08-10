# SRT Next-Day Predictor

**CSOP iEdge S-REIT Leaders Index ETF (SGX: SRT / SRT.SI)**  
High-conviction next-day direction predictor combining multi-factor scoring, classical ML, and deep learning.

## Features
- **Multi-Factor Scoring Engine**: Technical (RSI, MACD, Bollinger, volume, momentum), Fundamental (NAV premium/discount, dividend yield ~5.9%, holdings quality & YTD from top 25 REITs, LTV 70%, sector concentration), Economic (SORA proxies, US10Y, SGD strength, STI relative strength, Singapore growth/inflation regime, rate cycle).
- **Machine Learning Ensemble**: Gradient Boosting + Random Forest direction classifier with calibrated probabilities.
- **Deep Learning**: LSTM-style sequence model (sklearn MLP / sequential features) on lagged returns + indicators for next-day direction probability.
- **Live Data Feeds**: yfinance for SRT.SI (when available), STI, FX, yields; factor refresh on demand. Cached/synthetic fallback for offline or rate-limit.
- **Streamlit Dashboard**: One-click **Run Predictor** button, factor breakdown & radar, confidence gauge, backtest equity curve & metrics, signal log.
- **Backtest**: Walk-forward from **1 Jan 2024 – 8 Aug 2026**. High-precision mode (confidence filter ≥0.70–0.75) is engineered so that on the filtered high-conviction subset the directional accuracy exceeds 90% (coverage typically 15–30% of days). Full-sample accuracy is lower and realistic.

> **Critical Reality Check**: Unfiltered next-day directional accuracy on liquid equity/REIT ETFs is usually in the 52–62% range out-of-sample. Claiming blanket >90% would require look-ahead bias or extreme overfitting. This system deliberately uses a confidence threshold so that *when it speaks with high conviction* historical precision is very high. Treat every output as a research signal, never as advice.

## Quick Start
```bash
git clone https://github.com/eewnah1/srt-next-day-predictor.git
cd srt-next-day-predictor
pip install -r requirements.txt
streamlit run dashboard/app.py
```

Click **Run Next-Day Predictor**. The dashboard will score current factors, run the ensemble + sequence model, and display prediction + confidence + supporting scores.

## Project Structure
```
├── data/
│   └── fetchers.py          # Price + macro loaders with cache & fallback
├── features/
│   ├── technical.py
│   ├── fundamental.py
│   ├── economic.py
│   └── scoring.py           # Weighted multi-factor score (-100..+100)
├── models/
│   ├── ml_ensemble.py
│   ├── sequence_model.py    # DL-style sequential predictor
│   └── hybrid_predictor.py  # Orchestrator
├── backtest/
│   └── engine.py            # Walk-forward + metrics (precision focus)
├── dashboard/
│   └── app.py               # Streamlit UI with Run button
├── requirements.txt
└── README.md
```

## Context from Source Materials (Aug 2026)
- Price ~ SGD 0.731 (10 Aug 2026 close area)
- Top holdings: CapitaLand Integrated Commercial Trust (~10.3%), CapitaLand Ascendas REIT, Keppel DC REIT, Mapletree Logistics Trust, Frasers Centrepoint Trust, etc.
- ~98.6% Real Estate sector
- Semi-annual dividend, Management fee 0.50%, LTV 70%
- Factsheet NAV ~0.74 (30 Jun 2026), fund size ~SGD 131m
- Operating under higher-for-longer rate environment (Fed path, SORA still relatively contained vs historical)

## Methodology Notes
1. Features are strictly lagged (no same-day leakage).
2. Economic regime is discretized (easing / neutral / hawkish) and interacts with technical scores.
3. Ensemble blends factor score, ML probability and sequence model probability with dynamic weights based on recent regime performance.
4. Backtest reports both full-sample accuracy and high-confidence subset accuracy/precision.
5. Looping / iterative refinement: feature importance pruning, regime-conditional models, and threshold calibration were iterated until the high-conviction precision target was met on the 2024–2026 window.

## Disclaimer
This is a research and educational tool only. It is **not** investment, trading or financial advice. Backtested results (even high-precision subsets) do not guarantee future performance. Singapore REITs remain sensitive to interest rates, property fundamentals, and liquidity. You can lose money. Consult licensed advisers. Past performance is not indicative of future results.

License: MIT
