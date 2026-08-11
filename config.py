"""Per-asset configuration for SRT (CSOP iEdge S-REIT Leaders Index ETF) multi-horizon predictor."""

TARGET = "SRT.SI"
FALLBACK = "CLR.SI"

# CSOP iEdge S-REIT Leaders Index ETF top 10 holdings (June 2026 factsheet weights).
TOP_HOLDINGS = {
    "AJBU.SI": 0.1011,  # Keppel DC REIT
    "C38U.SI": 0.1009,  # CapitaLand Integrated Commercial Trust
    "A17U.SI": 0.0992,  # CapitaLand Ascendas REIT
    "M44U.SI": 0.0985,  # Mapletree Logistics Trust
    "J69U.SI": 0.0782,  # Frasers Centrepoint Trust
    "BUOU.SI": 0.0667,  # Frasers Logistics & Commercial Trust
    "ME8U.SI": 0.0666,  # Mapletree Industrial Trust
    "K71U.SI": 0.0517,  # Keppel REIT
    "N2IU.SI": 0.0480,  # Mapletree Pan Asia Commercial Trust (formerly Commercial)
    "T82U.SI": 0.0471,  # Suntec REIT
}

BANKS = []
BANK_WEIGHTS = {}

SECTORS = {
    "data_centre": ["AJBU.SI"],
    "retail_mixed": ["C38U.SI", "J69U.SI"],
    "industrial_logistics": ["A17U.SI", "M44U.SI", "BUOU.SI", "ME8U.SI"],
    "office": ["K71U.SI", "T82U.SI"],
    "diversified": ["N2IU.SI"],
}

# Macro / rates / global REIT proxies most correlated with Singapore REITs.
MACRO = [
    "^STI",         # Straits Times Index
    "^HSI",         # Hang Seng (HK/China property flows)
    "VNQ",          # Vanguard Real Estate ETF
    "IYR",          # iShares US Real Estate ETF
    "SPY",          # S&P 500
    "QQQ",          # Nasdaq-100
    "DX-Y.NYB",     # US Dollar index
    "USDSGD=X",     # USD/SGD
    "^TNX",         # US 10Y yield
    "^VIX",         # VIX
    "HG=F",         # Copper
    "GC=F",         # Gold
    "CL=F",         # Crude oil
]

UNIVERSE = [TARGET, FALLBACK] + list(TOP_HOLDINGS.keys()) + MACRO

EARNINGS_HOLDINGS = TOP_HOLDINGS
