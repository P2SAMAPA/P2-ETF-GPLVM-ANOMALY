"""
Configuration for GPLVM + Warped GP Engine.
"""
HF_TOKEN = "your_token_here"
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-gplvm-results"

# Universe definitions (same as before)
UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
    ]
}

# Macro factor columns (adjust based on actual master_data.parquet)
# Common names from typical datasets: "VIX", "DXY", "US10Y", "US2Y", "CRB", "GOLD", etc.
MACRO_COLUMNS = ["VIX", "DXY", "US10Y", "US2Y", "CRB", "GOLD"]

# Engine parameters
ROLLING_WINDOW = 252
LATENT_VARIANCE_THRESHOLD = 0.95   # auto‑select dimensions covering 95% variance
N_LATENT_FORECAST_DAYS = 20       # use last N days to fit time‑series GP
TOP_N = 3

# GP kernel hyperparameters (fixed for speed)
KERNEL_LENGTH_SCALE = 1.0
KERNEL_VARIANCE = 1.0
