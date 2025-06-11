import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# === SETTINGS ===
START_DATE = "2010-01-01"
END_DATE = "2024-12-31"
STABLECOIN_MONTHLY_YIELD = 0.05 / 12
CASH_DAILY_YIELD = 0.045 / 12

TICKERS = {
    "stocks": "SPY",
    "crypto": "BTC-USD",
    "commodities": "GLD",
    "cash": None
}

st.set_page_config(page_title="Regime-Based Investment Dashboard", layout="wide")
st.title("📈 Regime-Based Investment Strategy Dashboard")

# === LOAD PRICE DATA ===
st.sidebar.header("Data Settings")
regime_file = st.sidebar.file_uploader("Upload Regime Labels CSV", type="csv")
alloc_file = st.sidebar.file_uploader("Upload Optimal Allocations CSV", type="csv")

@st.cache_data
def load_prices():
    data = {}
    for asset, ticker in TICKERS.items():
        if ticker:
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            df = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
            data[asset] = df
    prices = pd.concat(data.values(), axis=1)
    prices.columns = [k for k in data.keys() if data[k] is not None]
    return prices.dropna()

prices = load_prices()

# === LOAD INPUT FILES ===
if regime_file is not None:
    regime_df = pd.read_csv(regime_file, parse_dates=["date"])
else:
    st.stop()

if alloc_file is not None:
    opt_alloc_df = pd.read_csv(alloc_file)
else:
    st.stop()

regime_df.set_index("date", inplace=True)
regime_df = regime_df.reindex(prices.index, method="ffill")
regime_df["regime"] = regime_df["regime"].str.capitalize()

allocations = opt_alloc_df.set_index("regime").to_dict(orient="index")

for alloc in allocations.values():
    if "cash" not in alloc:
        alloc["cash"] = 0.1
    total = sum(alloc.values())
    for k in alloc:
        alloc[k] = alloc[k] / total

# === CALCULATE RETURNS ===
returns = prices.pct_change().dropna()
returns["cash"] = CASH_DAILY_YIELD
returns["stablecoins"] = STABLECOIN_MONTHLY_YIELD

# === Ensure all assets used in allocations exist in returns ===
all_assets = set()
for regime_weights in allocations.values():
    all_assets.update(regime_weights.keys())

for asset in all_assets:
    if asset not in returns.columns:
        st.warning(f"Adding missing asset '{asset}' to returns with 0% yield.")
        returns[asset] = 0.0


# === BACKTEST FUNCTION ===
def backtest(prices, returns, regime_df, allocations):
    portfolio_returns = []
    current_weights = {asset: 0.25 for asset in TICKERS.keys()}
    prev_regime = None

    for date in returns.index:
        regime = regime_df.loc[date, "regime"]
        if pd.isna(regime):
            portfolio_returns.append(np.nan)
            continue
        if regime != prev_regime:
            if regime in allocations:
                current_weights = allocations[regime]
                prev_regime = regime

        daily_ret = sum(returns.loc[date, asset] * current_weights.get(asset, 0.0) for asset in current_weights)
        portfolio_returns.append(daily_ret)

    return pd.Series(portfolio_returns, index=returns.index)

# === RUN BACKTEST ===
# === CURRENT REGIME & ALLOCATION PIE CHART ===
st.subheader("🧭 Current Regime Allocation")

# Get the latest known regime
latest_date = regime_df.index[-1]
current_regime = regime_df.loc[latest_date, "regime"]

if current_regime in allocations:
    current_alloc = allocations[current_regime]

    # Create interactive pie chart with Plotly
    # === CUSTOMIZED PIE CHART FOR CURRENT ALLOCATION ===
st.subheader("📊 Allocation %")

latest_date = regime_df.index[-1]
current_regime = regime_df.loc[latest_date, "regime"]

if current_regime in allocations:
    current_alloc = allocations[current_regime]

    # Set custom colors (adjust as needed)
    custom_colors = {
        "stocks": "#00bf63",       # green
        "stablecoins": "#ff5757",  # light red
        "cash": "#ff3131",         # red
        "crypto": "#25a159",       # light green
        "commodities": "#f4b70f",  # yellow
    }

    fig_pie = px.pie(
        names=list(current_alloc.keys()),
        values=list(current_alloc.values()),
        hole=0.0,
        title="ALLOCATION %",
        color=list(current_alloc.keys()),
        color_discrete_map=custom_colors
    )

    fig_pie.update_traces(
        textinfo='percent',
        textfont_size=16,
        pull=[0.03] * len(current_alloc)  # slightly "explode" all slices
    )

    fig_pie.update_layout(
        title_font_size=24,
        showlegend=True,
        legend=dict(orientation="h", y=-0.2),
        paper_bgcolor='rgba(0,0,0,0)',  # transparent background
        plot_bgcolor='rgba(0,0,0,0)',
    )

    left, _ = st.columns([2, 1])
    with left:
        st.plotly_chart(fig_pie, use_container_width=True)
else:
    st.warning(f"No allocation found for regime: {current_regime}")



# === METRICS ===
def compute_metrics(rets):
    mean_daily = rets.mean()
    std_daily = rets.std()
    cagr = (1 + mean_daily) ** 252 - 1
    volatility = std_daily * np.sqrt(252)
    sharpe = (mean_daily / std_daily) * np.sqrt(252)
    drawdown = (1 + rets).cumprod().div((1 + rets).cumprod().cummax()) - 1
    max_dd = drawdown.min()
    return {
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd
    }


