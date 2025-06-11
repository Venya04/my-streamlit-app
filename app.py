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
    "gold": "GLD",
    "cash": None
}

st.set_page_config(page_title="Regime Report", layout="wide")

# === LOAD PRICE DATA ===
@st.cache_data
def load_csv_from_repo(path):
    return pd.read_csv(path, parse_dates=["date"] if "regime" in path else None)

regime_df = load_csv_from_repo("regime_labels_expanded.csv")
opt_alloc_df = load_csv_from_repo("optimal_allocations.csv")

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

returns = prices.pct_change().dropna()
returns["cash"] = CASH_DAILY_YIELD
returns["stablecoins"] = STABLECOIN_MONTHLY_YIELD

all_assets = set()
for regime_weights in allocations.values():
    all_assets.update(regime_weights.keys())

for asset in all_assets:
    if asset not in returns.columns:
        st.warning(f"Adding missing asset '{asset}' to returns with 0% yield.")
        returns[asset] = 0.0

# === BACKTEST ===
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

portfolio_returns = backtest(prices, returns, regime_df, allocations)

# === REGIME & ALLOCATION ===
latest_date = regime_df.index[-1]
current_regime = regime_df.loc[latest_date, "regime"]
current_alloc = allocations.get(current_regime, {})

# === HEADER ===
st.markdown(
    "<h1 style='text-align: center; font-family: serif; font-size: 48px;'>\U0001F4F0 Regime Report</h1>"
    "<h3 style='text-align: center; font-weight: 300;'>Asset Allocation in Current Market Conditions</h3>",
    unsafe_allow_html=True
)
st.markdown("---")

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("\U0001F4CA Allocation for Current Regime")
    st.markdown(f"**Current Regime:** {current_regime}")

    if current_alloc:
        fig_pie = px.pie(
            names=list(current_alloc.keys()),
            values=list(current_alloc.values()),
            hole=0.0,
            title="ALLOCATION %",
            color=list(current_alloc.keys()),
            color_discrete_map={
                "stocks": "#00bf63",
                "stablecoins": "#ff5757",
                "cash": "#ff3131",
                "crypto": "#25a159",
                "gold": "#f4b70f",
            }
        )
        fig_pie.update_traces(
            textinfo='percent',
            textfont_size=16,
            pull=[0.03] * len(current_alloc)
        )
        fig_pie.update_layout(
            title_font_size=24,
            showlegend=True,
            legend=dict(orientation="h", y=-0.2),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("\U0001F4C8 Portfolio Performance")
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
    metrics = compute_metrics(portfolio_returns.dropna())
    st.write(
        pd.DataFrame(metrics, index=["Value"]).T.style.format({
            "CAGR": "{:.2%}",
            "Volatility": "{:.2%}",
            "Max Drawdown": "{:.2%}",
            "Sharpe Ratio": "{:.2f}"
        })
    )

    sp500_raw = yf.download("SPY", start=START_DATE, end=END_DATE, progress=False)
sp500_series = sp500_raw["Adj Close"] if "Adj Close" in sp500_raw else sp500_raw["Close"]
sp500 = sp500_series.pct_change().dropna()
sp500_cum = (1 + sp500).cumprod()
portfolio_cum = (1 + portfolio_returns.dropna()).cumprod()

try:
    outperformance = (portfolio_cum.iloc[-1] / sp500_cum.iloc[-1]) - 1
    outperformance_value = outperformance.item() if hasattr(outperformance, "item") else outperformance

    st.metric(label="\U0001F4CA Outperformance vs S&P 500", value=f"{outperformance_value:.2%}")
except Exception as e:
    st.warning(f"Not enough data to calculate S&P 500 outperformance.\nError: {e}")


with right_col:
    st.subheader("\U0001F9E0 Interpretation of Data")
    interp = st.text_area("What are we seeing in the macro environment?", height=150)

    st.subheader("\U0001F52D Personal Outlook")
    outlook = st.text_area("Your thoughts on the market (e.g., technical signals)", height=150)

    st.subheader("\u2705 Conclusion")
    conclusion = st.text_area("Summarize your view and suggested action", height=100)

