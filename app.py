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
st.title("📈 Regime-Based Investment Strategy Dashboard")

# === LOAD PRICE DATA ===
st.sidebar.header("Data Settings")
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

# === LOAD INPUT FILES ===
# if regime_file is not None:
#     regime_df = pd.read_csv(regime_file, parse_dates=["date"])
# else:
#     st.stop()

# if alloc_file is not None:
#     opt_alloc_df = pd.read_csv(alloc_file)
# else:
#     st.stop()

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
# === BACKTEST PORTFOLIO RETURNS ===
portfolio_returns = backtest(prices, returns, regime_df, allocations)

# === HEADER ===
st.markdown(
    "<h1 style='text-align: center; font-family: serif; font-size: 48px;'>📰 Regime Report</h1>"
    "<h3 style='text-align: center; font-weight: 300;'>Asset Allocation in Current Market Conditions</h3>",
    unsafe_allow_html=True
)

st.markdown("---")

# === TWO COLUMN LAYOUT ===
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("📊 Allocation for Current Regime")
    st.markdown(f"**Current Regime:** {current_regime}")

    # PIE CHART
    if current_regime in allocations:
        current_alloc = allocations[current_regime]

        custom_colors = {
            "stocks": "#00bf63",
            "stablecoins": "#ff5757",
            "cash": "#ff3131",
            "crypto": "#25a159",
            "gold": "#f4b70f",
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

    else:
        st.warning(f"No allocation found for regime: {current_regime}")

    # METRICS
    st.subheader("📈 Portfolio Performance")
    metrics = compute_metrics(portfolio_returns.dropna())
    st.write(
        pd.DataFrame(metrics, index=["Value"]).T.style.format("{:.2%}")
    )

    # OUTPERFORMANCE
    sp500 = yf.download("SPY", start=START_DATE, end=END_DATE)["Adj Close"].pct_change().dropna()
    sp500_cum = (1 + sp500).cumprod()
    portfolio_cum = (1 + portfolio_returns.dropna()).cumprod()

    outperformance = (portfolio_cum.iloc[-1] / sp500_cum.iloc[-1]) - 1
    st.metric(label="📊 Outperformance vs S&P 500", value=f"{outperformance:.2%}")

with right_col:
    st.subheader("🧠 Interpretation of Data")
    interp = st.text_area("What are we seeing in the macro environment?", height=150)

    st.subheader("🔭 Personal Outlook")
    outlook = st.text_area("Your thoughts on the market (e.g., technical signals)", height=150)

    st.subheader("✅ Conclusion")
    conclusion = st.text_area("Summarize your view and suggested action", height=100)



