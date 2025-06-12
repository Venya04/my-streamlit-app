# import plotly.express as px
# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf

# # === SETTINGS ===
# START_DATE = "2010-01-01"
# END_DATE = "2024-12-31"
# STABLECOIN_MONTHLY_YIELD = 0.05 / 12
# CASH_DAILY_YIELD = 0.045 / 12

# TICKERS = {
#     "stocks": "SPY",
#     "crypto": "BTC-USD",
#     "commodities": "GLD",
#     "cash": None
# }

# st.set_page_config(page_title="Regime Report", layout="wide")

# # === LOAD PRICE DATA ===
# @st.cache_data
# def load_csv_from_repo(path):
#     return pd.read_csv(path, parse_dates=["date"] if "regime" in path else None)

# regime_df = load_csv_from_repo("regime_labels_expanded.csv")
# opt_alloc_df = load_csv_from_repo("optimal_allocations.csv")

# @st.cache_data
# def load_prices():
#     data = {}
#     for asset, ticker in TICKERS.items():
#         if ticker:
#             df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
#             df = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
#             data[asset] = df
#     prices = pd.concat(data.values(), axis=1)
#     column_names = [k for k in data.keys() if data[k] is not None]
#     prices.columns = column_names
#     return prices.dropna()

# prices = load_prices()

# regime_df.set_index("date", inplace=True)
# regime_df = regime_df.reindex(prices.index, method="ffill")
# regime_df["regime"] = regime_df["regime"].str.capitalize()

# allocations = opt_alloc_df.set_index("regime").to_dict(orient="index")
# for alloc in allocations.values():
#     if "cash" not in alloc:
#         alloc["cash"] = 0.1
#     total = sum(alloc.values())
#     for k in alloc:
#         alloc[k] = alloc[k] / total

# returns = prices.pct_change().dropna()
# returns["cash"] = CASH_DAILY_YIELD
# returns["stablecoins"] = STABLECOIN_MONTHLY_YIELD

# all_assets = set()
# for regime_weights in allocations.values():
#     all_assets.update(regime_weights.keys())

# for asset in all_assets:
#     if asset not in returns.columns:
#         returns[asset] = 0.0

# # === BACKTEST ===
# def backtest(prices, returns, regime_df, allocations):
#     portfolio_returns = []
#     current_weights = {asset: 0.25 for asset in TICKERS.keys()}
#     prev_regime = None
#     for date in returns.index:
#         regime = regime_df.loc[date, "regime"]
#         if pd.isna(regime):
#             portfolio_returns.append(np.nan)
#             continue
#         if regime != prev_regime:
#             if regime in allocations:
#                 current_weights = allocations[regime]
#                 prev_regime = regime
#         daily_ret = sum(returns.loc[date, asset] * current_weights.get(asset, 0.0) for asset in current_weights)
#         portfolio_returns.append(daily_ret)
#     return pd.Series(portfolio_returns, index=returns.index)

# portfolio_returns = backtest(prices, returns, regime_df, allocations)

# # === REGIME & ALLOCATION ===
# latest_date = regime_df.index[-1]
# current_regime = regime_df.loc[latest_date, "regime"]
# current_alloc = allocations.get(current_regime, {})

# # === HEADER ===
# st.markdown(
#     """
#     <style>
#    @import url('https://fonts.googleapis.com/css2?family=UnifrakturCook:wght@700&display=swap');


#     .gothic-title {
#         font-family: 'UnifrakturCook', serif;
#         text-align: center;
#         font-size: 60px;
#         font-weight: bold;
#         padding: 0.5rem 0;
#         letter-spacing: 1px;
#     }
#     .pub-info {
#         text-align: center;
#         font-family: 'Georgia', serif;
#         font-size: 13px;
#         margin-top: -18px;
#         color: #ccc;
#     }
#     </style>
#     <div class='gothic-title'>The Regime Report</div>
#     <div class='pub-info'>No. 01 · Published biWeekly · Market Bulletin · June 2025</div>
#     <h3 style='text-align: center; font-family: Georgia, serif; font-style: italic; margin-top: -10px;'>
#         Asset Allocation in Current Market Conditions
#     </h3>
#     """,
#     unsafe_allow_html=True
# )

# st.markdown(
#     """
#     <style>
#         .block-container {
#             padding-left: 10rem;
#             padding-right: 10rem;
#         }
#         .section-title {
#             font-family: Georgia, serif;
#             font-size: 18px;
#             font-weight: bold;
#             text-transform: uppercase;
#             margin-bottom: 6px;
#             color: #d4af37;
#             border-bottom: 1px solid #555;
#             padding-bottom: 4px;
#         }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # === MAIN LAYOUT ===
# left_col, right_col = st.columns([1.3, 1])

# with left_col:
#     left_box_style = """
#         <style>
#             .left-section-title {
#                 font-family: Georgia, serif;
#                 font-size: 18px;
#                 font-weight: bold;
#                 text-transform: uppercase;
#                 margin-bottom: 10px;
#                 text-align: center;
#             }
#         </style>
#     """
#     st.markdown(left_box_style, unsafe_allow_html=True)

#     # === PIE CHART ===
#     if current_alloc:
#         fig_pie = px.pie(
#             names=list(current_alloc.keys()),
#             values=list(current_alloc.values()),
#             hole=0.0,
#             color=list(current_alloc.keys()),
#             # color_discrete_map={
#             #     "stocks": "#00bf63",
#             #     "stablecoins": "#ff5757",
#             #     "cash": "#ff3131",
#             #     "crypto": "#25a159",
#             #     "commodities": "#f4b70f",
#             # }

# # color_discrete_map={
# #     "stocks": "#4b5320",        # Dark Olive
# #     "stablecoins": "#7c6c57",   # Faded Taupe
# #     "cash": "#b1a296",          # Vintage Beige
# #     "crypto": "#5e5a59",        # Charcoal Gray
# #     "commodities": "#aa8c5f",   # Bronze Gold
# # }

# # color_discrete_map = {
# #     "stocks": "#1E3A5F",       # Deep Navy
# #     "stablecoins": "#7C6C57",  # Faded Taupe
# #     "cash": "#B1A296",         # Vintage Beige
# #     "crypto": "#5E5A59",       # Charcoal Gray
# #     "commodities": "#AA8C5F",  # Bronze Gold
# # }
# # color_discrete_map = {
# #     "stocks": "#444444",       # Dark Gray
# #     "stablecoins": "#666666",  # Mid Gray
# #     "cash": "#888888",         # Soft Gray
# #     "crypto": "#AAAAAA",       # Light Gray
# #     "commodities": "#CCCCCC",  # Pale Gray
# # }

# color_discrete_map = {
#     "stocks": "#102030",       # Slate Navy
#     "stablecoins": "#3A3A3A",  # Gunmetal Gray
#     "cash": "#5C5149",         # Smoked Taupe
#     "crypto": "#2F4F4F",       # Dark Slate Gray
#     "commodities": "#6B4E23",  # Aged Bronze
# }
#         )
#         fig_pie.update_traces(
#             textinfo='percent',
#             textfont_size=16,
#             pull=[0.03] * len(current_alloc),
#             marker=dict(line=dict(color="#000000", width=2))
#         )
#         fig_pie.update_layout(
#             showlegend=False,
#             margin=dict(t=10, b=10, l=10, r=10),
#             paper_bgcolor='rgba(0,0,0,0)',
#             plot_bgcolor='rgba(0,0,0,0)',
#         )
#         st.plotly_chart(fig_pie, use_container_width=True)

#     # === PORTFOLIO HOLDINGS RIGHT UNDER PIE CHART ===
#     st.markdown("<div class='left-section-title'>Portfolio Holdings</div>", unsafe_allow_html=True)
#     st.markdown("""
#         <div style='text-align: center; margin-top: -5px;'>
#             <ul style='padding-left: 10; list-style-position: inside; text-align: left; display: inline-block;'>
#     """ + "".join([
#         f"<li><strong>{asset.capitalize()}</strong>: {weight:.1%}</li>"
#         for asset, weight in current_alloc.items()
#     ]) + """
#             </ul>
#         </div>
#     """, unsafe_allow_html=True)


# # with right_col:
# #     right_box_style = """
# #         <style>
# #             .left-section-title {
# #                 font-family: Georgia, serif;
# #                 font-size: 18px;
# #                 font-weight: bold;
# #                 text-transform: uppercase;
# #                 margin-bottom: 10px;
# #                 text-align: center;
# #             }
# #         </style>
# #     """
# #     st.markdown(left_box_style, unsafe_allow_html=True)

# #     cols = st.columns([0.6, 0.1])
# #     with cols[0]:
# #         st.markdown("<div class='section-title'>Market Insight</div>", unsafe_allow_html=True)
# #         interp = st.text_area("What are we seeing in the macro environment?", height=130)

# #     st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)

# #     cols = st.columns([0.6, 0.1])
# #     with cols[0]:
# #         st.markdown("<div class='section-title'>Top Strategy Note</div>", unsafe_allow_html=True)
# #         outlook = st.text_area("Thoughts on the market (e.g., technical signals)", height=130)

# #     st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)

# #     cols = st.columns([0.6, 0.1])
# #     with cols[0]:
# #         st.markdown("<div class='section-title'>Trader's Conclusion</div>", unsafe_allow_html=True)
# #         conclusion = st.text_area("Summary and suggested action", height=130)

# # current right col
# with right_col:
#     right_box_style = """
#         <style>
#             .section-title {
#                 font-family: Georgia, serif;
#                 font-size: 18px;
#                 font-weight: bold;
#                 text-transform: uppercase;
#                 margin-bottom: 10px;
#                 text-align: left;
#                 color: white;
#                 border-bottom: 1px solid #555;
#                 padding-bottom: 4px;
#             }
#         </style>
#     """
#     st.markdown(right_box_style, unsafe_allow_html=True)  # ✅ Apply the right_box_style, not left_box_style

#     for title, placeholder in [
#         ("Market Insight", "What are we seeing in the macro environment?"),
#         ("Top Strategy Note", "Thoughts on the market (e.g., technical signals)"),
#         ("Trader's Conclusion", "Summary and suggested action")
#     ]:
#         cols = st.columns([0.6, 0.1])
#         with cols[0]:
#             st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
#             st.text_area(placeholder, height=130)
#         st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)

import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# === SETTINGS ===
START_DATE = "2010-01-01"
END_DATE = "2024-12-31"
STABLECOIN_MONTHLY_YIELD = 0.05 / 12
CASH_DAILY_YIELD = 0.045 / 252

TICKERS = {
    "stocks": "SPY",
    "crypto": "BTC-USD",
    "commodities": "GLD",
    "cash": None,
    "stablecoins": None
}

st.set_page_config(page_title="Regime Report", layout="wide")

# === LOAD DATA ===
@st.cache_data
def load_csv_from_repo(path):
    try:
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return pd.DataFrame()

regime_df = load_csv_from_repo("regime_labels_expanded.csv")
opt_alloc_df = load_csv_from_repo("optimal_allocations.csv")

@st.cache_data
def load_prices():
    data = {}
    for asset, ticker in TICKERS.items():
        if ticker:
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)

            # Extract price series
            if "Adj Close" in df.columns:
                series = df["Adj Close"]
            elif "Close" in df.columns:
                series = df["Close"]
            else:
                continue  # Skip if no usable price column

            # Ensure it's a DataFrame with the correct column name
            df_clean = pd.DataFrame(series)
            df_clean.columns = [asset]

            data[asset] = df_clean

    if not data:
        return pd.DataFrame()

    prices = pd.concat(data.values(), axis=1)
    return prices.dropna()

prices = load_prices()

# === VALIDATE DATA ===
if prices.empty:
    st.error("Price data failed to load.")
    st.stop()

if regime_df.empty or "regime" not in regime_df.columns:
    st.error("Regime data missing or malformed.")
    st.stop()

# === PREPARE DATA ===
regime_df.set_index("date", inplace=True)
regime_df = regime_df.asfreq("D").ffill().reindex(prices.index, method="ffill")
regime_df["regime"] = regime_df["regime"].str.capitalize()

allocations = opt_alloc_df.set_index("regime").to_dict(orient="index")
for alloc in allocations.values():
    if "cash" not in alloc:
        alloc["cash"] = 0.1
    total = sum(alloc.values())
    for k in alloc:
        alloc[k] /= total

# === RETURNS ===
returns = prices.pct_change().dropna()
returns["cash"] = CASH_DAILY_YIELD
returns["stablecoins"] = (1 + STABLECOIN_MONTHLY_YIELD)**(1/22) - 1

all_assets = set()
for alloc in allocations.values():
    all_assets.update(alloc.keys())

for asset in all_assets:
    if asset not in returns.columns:
        returns[asset] = 0.0

# === BACKTEST ===
def backtest(returns, regime_df, allocations):
    portfolio_returns = []
    current_weights = {asset: 0.25 for asset in TICKERS}
    prev_regime = None
    for date in returns.index:
        regime = regime_df.loc[date, "regime"]
        if pd.isna(regime):
            portfolio_returns.append(np.nan)
            continue
        if regime != prev_regime and regime in allocations:
            current_weights = allocations[regime]
            prev_regime = regime
        ret = sum(returns.loc[date, asset] * current_weights.get(asset, 0) for asset in current_weights)
        portfolio_returns.append(ret)
    return pd.Series(portfolio_returns, index=returns.index)

portfolio_returns = backtest(returns, regime_df, allocations)

# === GET CURRENT REGIME ===
latest_date = regime_df.index[-1]
current_regime = regime_df.loc[latest_date, "regime"]
current_alloc = allocations.get(current_regime, {})

# === HEADER ===
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=UnifrakturCook:wght@700&display=swap');
    .gothic-title {
        font-family: 'UnifrakturCook', serif;
        text-align: center;
        font-size: 3.5rem;
        font-weight: bold;
        padding: 0.5rem 0;
        letter-spacing: 1px;
    }
    .pub-info {
        text-align: center;
        font-family: 'Georgia', serif;
        font-size: 0.8rem;
        margin-top: -18px;
        color: #ccc;
    }
    </style>
    <div class='gothic-title'>The Regime Report</div>
    <div class='pub-info'>No. 01 · Published biWeekly · Market Bulletin · June 2025</div>
    <h3 style='text-align: center; font-family: Georgia, serif; font-style: italic; margin-top: -10px;'>
        Asset Allocation in Current Market Conditions
    </h3>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
        .block-container {
            padding-left: 5vw;
            padding-right: 5vw;
        }
        .section-title {
            font-family: Georgia, serif;
            font-size: 1.1rem;
            font-weight: bold;
            text-transform: uppercase;
            margin-bottom: 6px;
            color: #d4af37;
            border-bottom: 1px solid #555;
            padding-bottom: 4px;
        }
    </style>
""", unsafe_allow_html=True)

# === LAYOUT ===
left_col, right_col = st.columns([1.3, 1])

with left_col:
    st.markdown("""
        <style>
            .left-section-title {
                font-family: Georgia, serif;
                font-size: 1.1rem;
                font-weight: bold;
                text-transform: uppercase;
                margin-bottom: 10px;
                text-align: center;
            }
        </style>
    """, unsafe_allow_html=True)

    if current_alloc:
        fig_pie = px.pie(
            names=list(current_alloc.keys()),
            values=list(current_alloc.values()),
            hole=0,
            color=list(current_alloc.keys()),
            color_discrete_map={
                "stocks": "#102030",
                "stablecoins": "#3A3A3A",
                "cash": "#5C5149",
                "crypto": "#2F4F4F",
                "commodities": "#6B4E23",
            }
        )
        fig_pie.update_traces(
            textinfo='percent',
            textfont_size=16,
            pull=[0.03] * len(current_alloc),
            marker=dict(line=dict(color="#000000", width=2))
        )
        fig_pie.update_layout(
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("<div class='left-section-title'>Portfolio Holdings</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='text-align: center; margin-top: -5px;'>
            <ul style='padding-left: 10; list-style-position: inside; text-align: left; display: inline-block;'>
        """ + "".join([
            f"<li><strong>{asset.capitalize()}</strong>: {weight:.1%}</li>"
            for asset, weight in current_alloc.items()
        ]) + """
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

with right_col:
    st.markdown("""
        <style>
            .section-title {
                font-family: Georgia, serif;
                font-size: 18px;
                font-weight: bold;
                text-transform: uppercase;
                margin-bottom: 10px;
                text-align: left;
                color: white; /* white for right section */
                border-bottom: 1px solid #555;
                padding-bottom: 4px;
            }
            @media (max-width: 768px) {
                .section-title {
                    font-size: 14px;
                }
            }
        </style>
    """, unsafe_allow_html=True)

    # Section structure: title = shown in styled header, placeholder = shown in textarea as italic prompt
    for title, placeholder in [
        ("Market Insight", "*What are we seeing in the macro environment?*"),
        ("Top Strategy Note", "*Thoughts on the market (e.g., technical signals)*"),
        ("Trader's Conclusion", "*Summary and suggested action*")
    ]:
        cols = st.columns([0.9, 0.1])
        with cols[0]:
            st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
            st.text_area(
                label="",
                placeholder=placeholder,
                height=130,
                label_visibility="collapsed"
            )
        st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)
