import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from src.data_fetcher import (
    POPULAR_INDIAN_STOCKS, SECTOR_STOCKS, fetch_stock_data, get_stock_info,
    get_stock_news, fetch_multiple_stocks,
    fetch_intraday_data, get_stock_fundamentals,
    get_sector_for_stock,
)
from src.hedge_trading import calculate_correlation_matrix, calculate_sharpe_ratio
from src.feature_engineering import add_technical_indicators
from src.predictor import train_and_predict, predict_with_saved_model
# sentiment imports done inline where needed (src.sentiment._parse_sentiment_response)
from src.model import load_saved_model
from src.prediction_tracker import (
    init_db, log_predictions, validate_pending_predictions,
    get_accuracy_metrics, get_prediction_history, get_tracked_tickers,
    should_relearn, relearn_models, compute_adaptive_weights,
    get_current_model_version,
)
from datetime import date, datetime
import pytz
from src.swing_trading import (
    calculate_pivot_points, calculate_support_resistance,
    calculate_fibonacci_retracements, calculate_atr_stop_loss,
    generate_swing_signals, calculate_trade_setup, identify_swing_patterns,
)
from src.scalping import (
    add_scalping_indicators, generate_scalp_signal,
    get_scalping_levels, get_market_microstructure,
)
from src.paper_trading import (
    init_paper_db, place_trade, check_open_trades,
    close_trade, delete_trade, get_open_trades, get_trade_history, get_paper_stats,
    clear_all_trades, init_paper_funds, get_fund_balance, set_fund_capital,
)

st.set_page_config(page_title="Indian Stock Predictor", page_icon="📈", layout="wide")

# ============================================================
# PREDICTION TRACKING — init DB & auto-validate on load
# ============================================================
init_db()
init_paper_db()
init_paper_funds()

@st.cache_data(ttl=300)
def _auto_validate():
    return validate_pending_predictions()

@st.cache_data(ttl=300)
def _auto_check_paper_trades():
    return check_open_trades()

_auto_validate()
_auto_check_paper_trades()

# ============================================================
# CLEAN DARK THEME CSS
# ============================================================
st.markdown("""<style>
    /* === BASE RESET === */
    .block-container { padding-top: 1rem; max-width: 1200px; }
    h4, h3, h2 { color: #e8e8f8 !important; font-weight: 700 !important; letter-spacing: -0.3px; }

    /* === ANIMATIONS (kept minimal) === */

    /* === ACTION CARDS === */
    .action-card {
        padding: 24px 24px; border-radius: 16px; text-align: center;
        font-size: 24px; font-weight: 800; color: #fff; margin: 12px 0;
        border: none;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        letter-spacing: 0.2px;
        position: relative;
        overflow: hidden;
        transition: box-shadow 0.25s ease;
    }
    .action-card:hover {
        box-shadow: 0 12px 40px rgba(0,0,0,0.5);
    }
    .action-card::before {
        content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, rgba(255,255,255,0.1), rgba(255,255,255,0.35), rgba(255,255,255,0.1));
    }
    .action-card span { display: block; font-size: 14px; font-weight: 500; opacity: 0.92; margin-top: 8px; line-height: 1.5; }
    .action-buy, .action-long, .action-up   { background: linear-gradient(135deg, #047857, #059669, #10b981); }
    .action-sell, .action-short, .action-down { background: linear-gradient(135deg, #991b1b, #dc2626, #f87171); }
    .action-hold, .action-sideways           { background: linear-gradient(135deg, #78350f, #b45309, #f59e0b); }
    .action-notrade                          { background: linear-gradient(135deg, #111827, #1f2937, #374151); }

    /* === BADGE === */
    .badge-best {
        display: inline-block; background: linear-gradient(135deg, #f59e0b, #fbbf24);
        color: #1a1a2e; font-size: 11px; font-weight: 800; text-transform: uppercase;
        letter-spacing: 0.8px; padding: 4px 12px; border-radius: 20px;
        margin-left: 8px; vertical-align: middle; line-height: 1;
        box-shadow: 0 2px 12px rgba(245,158,11,0.35);
        box-shadow: 0 0 8px rgba(0,212,255,0.15);
    }
    .badge-rank {
        display: inline-block; background: rgba(0,212,255,0.12);
        color: #00d4ff; font-size: 11px; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.6px; padding: 4px 12px; border-radius: 20px;
        margin-left: 8px; vertical-align: middle; line-height: 1;
        border: 1px solid rgba(0,212,255,0.25);
        backdrop-filter: blur(4px);
    }

    /* === PROFIT / LOSS TEXT === */
    .profit, .setup-card .profit, .setup-card .profit b { color: #34d399 !important; font-weight: 700; }
    .loss, .setup-card .loss, .setup-card .loss b { color: #f87171 !important; font-weight: 700; }
    .charges-dim, .setup-card .charges-dim { color: #6b6b80 !important; font-size: 13px; }

    /* === HELP BOX === */
    .help-box {
        background: rgba(0,212,255,0.03); border-left: 3px solid rgba(0,212,255,0.35);
        padding: 14px 18px; border-radius: 0 10px 10px 0; margin: 10px 0;
        font-size: 13px; color: #9999b0; line-height: 1.7;
        backdrop-filter: blur(4px);
    }
    .help-box b { color: #c0c0d8; }

    /* === SETUP CARD === */
    .setup-card {
        background: linear-gradient(135deg, #12122a, #141432); padding: 20px 22px; border-radius: 14px;
        border: 1px solid #1e1e3a; margin: 8px 0;
        color: #c8c8e0; line-height: 1.9; font-size: 15px;
        transition: border-color 0.3s ease;
    }
    .setup-card:hover {
        border-color: #2a2a50;
    }
    .setup-card b { color: #e0e0f0; font-size: 15px; }

    /* === VERDICT BOX === */
    .verdict-box {
        padding: 14px 18px; border-radius: 10px; margin: 8px 0;
        font-size: 14px; color: #c8c8e0; line-height: 1.7;
    }
    .verdict-box b { color: #e8e8f8; }
    .good    { background: rgba(16,185,129,0.06); border: 1px solid rgba(16,185,129,0.2); border-left: 4px solid #10b981; }
    .bad     { background: rgba(239,68,68,0.06);  border: 1px solid rgba(239,68,68,0.2);  border-left: 4px solid #ef4444; }
    .neutral { background: rgba(245,158,11,0.06); border: 1px solid rgba(245,158,11,0.2); border-left: 4px solid #f59e0b; }

    /* === CHECKLIST === */
    .checklist {
        background: linear-gradient(135deg, #12122a, #141432); padding: 16px 20px; border-radius: 12px; margin: 10px 0;
        color: #c8c8e0; line-height: 2.1; font-size: 14px; border: 1px solid #1e1e38;
    }
    .checklist b { color: #e0e0f0; font-size: 15px; }

    /* === METRIC CARDS === */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #12122a, #141432); border: 1px solid #1e1e3a; border-radius: 14px; padding: 16px;
        transition: border-color 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(0,212,255,0.25);
    }
    [data-testid="stMetricLabel"] { color: #7777a0 !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 1px; font-weight: 700 !important; }
    [data-testid="stMetricValue"] { color: #e8e8f8 !important; font-weight: 800 !important; font-size: 1.3rem !important; }
    [data-testid="stMetricDelta"] > div { font-size: 13px !important; }

    /* === TABS === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 3px; background: rgba(13,13,30,0.8); border-radius: 14px; padding: 5px; border: 1px solid #1e1e38;
        backdrop-filter: blur(8px);
        overflow-x: auto;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px; padding: 10px 18px; color: #7777a0; font-weight: 600; font-size: 13.5px;
        transition: all 0.25s ease; white-space: nowrap;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #c0c0e0; background: rgba(255,255,255,0.04); }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00bce0, #00d4ff) !important;
        color: #0a0a1a !important; font-weight: 700;
        box-shadow: 0 4px 16px rgba(0,212,255,0.25);
    }

    /* === BUTTONS === */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00a8cc, #00d4ff) !important; color: #0a0a1a !important;
        border: none !important; border-radius: 10px !important; font-weight: 700 !important;
        font-size: 14px !important; padding: 10px 22px !important;
        box-shadow: 0 4px 16px rgba(0,212,255,0.2);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 24px rgba(0,212,255,0.4) !important;
    }
    .stButton > button:not([kind="primary"]) {
        border-radius: 10px !important;
        border: 1px solid #2a2a4a !important;
        transition: all 0.25s ease;
    }
    .stButton > button:not([kind="primary"]):hover {
        border-color: #3a3a5a !important;
        background: rgba(255,255,255,0.03) !important;
    }

    /* === GRADIENT TITLE === */
    .app-title {
        text-align: center; padding: 10px 0 4px 0;
    }
    .app-title h1 {
        background: linear-gradient(135deg, #00d4ff 0%, #a855f7 50%, #f59e0b 100%);
        background-size: 200% auto;
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 2.4rem; font-weight: 800; margin: 0; letter-spacing: -0.5px;
    }
    .app-title p { color: #6b6b80; font-size: 13px; margin: 6px 0 0 0; letter-spacing: 0.3px; }

    /* === SCREENER PICK CARD === */
    .screener-pick {
        background: linear-gradient(135deg, #12122a, #141432); padding: 20px; border-radius: 14px;
        border: 1px solid #1e1e38; margin: 8px 0;
        color: #c8c8e0; line-height: 1.8; font-size: 14px;
        transition: border-color 0.3s ease;
    }
    .screener-pick:hover { border-color: #2a2a50; }
    .screener-pick b { color: #e0e0f0; }
    .screener-pick .price { font-size: 22px; font-weight: 800; color: #00d4ff; }
    .screener-pick .detail { font-size: 13px; color: #8888a8; }

    /* === SECTION DIVIDER === */
    .section-label {
        font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px;
        color: #555578; margin: 20px 0 10px 0; padding-bottom: 8px;
        border-bottom: 1px solid #1e1e38;
    }

    /* === STOCK BANNER === */
    .stock-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #1a1a30 100%);
        border: 1px solid #2a2a45; border-radius: 18px;
        padding: 28px 32px; margin-bottom: 20px;
        position: relative; overflow: hidden;
        transition: border-color 0.3s ease;
    }
    .stock-banner:hover { border-color: #3a3a55; }
    .stock-banner::after {
        content: ""; position: absolute; top: -50%; right: -20%; width: 300px; height: 300px;
        background: radial-gradient(circle, rgba(0,212,255,0.04) 0%, transparent 70%);
        pointer-events: none;
    }

    /* === QUICK STATS ROW === */
    .quick-stats {
        display: flex; gap: 12px; margin-bottom: 18px; flex-wrap: wrap;
    }
    .quick-stat {
        background: rgba(18,18,42,0.7); border: 1px solid #1e1e3a; border-radius: 12px;
        padding: 14px 20px; flex: 1; min-width: 140px; text-align: center;
        backdrop-filter: blur(4px);
        transition: border-color 0.3s ease;
    }
    .quick-stat:hover { border-color: rgba(0,212,255,0.2); }
    .quick-stat .qs-label { font-size: 11px; color: #7777a0; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 700; }
    .quick-stat .qs-value { font-size: 18px; font-weight: 800; color: #e0e0f0; margin-top: 4px; }
    .quick-stat .qs-delta { font-size: 13px; font-weight: 600; margin-top: 2px; }

    /* === DATAFRAME STYLING === */
    .stDataFrame { font-size: 13px !important; }
    .stDataFrame [data-testid="stDataFrameResizable"] {
        border-radius: 12px; overflow: hidden;
        border: 1px solid #1e1e38 !important;
    }

    /* === FOOTER === */
    .footer-card {
        background: linear-gradient(135deg, #0d0d1e, #111128); border: 1px solid #1e1e38; border-radius: 12px;
        padding: 16px 20px; margin-top: 24px; text-align: center;
        font-size: 12px; color: #555570; line-height: 1.6;
    }
    .footer-card b { color: #b45309; }

    /* === PROMPT BOX === */
    .prompt-box {
        background: #0a0a1a; border: 1px solid #1e1e38; border-radius: 12px;
        padding: 18px; margin: 10px 0; font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 12px; color: #b0b0c8; line-height: 1.7;
        white-space: pre-wrap; word-wrap: break-word;
        max-height: 400px; overflow-y: auto;
    }
    .prompt-box b { color: #00d4ff; }

    /* === EXPANDER === */
    .streamlit-expanderHeader {
        font-size: 14px !important; font-weight: 600 !important; color: #9999b0 !important;
        border-radius: 10px !important;
    }
    .streamlit-expanderContent { border-color: #1e1e38 !important; }

    /* === SIDEBAR === */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0d1e 0%, #111128 100%);
        border-right: 1px solid #1e1e38;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label {
        font-size: 13px !important;
        font-weight: 600 !important;
        color: #9999b0 !important;
    }

    /* === EMPTY STATE === */
    .empty-state {
        text-align: center; padding: 48px 24px; color: #555578;
    }
    .empty-state .es-icon { font-size: 48px; margin-bottom: 12px; opacity: 0.5; }
    .empty-state .es-title { font-size: 18px; font-weight: 700; color: #8888a0; margin-bottom: 6px; }
    .empty-state .es-desc { font-size: 14px; color: #6b6b80; line-height: 1.6; }

    /* === DIVIDERS === */
    hr { border-color: #1e1e38 !important; opacity: 0.5; margin: 20px 0 !important; }

    /* === SCROLLBAR === */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0a0a1a; }
    ::-webkit-scrollbar-thumb { background: #2a2a40; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #3a3a55; }

    /* === NUMBER INPUT / SELECT POLISH === */
    .stNumberInput input, .stTextInput input {
        border-radius: 10px !important;
        transition: border-color 0.3s ease !important;
    }
    .stNumberInput input:focus, .stTextInput input:focus {
        border-color: #00d4ff !important;
        box-shadow: 0 0 0 2px rgba(0,212,255,0.15) !important;
    }

    /* === PROGRESS BAR === */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00d4ff, #a855f7) !important;
        border-radius: 4px;
    }

    /* === SCREENER ACTION BUTTONS === */
    .scr-actions { display: flex; gap: 6px; margin-top: 8px; }
    .scr-btn {
        display: inline-flex; align-items: center; justify-content: center; gap: 4px;
        padding: 5px 10px; border-radius: 8px; font-size: 12px; font-weight: 600;
        text-decoration: none; cursor: pointer; border: none; transition: all 0.2s;
        flex: 1; text-align: center;
    }
    .scr-btn-trade {
        background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.3);
    }
    .scr-btn-trade:hover { background: rgba(16,185,129,0.25); }
    .scr-btn-analyse {
        background: rgba(0,212,255,0.1); color: #00d4ff; border: 1px solid rgba(0,212,255,0.25);
    }
    .scr-btn-analyse:hover { background: rgba(0,212,255,0.2); color: #00d4ff; text-decoration: none; }
</style>""", unsafe_allow_html=True)

# ============================================================
# TITLE
# ============================================================
st.markdown("""<div class="app-title">
    <h1>Indian Stock Market Predictor</h1>
    <p>AI-Powered Predictions &bull; Swing &amp; Scalp Trading &bull; Multi-Factor Screener &bull; Fundamentals</p>
</div>""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
_qp_stock = st.query_params.get("stock")

with st.sidebar:
    st.markdown("""<div style="text-align: center; padding: 4px 0 12px 0;">
        <span style="font-size: 28px;">📈</span>
        <div style="font-size: 15px; font-weight: 700; color: #e0e0f0; margin-top: 2px;">Stock Predictor</div>
        <div style="font-size: 11px; color: #6b6b80;">AI-Powered Analysis</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("##### Select Stock")
    stock_options = {f"{name} ({ticker})": ticker for ticker, name in POPULAR_INDIAN_STOCKS.items()}
    _stock_keys = list(stock_options.keys())
    _default_stock_idx = 0
    if _qp_stock:
        _qp_upper = _qp_stock.strip().upper()
        for _i, _key in enumerate(_stock_keys):
            if _qp_upper in _key:
                _default_stock_idx = _i
                break
    selected_label = st.selectbox("Select Stock", options=_stock_keys,
                                  index=_default_stock_idx,
                                  help="Pick any popular Indian stock to analyze",
                                  label_visibility="collapsed")
    ticker = stock_options[selected_label]

    custom_ticker = st.text_input("Custom ticker", placeholder="e.g. ADANIENT.NS",
                                  help="Add .NS for NSE, .BO for BSE",
                                  label_visibility="collapsed")
    if custom_ticker.strip():
        ticker = custom_ticker.strip().upper()
    elif _qp_stock and _qp_stock.strip().upper() not in POPULAR_INDIAN_STOCKS:
        ticker = _qp_stock.strip().upper()

    st.divider()
    st.markdown("##### Prediction Settings")
    forecast_days = st.slider("Forecast days", min_value=1, max_value=30, value=7,
                              help="Shorter = more accurate. 7 days is a good balance.")
    epochs = st.slider("Training quality (epochs)", min_value=10, max_value=150, value=80,
                       help="Higher = better model but slower training. 80 is recommended.")

    with st.expander("Advanced Model Settings", expanded=False):
        st.caption("Leave defaults unless experienced")
        xgb_weight = st.slider("XGBoost Weight", 0.2, 0.8, 0.55, 0.05,
                               help="How much to trust XGBoost vs LSTM.")
        use_market_context = st.checkbox("Include market data (Nifty/VIX)", value=True,
                                         help="Uses overall market trends to improve predictions")

    with st.expander("Swing Trading Settings", expanded=False):
        capital = st.number_input("Capital (₹)", 10000, 10000000, 100000, step=10000,
                                  help="How much money you plan to invest")
        max_risk_pct = st.slider("Max risk per trade (%)", 0.5, 5.0, 2.0, 0.5,
                                 help="Maximum % of capital to risk. 2% is safe.")
        sr_lookback = st.slider("Analysis lookback (days)", 30, 180, 90,
                                help="How far back for support/resistance levels")
        pivot_method = st.selectbox("Pivot calculation", ["standard", "fibonacci", "camarilla", "woodie"])

    with st.expander("Paper Trading Funds", expanded=False):
        st.caption("Set capital for paper trading. Scalp trades get 5× leverage (buy ₹5L with ₹1L).")
        _cur_swing_fund = get_fund_balance("swing")

        _fund_input = st.number_input(
            "Trading Capital (₹)", 0, 50000000,
            int(_cur_swing_fund["initial_capital"]),
            step=50000, key="sidebar_trading_fund",
            help="Capital for paper trading. Swing uses full amount, scalp gets 5× leverage.",
        )
        if st.button("Update Funds", key="sidebar_update_funds", use_container_width=True):
            set_fund_capital("swing", _fund_input)
            set_fund_capital("scalp", _fund_input)
            st.success("Funds updated!")
            st.rerun()

    st.divider()
    st.markdown("""<div style="background: rgba(0,212,255,0.05); border: 1px solid rgba(0,212,255,0.15);
        border-radius: 10px; padding: 10px 14px; font-size: 12px; color: #8888a0; line-height: 1.5;">
        💡 <b style="color: #b0b0d0;">Tip:</b> Start with the <b style="color: #00d4ff;">Chart</b> tab to see
        the stock's current health, then check <b style="color: #00d4ff;">Swing</b> or
        <b style="color: #00d4ff;">Scalping</b> for trading signals.
    </div>""", unsafe_allow_html=True)

# ============================================================
# FETCH DATA
# ============================================================
try:
    with st.spinner("Fetching stock data..."):
        raw_df = fetch_stock_data(ticker)
        df = add_technical_indicators(raw_df)
        info = get_stock_info(ticker)
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# ============================================================
# STOCK BANNER — prominent header with stock name + metrics
# ============================================================
price_change = raw_df["Close"].iloc[-1] - raw_df["Close"].iloc[-2]
pct_change = (price_change / raw_df["Close"].iloc[-2]) * 100
change_color = "#34d399" if pct_change >= 0 else "#f87171"
change_arrow = "▲" if pct_change >= 0 else "▼"
change_bg = "rgba(16,185,129,0.1)" if pct_change >= 0 else "rgba(239,68,68,0.1)"
mcap_str = f"₹{info['market_cap'] / 1e7:,.0f} Cr" if info["market_cap"] else "N/A"

# 52-week range for context
w52_high = info.get("year_high", raw_df["High"].tail(252).max())
w52_low = info.get("year_low", raw_df["Low"].tail(252).min())
cur_price = info['current_price']
w52_pct = ((cur_price - w52_low) / (w52_high - w52_low) * 100) if w52_high > w52_low else 50

# 5-day trend mini indicator
recent_5d = raw_df["Close"].tail(5)
trend_5d_pct = ((recent_5d.iloc[-1] - recent_5d.iloc[0]) / recent_5d.iloc[0] * 100) if len(recent_5d) >= 5 else 0
trend_5d_color = "#34d399" if trend_5d_pct >= 0 else "#f87171"
trend_5d_arrow = "▲" if trend_5d_pct >= 0 else "▼"

# Volume context
avg_vol = raw_df["Volume"].tail(20).mean()
today_vol = raw_df["Volume"].iloc[-1]
vol_ratio_banner = today_vol / avg_vol if avg_vol > 0 else 1

st.markdown(f"""<div class="stock-banner">
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 20px;">
        <div>
            <div style="font-size: 30px; font-weight: 800; color: #f0f0ff; letter-spacing: 0.3px;">
                {info['name']}
            </div>
            <div style="display: flex; align-items: center; gap: 12px; margin-top: 6px;">
                <span style="font-size: 14px; color: #8888a0; font-weight: 600;">{ticker}</span>
                <span style="background: rgba(168,85,247,0.12); color: #c084fc; font-size: 11px; font-weight: 700;
                       padding: 3px 10px; border-radius: 6px; letter-spacing: 0.3px;">{info['sector']}</span>
            </div>
        </div>
        <div style="display: flex; gap: 28px; align-items: center; flex-wrap: wrap;">
            <div style="text-align: right;">
                <div style="font-size: 34px; font-weight: 800; color: #00d4ff; line-height: 1.1;">
                    ₹{cur_price:,.2f}
                </div>
                <div style="display: inline-flex; align-items: center; gap: 6px; margin-top: 6px;
                            background: {change_bg}; padding: 4px 12px; border-radius: 8px;">
                    <span style="font-size: 16px; font-weight: 700; color: {change_color};">
                        {change_arrow} ₹{abs(price_change):,.2f} ({pct_change:+.2f}%)
                    </span>
                </div>
            </div>
            <div style="text-align: center; border-left: 1px solid #2a2a45; padding-left: 24px;">
                <div style="font-size: 11px; color: #7777a0; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 700;">Market Cap</div>
                <div style="font-size: 20px; font-weight: 700; color: #d8d8e8; margin-top: 2px;">{mcap_str}</div>
            </div>
            <div style="text-align: center; border-left: 1px solid #2a2a45; padding-left: 24px;">
                <div style="font-size: 11px; color: #7777a0; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 700;">5-Day</div>
                <div style="font-size: 18px; font-weight: 700; color: {trend_5d_color}; margin-top: 2px;">{trend_5d_arrow} {trend_5d_pct:+.1f}%</div>
            </div>
        </div>
    </div>
    <div style="margin-top: 16px; display: flex; align-items: center; gap: 12px;">
        <span style="font-size: 11px; color: #6b6b80; font-weight: 600;">52W LOW ₹{w52_low:,.0f}</span>
        <div style="flex: 1; height: 6px; background: #1e1e38; border-radius: 3px; position: relative; overflow: hidden;">
            <div style="width: {w52_pct:.0f}%; height: 100%; border-radius: 3px;
                        background: linear-gradient(90deg, #ef4444, #f59e0b, #10b981);"></div>
            <div style="position: absolute; left: {w52_pct:.0f}%; top: -4px; width: 3px; height: 14px;
                        background: #00d4ff; border-radius: 2px; transform: translateX(-50%);
                        box-shadow: 0 0 8px rgba(0,212,255,0.5);"></div>
        </div>
        <span style="font-size: 11px; color: #6b6b80; font-weight: 600;">52W HIGH ₹{w52_high:,.0f}</span>
    </div>
</div>""", unsafe_allow_html=True)

# ============================================================
# QUICK STATS ROW
# ============================================================
latest_row = df.iloc[-1]
qs_rsi = latest_row.get("RSI", 50)
qs_rsi_color = "#f87171" if qs_rsi > 70 else "#34d399" if qs_rsi < 30 else "#e0e0f0"
qs_rsi_label = "Overbought" if qs_rsi > 70 else "Oversold" if qs_rsi < 30 else "Neutral"
qs_macd = latest_row.get("MACD", 0)
qs_macd_sig = latest_row.get("MACD_Signal", 0)
qs_macd_color = "#34d399" if qs_macd > qs_macd_sig else "#f87171"
qs_macd_label = "Bullish" if qs_macd > qs_macd_sig else "Bearish"
qs_adx = latest_row.get("ADX", 0)
qs_adx_label = "Strong" if qs_adx > 25 else "Weak"
qs_vol = latest_row.get("Volume_Ratio", 1)
qs_vol_color = "#34d399" if qs_vol > 1.2 else "#f87171" if qs_vol < 0.7 else "#e0e0f0"

st.markdown(f"""<div class="quick-stats">
    <div class="quick-stat">
        <div class="qs-label">RSI</div>
        <div class="qs-value" style="color: {qs_rsi_color};">{qs_rsi:.0f}</div>
        <div class="qs-delta" style="color: {qs_rsi_color};">{qs_rsi_label}</div>
    </div>
    <div class="quick-stat">
        <div class="qs-label">MACD</div>
        <div class="qs-value" style="color: {qs_macd_color};">{qs_macd:.2f}</div>
        <div class="qs-delta" style="color: {qs_macd_color};">{qs_macd_label}</div>
    </div>
    <div class="quick-stat">
        <div class="qs-label">ADX (Trend)</div>
        <div class="qs-value">{qs_adx:.0f}</div>
        <div class="qs-delta" style="color: {'#34d399' if qs_adx > 25 else '#7777a0'};">{qs_adx_label} Trend</div>
    </div>
    <div class="quick-stat">
        <div class="qs-label">Volume</div>
        <div class="qs-value" style="color: {qs_vol_color};">{qs_vol:.1f}x</div>
        <div class="qs-delta" style="color: {qs_vol_color};">vs 20d avg</div>
    </div>
    <div class="quick-stat">
        <div class="qs-label">Day Range</div>
        <div class="qs-value">₹{raw_df['Low'].iloc[-1]:,.0f}-{raw_df['High'].iloc[-1]:,.0f}</div>
        <div class="qs-delta" style="color: #7777a0;">Today</div>
    </div>
</div>""", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
_TAB_NAMES = [
    "🔍 Screener", "📊 Chart & Predictions",
    "📈 Swing", "⚡ Scalping", "📝 Paper Trading", "📋 Fundamentals", "🏭 Sector", "📰 Sentiment", "🎯 Accuracy",
]
_TAB_KEYS = ["screener", "chart", "swing", "scalp", "paper", "fundamentals", "sector", "sentiment", "accuracy"]

tab_screener, tab_chart, tab_swing, tab_scalp, tab_paper, tab_fund, tab_sector, tab_sentiment, tab_accuracy = st.tabs(_TAB_NAMES)

# Auto-switch tab from URL query param (?tab=swing)
_qp_tab = st.query_params.get("tab")
if _qp_tab and _qp_tab in _TAB_KEYS:
    _tab_idx = _TAB_KEYS.index(_qp_tab)
    import streamlit.components.v1 as _components
    _components.html(f"""
        <script>
            const tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
            if (tabs.length > {_tab_idx}) tabs[{_tab_idx}].click();
        </script>
    """, height=0)

# ============================================================
# PLOTLY DEFAULTS
# ============================================================
CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0e0e1c",
    plot_bgcolor="#0e0e1c",
    font=dict(color="#c0c0d0", family="Inter, sans-serif"),
    xaxis=dict(gridcolor="#1a1a32", zerolinecolor="#1a1a32", gridwidth=1),
    yaxis=dict(gridcolor="#1a1a32", zerolinecolor="#1a1a32", gridwidth=1),
    margin=dict(l=10, r=10, t=30, b=10),
    hoverlabel=dict(bgcolor="#1a1a2e", font_size=13, font_color="#e0e0f0", bordercolor="#2a2a45"),
)

# colors
C_CYAN = "#00d4ff"
C_AMBER = "#f59e0b"
C_PURPLE = "#a855f7"
C_GREEN = "#10b981"
C_RED = "#ef4444"
VOL_UP, VOL_DOWN = "#10b981", "#ef4444"


# ============================================================
# HELPERS
# ============================================================
def help_box(text: str):
    st.markdown(f"<div class='help-box'>💡 <b>What does this mean?</b> {text}</div>", unsafe_allow_html=True)

def verdict_box(text: str, sentiment: str = "neutral"):
    st.markdown(f"<div class='verdict-box {sentiment}'>{text}</div>", unsafe_allow_html=True)

def action_card(action: str, subtitle: str, css_class: str):
    st.markdown(f"<div class='action-card {css_class}'>{action}<span>{subtitle}</span></div>", unsafe_allow_html=True)

def checklist(items: list[tuple[str, bool]]):
    html = "<div class='checklist'><b>Checklist — Why this signal?</b><br>"
    for text, positive in items:
        html += f"{'✅' if positive else '❌'} {text}<br>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def calc_angel_one_charges(buy_price: float, sell_price: float, qty: int = 1) -> dict:
    """Calculate Angel One brokerage and net profit for a trade.

    Charges: lower of ₹20 or 0.1% per executed order, minimum ₹5.
    Additional charges: STT, exchange txn, SEBI, stamp duty, GST.
    """
    buy_value = buy_price * qty
    sell_value = sell_price * qty
    turnover = buy_value + sell_value

    # Brokerage: min(₹20, 0.1% of order value), minimum ₹5 per side
    buy_brokerage = max(5, min(20, buy_value * 0.001))
    sell_brokerage = max(5, min(20, sell_value * 0.001))
    total_brokerage = buy_brokerage + sell_brokerage

    # STT: 0.025% on sell side (intraday equity)
    stt = sell_value * 0.00025

    # Exchange transaction: 0.00345% on turnover (NSE)
    exchange_txn = turnover * 0.0000345

    # SEBI charges: ₹10 per crore
    sebi = turnover * 0.000001

    # Stamp duty: 0.003% on buy side
    stamp_duty = buy_value * 0.00003

    # GST: 18% on (brokerage + exchange txn + SEBI)
    gst = (total_brokerage + exchange_txn + sebi) * 0.18

    total_charges = total_brokerage + stt + exchange_txn + sebi + stamp_duty + gst
    gross_profit = (sell_price - buy_price) * qty
    net_profit = gross_profit - total_charges

    return {
        "gross_profit": round(gross_profit, 2),
        "total_charges": round(total_charges, 2),
        "net_profit": round(net_profit, 2),
        "brokerage": round(total_brokerage, 2),
        "stt": round(stt, 2),
        "gst": round(gst, 2),
        "other": round(exchange_txn + sebi + stamp_duty, 2),
    }

def interpret_rsi(rsi):
    if rsi > 70: return "Overbought — stock may be overpriced, could drop soon", "bad"
    if rsi < 30: return "Oversold — stock may be underpriced, could bounce up", "good"
    return "Neutral — no extreme buying or selling pressure", "neutral"

def interpret_macd(macd, signal):
    if macd > signal: return "Bullish — upward momentum is building", "good"
    return "Bearish — downward momentum is building", "bad"


# ========================================================================
# TAB 1: CHART
# ========================================================================
with tab_chart:
    help_box("Green candles = price went up. Red candles = price went down. "
             "Lines show average prices over different time periods. Use the checkboxes to customize the chart.")

    cc1, cc2, cc3, cc4 = st.columns([1, 1, 1, 2])
    show_sma = cc1.checkbox("Moving Averages", value=True)
    show_bollinger = cc2.checkbox("Bollinger Bands", value=False)
    show_volume = cc3.checkbox("Volume", value=True)

    rows = 2 if show_volume else 1
    row_heights = [0.75, 0.25] if show_volume else [1]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=row_heights)

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price",
    ), row=1, col=1)

    if show_sma:
        for col, color, label in [("SMA_20", C_CYAN, "20-Day"), ("SMA_50", C_AMBER, "50-Day"), ("SMA_200", C_PURPLE, "200-Day")]:
            if col in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], name=label, line=dict(width=1, color=color)), row=1, col=1)

    if show_bollinger and "BB_Upper" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper", line=dict(width=1, color="rgba(150,150,150,0.5)")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower", line=dict(width=1, color="rgba(150,150,150,0.5)"), fill="tonexty", fillcolor="rgba(150,150,150,0.07)"), row=1, col=1)

    if show_volume:
        vc = [VOL_UP if c >= o else VOL_DOWN for o, c in zip(df["Open"], df["Close"])]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=vc, opacity=0.5), row=2, col=1)

    fig.update_layout(height=550, xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02), **CHART_LAYOUT)
    if show_volume:
        fig.update_yaxes(gridcolor="#1e1e35", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # --- HEALTH CHECK ---
    st.markdown("#### Stock Health Check")
    help_box("Quick vital signs for this stock — translated into plain English.")

    latest = df.iloc[-1]
    rsi_val = latest["RSI"]
    rsi_text, rsi_sent = interpret_rsi(rsi_val)
    macd_val, macd_sig = latest["MACD"], latest["MACD_Signal"]
    macd_text, macd_sent = interpret_macd(macd_val, macd_sig)
    adx_val = latest.get("ADX", 0)
    vol_ratio = latest.get("Volume_Ratio", 1)

    bullish_count = sum([
        30 < rsi_val < 70, macd_val > macd_sig, adx_val > 25,
        latest["Close"] > latest.get("SMA_50", latest["Close"]),
    ])

    if bullish_count >= 3:
        verdict_box("📊 <b>Overall: Healthy.</b> Most indicators are positive — trend and momentum favor buyers.", "good")
    elif bullish_count <= 1:
        verdict_box("📊 <b>Overall: Weak.</b> Most indicators are negative — be cautious, sellers have control.", "bad")
    else:
        verdict_box("📊 <b>Overall: Mixed.</b> No clear direction — best to wait for a clearer setup.", "neutral")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("RSI", f"{rsi_val:.0f}/100")
        verdict_box(rsi_text, rsi_sent)
        st.metric("MACD", f"{macd_val:.2f}")
        verdict_box(macd_text, macd_sent)
    with c2:
        st.metric("ADX (Trend)", f"{adx_val:.0f}/100")
        verdict_box("Strong trend — stock moving decisively." if adx_val > 25 else "Weak trend — stock moving sideways.", "good" if adx_val > 25 else "neutral")
        st.metric("Volume", f"{vol_ratio:.1f}x avg")
        if vol_ratio > 1.5: verdict_box("High volume — price moves are reliable.", "good")
        elif vol_ratio < 0.7: verdict_box("Low volume — price moves may not stick.", "bad")
        else: verdict_box("Normal volume.", "neutral")

    with st.expander("Moving Average Trend"):
        sma_20 = latest.get("SMA_20", latest["Close"])
        sma_50 = latest.get("SMA_50", latest["Close"])
        sma_200 = latest.get("SMA_200", latest["Close"])
        price = latest["Close"]
        checklist([
            (f"Price ₹{price:,.0f} {'above' if price > sma_20 else 'below'} 20-day avg ₹{sma_20:,.0f}", price > sma_20),
            (f"Price {'above' if price > sma_50 else 'below'} 50-day avg ₹{sma_50:,.0f}", price > sma_50),
            (f"Price {'above' if price > sma_200 else 'below'} 200-day avg ₹{sma_200:,.0f}", price > sma_200),
            (f"Short-term {'uptrend' if sma_20 > sma_50 else 'downtrend'} (20d vs 50d)", sma_20 > sma_50),
        ])

    # ---- AI PREDICTIONS (merged into Chart tab) ----
    st.markdown("<div class='section-label'>AI Price Predictions</div>", unsafe_allow_html=True)
    help_box(
        "Our AI uses <b>LSTM</b> (neural network) + <b>XGBoost</b> (decision trees), "
        "blended together for best accuracy. Click <b>Train & Predict</b> to start."
    )

    c_train, c_load = st.columns(2)
    with c_train:
        train_btn = st.button("🚀 Train & Predict", type="primary", use_container_width=True,
                              help="Trains on 5 years of data. Takes 2-5 minutes.")
    with c_load:
        saved = load_saved_model(ticker)
        load_btn = st.button("📂 Use Saved Model" if saved else "No Saved Model",
                             disabled=saved is None, use_container_width=True)

    if "prediction_results" not in st.session_state and not train_btn and not load_btn:
        st.markdown("""<div class="empty-state">
            <div class="es-icon">🤖</div>
            <div class="es-title">No predictions yet</div>
            <div class="es-desc">Click <b>Train & Predict</b> to train AI models on 5 years of data,<br>
            or <b>Use Saved Model</b> if you've trained before.</div>
        </div>""", unsafe_allow_html=True)

    if train_btn:
        progress_bar = st.progress(0, text="Training AI models...")
        status_text = st.empty()
        def update_progress(step, total, model_name=""):
            pct = step / total if total > 0 else 0
            if model_name == "XGBoost":
                pct = 0.8 + 0.2 * pct
                status_text.text("Training XGBoost...")
            else:
                pct = 0.8 * pct
                status_text.text(f"Training LSTM — epoch {step}/{total}")
            progress_bar.progress(min(pct, 1.0))
        with st.spinner("Training..."):
            results = train_and_predict(ticker, forecast_days=forecast_days, epochs=epochs,
                                        progress_callback=update_progress, use_market_context=use_market_context,
                                        xgb_base_weight=xgb_weight)
        progress_bar.progress(1.0, text="Training complete!")
        status_text.empty()
        st.session_state["prediction_results"] = results
        try:
            log_predictions(
                ticker=ticker, prediction_date=date.today(),
                target_dates=results["prediction_dates"],
                ensemble_prices=results["predictions"],
                lstm_prices=results["lstm_predictions"],
                xgb_prices=results["xgb_predictions"],
                confidence=results.get("confidence", []),
                last_known_price=results["last_price"],
            )
        except Exception:
            pass

    if load_btn:
        with st.spinner("Generating predictions..."):
            results = predict_with_saved_model(ticker, forecast_days=forecast_days,
                                               use_market_context=use_market_context, xgb_base_weight=xgb_weight)
        if results:
            st.session_state["prediction_results"] = results
            try:
                log_predictions(
                    ticker=ticker, prediction_date=date.today(),
                    target_dates=results["prediction_dates"],
                    ensemble_prices=results["predictions"],
                    lstm_prices=results["lstm_predictions"],
                    xgb_prices=results["xgb_predictions"],
                    confidence=results.get("confidence", []),
                    last_known_price=results["last_price"],
                )
            except Exception:
                pass

    if "prediction_results" in st.session_state:
        results = st.session_state["prediction_results"]
        pred_change = results["predictions"][-1] - results["last_price"]
        pred_pct = (pred_change / results["last_price"]) * 100

        if pred_pct > 2:
            action_card(f"AI Predicts: UP {pred_pct:+.1f}%",
                        f"₹{results['last_price']:,.2f} → ₹{results['predictions'][-1]:,.2f} in {forecast_days} days", "action-up")
        elif pred_pct < -2:
            action_card(f"AI Predicts: DOWN {pred_pct:+.1f}%",
                        f"₹{results['last_price']:,.2f} → ₹{results['predictions'][-1]:,.2f} in {forecast_days} days", "action-down")
        else:
            action_card(f"AI Predicts: SIDEWAYS ({pred_pct:+.1f}%)",
                        f"₹{results['last_price']:,.2f} → ₹{results['predictions'][-1]:,.2f} in {forecast_days} days", "action-sideways")

        dir_word = "increase" if pred_change > 0 else "decrease"
        help_box(f"The AI predicts the stock will <b>{dir_word}</b> by <b>₹{abs(pred_change):,.2f}</b> "
                 f"(<b>{abs(pred_pct):.1f}%</b>) over <b>{forecast_days} days</b>. "
                 f"The shaded area shows the confidence range.")

        # Chart
        fig_pred = go.Figure()
        hist_df = results["historical_df"].tail(60)
        fig_pred.add_trace(go.Scatter(x=hist_df.index, y=hist_df["Close"], name="Actual Price", line=dict(color=C_CYAN)))
        fig_pred.add_trace(go.Scatter(x=results["prediction_dates"], y=results["predictions"],
                                       name="AI Prediction", line=dict(color=C_AMBER, dash="dash", width=2.5)))
        if "lstm_predictions" in results:
            fig_pred.add_trace(go.Scatter(x=results["prediction_dates"], y=results["lstm_predictions"],
                                           name="LSTM", line=dict(color=C_PURPLE, dash="dot", width=1), visible="legendonly"))
        if "xgb_predictions" in results:
            fig_pred.add_trace(go.Scatter(x=results["prediction_dates"], y=results["xgb_predictions"],
                                           name="XGBoost", line=dict(color=C_GREEN, dash="dot", width=1), visible="legendonly"))
        if "confidence" in results:
            upper = [c["upper"] for c in results["confidence"]]
            lower = [c["lower"] for c in results["confidence"]]
            dates = results["prediction_dates"]
            fig_pred.add_trace(go.Scatter(x=list(dates) + list(dates[::-1]), y=upper + lower[::-1],
                                           fill="toself", fillcolor="rgba(245,158,11,0.1)", line=dict(width=0), name="Confidence"))
        fig_pred.update_layout(height=380, **CHART_LAYOUT)
        st.plotly_chart(fig_pred, use_container_width=True)

        # Table
        st.markdown("#### Day-by-Day Forecast")
        pred_table = {"Date": results["prediction_dates"],
                      "Predicted (₹)": [f"{p:,.2f}" for p in results["predictions"]],
                      "Change": [f"{'🟢' if p > results['last_price'] else '🔴'} {((p - results['last_price']) / results['last_price'] * 100):+.2f}%"
                                 for p in results["predictions"]]}
        st.dataframe(pd.DataFrame(pred_table), use_container_width=True, hide_index=True)

        # Accuracy
        if "backtest_metrics" in results:
            bm = results["backtest_metrics"]
            ed = bm["ensemble"]["directional_accuracy"]
            er = bm["ensemble"]["rmse"]
            if ed >= 55:
                verdict_box(f"🎯 <b>Directional Accuracy: {ed:.1f}%</b> — correctly predicts up/down ~{ed:.0f}/100 times. Avg error: ₹{er:.2f}", "good")
            else:
                verdict_box(f"⚠️ <b>Directional Accuracy: {ed:.1f}%</b> — modest. Use as one input, not sole decision maker. Avg error: ₹{er:.2f}", "neutral")
            if bm.get("improvement_pct", 0) > 0:
                st.success(f"Ensemble is {bm['improvement_pct']:.1f}% better than LSTM alone")

            # Show auto-tuned weight
            if "auto_tune" in results:
                at = results["auto_tune"]
                used_w = results.get("used_xgb_weight", xgb_weight)
                help_box(
                    f"Auto-tuned optimal XGBoost weight: <b>{at['optimal_weight']:.2f}</b> "
                    f"(RMSE: ₹{at['best_rmse']:.2f}). Using: <b>{used_w:.2f}</b>"
                )

            with st.expander("Model Comparison"):
                mc1, mc2, mc3 = st.columns(3)
                for col, name, key in [(mc1, "LSTM", "lstm"), (mc2, "XGBoost", "xgb"), (mc3, "Ensemble", "ensemble")]:
                    with col:
                        st.markdown(f"**{name}**")
                        st.write(f"Error: ₹{bm[key]['rmse']:.2f}")
                        st.write(f"Direction: {bm[key]['directional_accuracy']:.1f}%")

        if "actual_vs_predicted" in results:
            with st.expander("Backtest Chart"):
                eval_df = results["actual_vs_predicted"].tail(120)
                fig_e = go.Figure()
                fig_e.add_trace(go.Scatter(x=eval_df.index, y=eval_df["Actual"], name="Actual", line=dict(color=C_CYAN)))
                fig_e.add_trace(go.Scatter(x=eval_df.index, y=eval_df["Predicted"], name="Predicted", line=dict(color=C_AMBER)))
                fig_e.update_layout(height=320, **CHART_LAYOUT)
                st.plotly_chart(fig_e, use_container_width=True)

        if "feature_importance" in results and results["feature_importance"]:
            with st.expander("Feature Importance"):
                fi = results["feature_importance"]
                top = dict(list(fi.items())[:10])
                fig_fi = go.Figure(go.Bar(x=list(top.values()), y=list(top.keys()), orientation="h", marker_color=C_CYAN))
                fig_fi.update_layout(height=320, **CHART_LAYOUT)
                fig_fi.update_yaxes(autorange="reversed")
                st.plotly_chart(fig_fi, use_container_width=True)

        if "train_metrics" in results:
            with st.expander("Training Details"):
                m1, m2 = st.columns(2)
                with m1:
                    f1 = go.Figure()
                    f1.add_trace(go.Scatter(y=results["train_metrics"]["loss"], name="Train"))
                    f1.add_trace(go.Scatter(y=results["train_metrics"]["val_loss"], name="Val"))
                    f1.update_layout(height=280, title="Loss", **CHART_LAYOUT)
                    st.plotly_chart(f1, use_container_width=True)
                with m2:
                    f2 = go.Figure()
                    f2.add_trace(go.Scatter(y=results["train_metrics"]["mae"], name="Train"))
                    f2.add_trace(go.Scatter(y=results["train_metrics"]["val_mae"], name="Val"))
                    f2.update_layout(height=280, title="MAE", **CHART_LAYOUT)
                    st.plotly_chart(f2, use_container_width=True)


# ========================================================================
# TAB 3: SWING
# ========================================================================
with tab_swing:
    help_box("<b>Swing trading</b> = hold a stock for days to weeks. Buy at support, sell at resistance. "
             "This tab tells you: <b>buy, sell, or wait?</b> With exact entry, stop loss, and targets.")

    signal = generate_swing_signals(df)
    atr_data = calculate_atr_stop_loss(df)
    setup = calculate_trade_setup(df, signal, capital=capital, max_risk_pct=max_risk_pct / 100)
    sr_data = calculate_support_resistance(df, lookback=sr_lookback)
    fib_data = calculate_fibonacci_retracements(df, lookback=sr_lookback)
    pivot_data = calculate_pivot_points(df, method=pivot_method)
    patterns = identify_swing_patterns(df)

    sig_class = {"BUY": "action-buy", "SELL": "action-sell", "HOLD": "action-hold"}
    sig_advice = {"BUY": "Indicators suggest buying.", "SELL": "Indicators suggest selling.", "HOLD": "Mixed signals — wait."}
    action_card(f"Signal: {signal.signal} — {signal.strength}",
                f"Confidence: {signal.confidence * 100:.0f}% — {sig_advice.get(signal.signal, '')}", sig_class.get(signal.signal, "action-hold"))

    if signal.reasons:
        checklist([(r, any(w in r.lower() for w in ["bullish", "above", "bounce", "oversold", "upward", "positive"])) for r in signal.reasons])

    st.markdown("#### Trade Plan")
    help_box("• <b>Entry</b> = price to buy/sell at • <b>Stop Loss</b> = exit here to limit loss "
             "• <b>Targets</b> = take profit here • <b>R:R</b> = risk vs reward (1:2 = risk ₹1 to gain ₹2)")

    swing_qty = st.number_input("Quantity (shares)", min_value=1, value=10, step=5, key="swing_qty",
                                help="Number of shares for profit calculation")

    # Calculate profit after Angel One charges for each target
    if signal.signal == "SELL":
        sw_t1 = calc_angel_one_charges(setup.target_1, setup.entry_price, swing_qty)
        sw_t2 = calc_angel_one_charges(setup.target_2, setup.entry_price, swing_qty)
        sw_t3 = calc_angel_one_charges(setup.target_3, setup.entry_price, swing_qty)
        sw_sl = calc_angel_one_charges(setup.stop_loss, setup.entry_price, swing_qty)
    else:  # BUY or HOLD — treat as buy for display
        sw_t1 = calc_angel_one_charges(setup.entry_price, setup.target_1, swing_qty)
        sw_t2 = calc_angel_one_charges(setup.entry_price, setup.target_2, swing_qty)
        sw_t3 = calc_angel_one_charges(setup.entry_price, setup.target_3, swing_qty)
        sw_sl = calc_angel_one_charges(setup.entry_price, setup.stop_loss, swing_qty)

    p1, p2 = st.columns(2)
    with p1:
        sl_pct = abs(setup.entry_price - setup.stop_loss) / setup.entry_price * 100
        st.markdown(f"""<div class='setup-card' style='border-left: 3px solid #00d4ff;'>
            <b>Entry:</b> ₹{setup.entry_price:,.2f}<br>
            <b>Stop Loss:</b> ₹{setup.stop_loss:,.2f} <span class='loss'>(−{sl_pct:.1f}%)</span><br>
            <b>Position:</b> {setup.position_size_pct:.1f}% of ₹{capital:,}<br>
            <b>Volatility:</b> {atr_data['volatility_regime']} (ATR: ₹{atr_data['current_atr']:.2f})<br>
            <span class='loss'><b>If stopped:</b> -₹{abs(sw_sl['net_profit']):,.2f} ({swing_qty} shares)</span>
        </div>""", unsafe_allow_html=True)
    with p2:
        t1_cls = "profit" if sw_t1["net_profit"] > 0 else "loss"
        t2_cls = "profit" if sw_t2["net_profit"] > 0 else "loss"
        t3_cls = "profit" if sw_t3["net_profit"] > 0 else "loss"
        st.markdown(f"""<div class='setup-card' style='border-left: 3px solid #10b981;'>
            <b>Target 1:</b> ₹{setup.target_1:,.2f} (1:{setup.risk_reward_1})
            — <span class='{t1_cls}'><b>Net ₹{sw_t1['net_profit']:,.2f}</b></span>
            <span class='charges-dim'>(charges ₹{sw_t1['total_charges']:.2f})</span><br>
            <b>Target 2:</b> ₹{setup.target_2:,.2f} (1:{setup.risk_reward_2})
            — <span class='{t2_cls}'><b>Net ₹{sw_t2['net_profit']:,.2f}</b></span>
            <span class='charges-dim'>(charges ₹{sw_t2['total_charges']:.2f})</span><br>
            <b>Target 3:</b> ₹{setup.target_3:,.2f} (1:{setup.risk_reward_3})
            — <span class='{t3_cls}'><b>Net ₹{sw_t3['net_profit']:,.2f}</b></span>
            <span class='charges-dim'>(charges ₹{sw_t3['total_charges']:.2f})</span>
        </div>""", unsafe_allow_html=True)

    risk_amount = setup.entry_price - setup.stop_loss if signal.signal == "BUY" else setup.stop_loss - setup.entry_price
    max_loss = abs(risk_amount) * (capital * setup.position_size_pct / 100) / setup.entry_price
    verdict_box(f"⚠️ <b>Max loss if stopped out:</b> ~₹{max_loss:,.0f} ({max_risk_pct:.1f}% of capital). Always use a stop loss!", "neutral")

    # Paper trade button for swing
    if signal.signal in ("BUY", "SELL"):
        _sw_capital_needed = setup.entry_price * swing_qty
        _sw_fund = get_fund_balance("swing")
        _sw_fund_active = _sw_fund["initial_capital"] > 0
        _sw_btn_label = f"📝 Paper Trade — {signal.signal} {ticker}"
        if _sw_fund_active:
            _sw_btn_label += f" (₹{_sw_capital_needed:,.0f})"
        if _sw_fund_active and _sw_fund["available"] < _sw_capital_needed:
            st.warning(f"Insufficient swing funds. Need ₹{_sw_capital_needed:,.0f}, available ₹{_sw_fund['available']:,.0f}")
        if st.button(_sw_btn_label, key="swing_paper_btn", type="secondary"):
            try:
                tid = place_trade(
                    ticker=ticker,
                    trade_type="swing",
                    direction=signal.signal,
                    entry_price=setup.entry_price,
                    stop_loss=setup.stop_loss,
                    target_1=setup.target_1,
                    target_2=setup.target_2,
                    target_3=setup.target_3,
                    quantity=swing_qty,
                    signal_strength=signal.strength,
                    confidence=signal.confidence,
                    reasons=", ".join(signal.reasons) if signal.reasons else "",
                )
                st.success(f"Swing paper trade #{tid} placed! Entry ₹{setup.entry_price:,.2f} × {swing_qty} shares")
            except ValueError as e:
                st.error(str(e))

    # S/R Chart
    st.markdown("#### Support & Resistance")
    help_box("<b>Support</b> (green) = floor where price bounces up. <b>Resistance</b> (red) = ceiling where price turns down.")

    fig_sr = go.Figure()
    chart_data = df.tail(sr_lookback)
    fig_sr.add_trace(go.Candlestick(x=chart_data.index, open=chart_data["Open"], high=chart_data["High"],
                                     low=chart_data["Low"], close=chart_data["Close"], name="Price"))
    for lv in sr_data["support_levels"]:
        fig_sr.add_hline(y=lv["price"], line_dash="dash", line_color="rgba(16,185,129,0.6)",
                         annotation_text=f"S ₹{lv['price']:.0f} ({lv['touches']}x)")
    for lv in sr_data["resistance_levels"]:
        fig_sr.add_hline(y=lv["price"], line_dash="dash", line_color="rgba(239,68,68,0.6)",
                         annotation_text=f"R ₹{lv['price']:.0f} ({lv['touches']}x)")
    fig_sr.add_hline(y=pivot_data["PP"], line_dash="dot", line_color=C_AMBER, annotation_text=f"Pivot ₹{pivot_data['PP']:.0f}")
    fig_sr.update_layout(height=450, xaxis_rangeslider_visible=False, **CHART_LAYOUT)
    st.plotly_chart(fig_sr, use_container_width=True)

    with st.expander("Fibonacci Levels"):
        fig_fib = go.Figure()
        fig_fib.add_trace(go.Candlestick(x=chart_data.index, open=chart_data["Open"], high=chart_data["High"],
                                          low=chart_data["Low"], close=chart_data["Close"], name="Price"))
        fib_colors = [C_RED, C_AMBER, "#eab308", C_GREEN, C_CYAN]
        for i, (label, level) in enumerate(fib_data["retracements"].items()):
            fig_fib.add_hline(y=level, line_dash="dash", line_color=fib_colors[i % len(fib_colors)],
                              annotation_text=f"Fib {label}: ₹{level:.0f}")
        fig_fib.update_layout(height=400, xaxis_rangeslider_visible=False, **CHART_LAYOUT)
        st.plotly_chart(fig_fib, use_container_width=True)

    with st.expander("Pivot Points"):
        pivot_cols = st.columns(7)
        for i, key in enumerate(["S3", "S2", "S1", "PP", "R1", "R2", "R3"]):
            pivot_cols[i].metric(key, f"₹{pivot_data[key]:,.2f}")

    if patterns:
        st.markdown("#### Detected Patterns")
        for p in patterns:
            s = "good" if "bullish" in p["implication"].lower() else "bad" if "bearish" in p["implication"].lower() else "neutral"
            verdict_box(f"<b>{p['pattern']}</b> ({p['confidence']*100:.0f}%) — {p['implication']}", s)
    else:
        st.markdown("""<div class="empty-state" style="padding: 24px;">
            <div class="es-icon" style="font-size: 32px;">🔎</div>
            <div class="es-title" style="font-size: 15px;">No chart patterns detected</div>
            <div class="es-desc" style="font-size: 13px;">No significant bullish or bearish patterns found in recent data.</div>
        </div>""", unsafe_allow_html=True)


# ========================================================================
# TAB 4: SCALPING
# ========================================================================
with tab_scalp:
    help_box("<b>Scalping</b> = buy & sell within minutes on 5-min candles. Always use stop losses!")

    # Check if screener has scalp data for this ticker
    screener_scalp_match = None
    if "scalp_results" in st.session_state:
        for sr in st.session_state["scalp_results"]:
            if sr["ticker"] == ticker:
                screener_scalp_match = sr
                break

    sc1, sc2 = st.columns([2, 1])
    with sc1:
        scalp_period = st.selectbox("Data period", ["1d", "2d", "5d"], index=2,
                                     help="Use 5d for consistency with screener results")
    with sc2:
        scalp_btn = st.button("Load Intraday Data", type="primary", use_container_width=True)

    # Clear stale scalp data if ticker changed
    if "scalp_data_ticker" in st.session_state and st.session_state["scalp_data_ticker"] != ticker:
        st.session_state.pop("scalp_data", None)
        st.session_state.pop("scalp_data_ticker", None)

    if scalp_btn:
        with st.spinner("Fetching 5-min candles..."):
            try:
                idf = fetch_intraday_data(ticker, interval="5m", period=scalp_period)
                idf = add_scalping_indicators(idf)
                st.session_state["scalp_data"] = idf
                st.session_state["scalp_data_ticker"] = ticker
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Intraday data only available during market hours (Mon-Fri, 9:15 AM - 3:30 PM IST)")

    if "scalp_data" not in st.session_state and not scalp_btn:
        st.markdown("""<div class="empty-state">
            <div class="es-icon">⚡</div>
            <div class="es-title">Load intraday data to begin</div>
            <div class="es-desc">Click <b>Load Intraday Data</b> above to fetch 5-minute candles<br>
            and generate scalping signals with entry/exit levels.</div>
        </div>""", unsafe_allow_html=True)

    if "scalp_data" in st.session_state:
        idf = st.session_state["scalp_data"]
        ss = generate_scalp_signal(idf)
        levels = get_scalping_levels(idf)
        micro = get_market_microstructure(idf)

        # --- Signal + scalpability in one row ---
        scalpability = micro["scalpability_score"]
        scalp_class = {"LONG": "action-long", "SHORT": "action-short", "NO_TRADE": "action-notrade"}
        scalp_label = {"LONG": "BUY (Long)", "SHORT": "SELL (Short)", "NO_TRADE": "NO TRADE — Wait"}
        scalp_advice = {"LONG": "Price likely going UP.", "SHORT": "Price likely going DOWN.", "NO_TRADE": "Signals weak — don't trade."}
        scalp_score_color = "#10b981" if scalpability >= 65 else "#f59e0b" if scalpability >= 40 else "#ef4444"
        action_card(
            scalp_label.get(ss.signal, "NO TRADE"),
            f"{ss.strength} | {ss.confidence*100:.0f}% — {scalp_advice.get(ss.signal, '')} "
            f"| Scalpability: <span style='color:{scalp_score_color}'>{scalpability}/100</span>",
            scalp_class.get(ss.signal, "action-notrade"),
        )

        # --- MARKET TIME WARNING ---
        ist = pytz.timezone("Asia/Kolkata")
        now_ist = datetime.now(ist)
        market_open = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
        is_weekday = now_ist.weekday() < 5

        if is_weekday and market_open <= now_ist <= market_close:
            mins_left = int((market_close - now_ist).total_seconds() / 60)
            if mins_left <= 30:
                verdict_box(
                    f"<b>Market closes in {mins_left} min!</b> Avoid new scalp trades — "
                    f"not enough time for targets to be reached. Exit open positions.", "bad")
            elif mins_left <= 60:
                verdict_box(
                    f"<b>{mins_left} min to market close.</b> Use tighter targets only. "
                    f"Target 2 unlikely — focus on Target 1 or skip.", "neutral")
            elif mins_left <= 90:
                verdict_box(f"<b>{mins_left} min to close.</b> Be selective — only high-confidence setups.", "neutral")
        elif is_weekday and now_ist > market_close:
            verdict_box("<b>Market closed.</b> Data from last trading session.", "neutral")
        elif not is_weekday:
            verdict_box("<b>Weekend — market closed.</b> Data from last trading session.", "neutral")

        # Screener consistency (compact)
        if screener_scalp_match and screener_scalp_match["signal"] != ss.signal:
            verdict_box(
                f"Screener showed <b>{screener_scalp_match['signal']}</b>, now <b>{ss.signal}</b> — signal changed. Trust latest.",
                "neutral")

        # --- TRADE SETUP ---
        risk_ps = abs(ss.entry_price - ss.stop_loss)
        rew_ps = abs(ss.target_1 - ss.entry_price)

        scalp_qty = st.number_input("Quantity", min_value=1, value=100, step=25, key="scalp_qty")
        # Always calculate charges — even NO_TRADE has entry/target/SL values to show
        if ss.signal == "SHORT":
            buy_p, sell_p = ss.target_1, ss.entry_price
            buy_p2, sell_p2 = ss.target_2, ss.entry_price
            sl_buy, sl_sell = ss.stop_loss, ss.entry_price
        else:  # LONG or NO_TRADE — treat as long for display
            buy_p, sell_p = ss.entry_price, ss.target_1
            buy_p2, sell_p2 = ss.entry_price, ss.target_2
            sl_buy, sl_sell = ss.entry_price, ss.stop_loss
        t1_charges = calc_angel_one_charges(buy_p, sell_p, scalp_qty)
        t2_charges = calc_angel_one_charges(buy_p2, sell_p2, scalp_qty)
        loss_charges = calc_angel_one_charges(sl_buy, sl_sell, scalp_qty)

        t1c, t2c = st.columns(2)
        with t1c:
            st.markdown(f"""<div class='setup-card' style='border-left: 3px solid #00d4ff;'>
                <b>Entry:</b> ₹{ss.entry_price:,.2f}<br>
                <b>Stop Loss:</b> ₹{ss.stop_loss:,.2f} <span class='loss'>(₹{risk_ps:.2f} risk)</span><br>
                <b>R:R:</b> 1:{ss.risk_reward}<br>
                <span class='loss'><b>If stopped:</b> -₹{abs(loss_charges['net_profit']):,.2f} ({scalp_qty} qty)</span>
            </div>""", unsafe_allow_html=True)
        with t2c:
            p1_class = "profit" if t1_charges["net_profit"] > 0 else "loss"
            p2_class = "profit" if t2_charges["net_profit"] > 0 else "loss"
            st.markdown(f"""<div class='setup-card' style='border-left: 3px solid #10b981;'>
                <b>Target 1:</b> ₹{ss.target_1:,.2f} (+₹{rew_ps:.2f}/share)
                — <span class='{p1_class}'><b>Net ₹{t1_charges['net_profit']:,.2f}</b></span>
                <span class='charges-dim'>(charges ₹{t1_charges['total_charges']:.2f})</span><br>
                <b>Target 2:</b> ₹{ss.target_2:,.2f} (+₹{abs(ss.target_2 - ss.entry_price):.2f}/share)
                — <span class='{p2_class}'><b>Net ₹{t2_charges['net_profit']:,.2f}</b></span>
                <span class='charges-dim'>(charges ₹{t2_charges['total_charges']:.2f})</span>
            </div>""", unsafe_allow_html=True)

        # Paper trade button for scalping
        if ss.signal in ("LONG", "SHORT"):
            _sc_position_value = ss.entry_price * scalp_qty
            _sc_margin_needed = _sc_position_value / 5  # 5x leverage
            _sc_fund = get_fund_balance("scalp")
            _sc_fund_active = _sc_fund["initial_capital"] > 0
            _sc_btn_label = f"📝 Paper Trade — {ss.signal} {ticker}"
            if _sc_fund_active:
                _sc_btn_label += f" (₹{_sc_margin_needed:,.0f} margin)"
            if _sc_fund_active and _sc_fund["available"] < _sc_margin_needed:
                st.warning(f"Insufficient funds. Need ₹{_sc_margin_needed:,.0f} margin (5× on ₹{_sc_position_value:,.0f}), available ₹{_sc_fund['available']:,.0f}")
            if st.button(_sc_btn_label, key="scalp_paper_btn", type="secondary"):
                try:
                    tid = place_trade(
                        ticker=ticker,
                        trade_type="scalp",
                        direction=ss.signal,
                        entry_price=ss.entry_price,
                        stop_loss=ss.stop_loss,
                        target_1=ss.target_1,
                        target_2=ss.target_2,
                        quantity=scalp_qty,
                        signal_strength=ss.strength,
                        confidence=ss.confidence,
                        reasons=", ".join(ss.reasons) if ss.reasons else "",
                    )
                    st.success(f"Scalp paper trade #{tid} placed! Entry ₹{ss.entry_price:,.2f} × {scalp_qty} qty")
                except ValueError as e:
                    st.error(str(e))

        # --- Key levels row ---
        lv1, lv2, lv3, lv4 = st.columns(4)
        lv1.metric("VWAP", f"₹{levels['vwap']:,.2f}", "Above ↑" if idf["Close"].iloc[-1] > levels["vwap"] else "Below ↓")
        lv2.metric("Pivot", f"₹{levels['pivot']:,.2f}")
        lv3.metric("Today High", f"₹{levels['today_high']:,.2f}")
        lv4.metric("Today Low", f"₹{levels['today_low']:,.2f}")

        # --- Chart ---
        fig_s = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
        fig_s.add_trace(go.Candlestick(x=idf.index, open=idf["Open"], high=idf["High"], low=idf["Low"], close=idf["Close"], name="Price"), row=1, col=1)
        fig_s.add_trace(go.Scatter(x=idf.index, y=idf["EMA_5"], name="EMA 5", line=dict(width=1, color=C_CYAN)), row=1, col=1)
        fig_s.add_trace(go.Scatter(x=idf.index, y=idf["EMA_9"], name="EMA 9", line=dict(width=1, color=C_AMBER)), row=1, col=1)
        fig_s.add_trace(go.Scatter(x=idf.index, y=idf["EMA_21"], name="EMA 21", line=dict(width=1, color=C_PURPLE)), row=1, col=1)
        fig_s.add_trace(go.Scatter(x=idf.index, y=idf["VWAP"], name="VWAP", line=dict(width=2, color="#eab308", dash="dash")), row=1, col=1)
        fig_s.add_trace(go.Scatter(x=idf.index, y=idf["BB_Upper_Scalp"], name="BB+", line=dict(width=1, color="rgba(150,150,150,0.4)")), row=1, col=1)
        fig_s.add_trace(go.Scatter(x=idf.index, y=idf["BB_Lower_Scalp"], name="BB-", line=dict(width=1, color="rgba(150,150,150,0.4)"), fill="tonexty", fillcolor="rgba(150,150,150,0.06)"), row=1, col=1)
        fig_s.add_hline(y=levels["pivot"], line_dash="dot", line_color=C_CYAN, annotation_text="Pivot", row=1, col=1)
        fig_s.add_hline(y=levels["prev_high"], line_dash="dot", line_color=C_RED, annotation_text="Prev High", row=1, col=1)
        fig_s.add_hline(y=levels["prev_low"], line_dash="dot", line_color=C_GREEN, annotation_text="Prev Low", row=1, col=1)
        svc = [VOL_UP if c >= o else VOL_DOWN for o, c in zip(idf["Open"], idf["Close"])]
        fig_s.add_trace(go.Bar(x=idf.index, y=idf["Volume"], name="Vol", marker_color=svc, opacity=0.5), row=2, col=1)
        fig_s.update_layout(height=500, xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02), **CHART_LAYOUT)
        fig_s.update_yaxes(gridcolor="#1e1e35", row=2, col=1)
        st.plotly_chart(fig_s, use_container_width=True)

        # --- Collapsible details ---
        with st.expander("Signal Reasons"):
            if ss.reasons:
                checklist([(r, any(w in r.lower() for w in ["bullish", "above", "bounce", "oversold", "positive", "up candle", "upward"])) for r in ss.reasons])

        with st.expander("Camarilla & Market Details"):
            cam = st.columns(6)
            for i, (l, k) in enumerate(zip(["S3", "S2", "S1", "R1", "R2", "R3"],
                                            ["cam_s3", "cam_s2", "cam_s1", "cam_r1", "cam_r2", "cam_r3"])):
                cam[i].metric(l, f"₹{levels[k]:,.2f}")
            mi1, mi2, mi3, mi4 = st.columns(4)
            mi1.metric("ATR", f"₹{micro['atr']:.2f}")
            mi2.metric("Candle Size", f"{micro['avg_candle_range']:.3f}%")
            mi3.metric("Consecutive", f"{micro['consecutive_candles']} {micro['consecutive_direction']}")
            mi4.metric("Trend Slope", f"{micro['trend_slope']:.4f}")


# ========================================================================
# TAB: PAPER TRADING DASHBOARD
# ========================================================================
with tab_paper:
    # Aggregate stats for the header (all trades)
    _pt_all_stats = get_paper_stats(None)
    _pt_swing_stats = get_paper_stats("swing")
    _pt_scalp_stats = get_paper_stats("scalp")

    # -- Performance summary banner --
    _pnl = _pt_all_stats["total_pnl"]
    _pnl_color = "#34d399" if _pnl >= 0 else "#f87171"
    _pnl_sign = "+" if _pnl >= 0 else ""
    _total_profit = _pt_all_stats["total_profit"]
    _total_loss = _pt_all_stats["total_loss"]
    _wr = _pt_all_stats["win_rate"]
    _wr_color = "#34d399" if _wr >= 50 else "#f59e0b" if _wr >= 30 else "#f87171"

    # Compute combined funds available
    _fund_swing = get_fund_balance("swing")
    _fund_scalp = get_fund_balance("scalp")
    _any_funds = _fund_swing["initial_capital"] > 0 or _fund_scalp["initial_capital"] > 0
    _total_available = _fund_swing["available"] + _fund_scalp["available"]
    _total_deployed = _fund_swing["deployed"] + _fund_scalp["deployed"]
    _total_initial = _fund_swing["initial_capital"] + _fund_scalp["initial_capital"]

    _funds_html = ""
    if _any_funds:
        _funds_html = (
            f'<div style="margin-top: 16px; padding-top: 14px; border-top: 1px solid #2a2a45;'
            f' display: flex; gap: 24px; flex-wrap: wrap; align-items: center;">'
            f'<div>'
            f'<div style="font-size: 11px; color: #7777a0; text-transform: uppercase; letter-spacing: 0.5px;">Funds Available</div>'
            f'<div style="font-size: 22px; font-weight: 800; color: #34d399;">₹{_total_available:,.0f}</div>'
            f'</div>'
            f'<div>'
            f'<div style="font-size: 11px; color: #7777a0; text-transform: uppercase; letter-spacing: 0.5px;">Deployed</div>'
            f'<div style="font-size: 22px; font-weight: 800; color: #f59e0b;">₹{_total_deployed:,.0f}</div>'
            f'</div>'
            f'<div>'
            f'<div style="font-size: 11px; color: #7777a0; text-transform: uppercase; letter-spacing: 0.5px;">Total Capital</div>'
            f'<div style="font-size: 22px; font-weight: 800; color: #e0e0f0;">₹{_total_initial:,.0f}</div>'
            f'</div>'
            f'</div>'
        )

    st.markdown(f"""<div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #1a1a30 100%);
        border: 1px solid #2a2a45; border-radius: 18px;
        padding: 28px 32px; margin-bottom: 16px; position: relative; overflow: hidden;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 16px;">
            <div>
                <div style="font-size: 13px; color: #7777a0; text-transform: uppercase; letter-spacing: 1px; font-weight: 700;">
                    Paper Trading — Net P&L</div>
                <div style="font-size: 32px; font-weight: 800; color: {_pnl_color}; margin-top: 4px;">
                    {_pnl_sign}₹{_pnl:,.2f}</div>
                <div style="font-size: 13px; color: #8888a8; margin-top: 2px;">
                    {_pt_all_stats['total_trades']} closed trades | {_pt_all_stats['open_trades']} open</div>
            </div>
            <div style="display: flex; gap: 24px; flex-wrap: wrap;">
                <div style="text-align: center;">
                    <div style="font-size: 20px; font-weight: 800; color: #34d399;">+₹{_total_profit:,.0f}</div>
                    <div style="font-size: 11px; color: #7777a0; text-transform: uppercase; letter-spacing: 0.6px; font-weight: 600;">Total Profit</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 20px; font-weight: 800; color: #f87171;">₹{_total_loss:,.0f}</div>
                    <div style="font-size: 11px; color: #7777a0; text-transform: uppercase; letter-spacing: 0.6px; font-weight: 600;">Total Loss</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 20px; font-weight: 800; color: {_wr_color};">{_wr}%</div>
                    <div style="font-size: 11px; color: #7777a0; text-transform: uppercase; letter-spacing: 0.6px; font-weight: 600;">Win Rate</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 20px; font-weight: 800; color: #e0e0f0;">{_pt_all_stats['profit_factor']}</div>
                    <div style="font-size: 11px; color: #7777a0; text-transform: uppercase; letter-spacing: 0.6px; font-weight: 600;">Profit Factor</div>
                </div>
            </div>
        </div>
        {_funds_html}
    </div>""", unsafe_allow_html=True)

    # -- Fund balance cards (per-type breakdown) --
    if _any_funds:
        _fc1, _fc2 = st.columns(2)

        def _fund_card(label, fund, accent, border_color, leverage=1):
            _total = fund["initial_capital"]
            _avail = fund["available"]
            _deployed = fund["deployed"]
            _rpnl = fund["realized_pnl"]
            _buying_power = _avail * leverage
            _current_val = _avail + _deployed
            _return_pct = ((_current_val - _total) / _total * 100) if _total > 0 else 0
            _ret_color = "#34d399" if _return_pct >= 0 else "#f87171"
            _ret_sign = "+" if _return_pct >= 0 else ""
            _util_pct = (_deployed / _total * 100) if _total > 0 else 0
            _bar_width = min(_util_pct, 100)
            _rpnl_color = "#34d399" if _rpnl >= 0 else "#f87171"
            _rpnl_sign = "+" if _rpnl >= 0 else ""
            _leverage_badge = (
                f'<span style="font-size: 10px; background: rgba(168,85,247,0.2); color: #a855f7; '
                f'padding: 2px 6px; border-radius: 4px; margin-left: 6px; font-weight: 600;">'
                f'{leverage}\u00d7 leverage</span>'
            ) if leverage > 1 else ""
            # Build metric cells
            _cells = (
                f'<div style="flex: 1; min-width: 90px;">'
                f'<div style="font-size: 11px; color: #7777a0; text-transform: uppercase; letter-spacing: 0.5px;">Available</div>'
                f'<div style="font-size: 18px; font-weight: 800; color: #34d399;">\u20b9{_avail:,.0f}</div></div>'
            )
            if leverage > 1:
                _cells += (
                    f'<div style="flex: 1; min-width: 90px;">'
                    f'<div style="font-size: 11px; color: #7777a0; text-transform: uppercase; letter-spacing: 0.5px;">Buying Power</div>'
                    f'<div style="font-size: 18px; font-weight: 800; color: #a855f7;">\u20b9{_buying_power:,.0f}</div></div>'
                )
            _cells += (
                f'<div style="flex: 1; min-width: 90px;">'
                f'<div style="font-size: 11px; color: #7777a0; text-transform: uppercase; letter-spacing: 0.5px;">Deployed</div>'
                f'<div style="font-size: 18px; font-weight: 800; color: #f59e0b;">\u20b9{_deployed:,.0f}</div></div>'
                f'<div style="flex: 1; min-width: 90px;">'
                f'<div style="font-size: 11px; color: #7777a0; text-transform: uppercase; letter-spacing: 0.5px;">Realized P&L</div>'
                f'<div style="font-size: 18px; font-weight: 800; color: {_rpnl_color};">'
                f'{_rpnl_sign}\u20b9{_rpnl:,.0f}</div></div>'
            )
            return (
                f'<div style="background: linear-gradient(135deg, #12122a, #141432);'
                f' border: 1px solid {border_color}; border-radius: 14px;'
                f' padding: 18px 20px; margin-bottom: 12px;">'
                f'<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">'
                f'<span style="font-size: 13px; font-weight: 700; color: {accent};'
                f' text-transform: uppercase; letter-spacing: 1px;">{label}{_leverage_badge}</span>'
                f'<span style="font-size: 12px; color: {_ret_color}; font-weight: 700;">'
                f'{_ret_sign}{_return_pct:.1f}% return</span></div>'
                f'<div style="display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 12px;">{_cells}</div>'
                f'<div style="background: rgba(255,255,255,0.05); border-radius: 6px; height: 8px; overflow: hidden;">'
                f'<div style="background: {accent}; width: {_bar_width}%; height: 100%; border-radius: 6px;'
                f' transition: width 0.3s;"></div></div>'
                f'<div style="font-size: 11px; color: #6b6b80; margin-top: 4px;">'
                f'{_util_pct:.0f}% deployed of \u20b9{_total:,.0f} capital</div></div>'
            )

        with _fc1:
            if _fund_swing["initial_capital"] > 0:
                st.markdown(_fund_card("Swing", _fund_swing, "#00d4ff", "rgba(0,212,255,0.2)"),
                            unsafe_allow_html=True)
        with _fc2:
            if _fund_scalp["initial_capital"] > 0:
                st.markdown(_fund_card("Scalp", _fund_scalp, "#a855f7", "rgba(168,85,247,0.2)", leverage=5),
                            unsafe_allow_html=True)
    else:
        st.markdown("""<div style="background: rgba(245,158,11,0.06); border: 1px solid rgba(245,158,11,0.2);
            border-radius: 10px; padding: 12px 16px; margin-bottom: 12px; font-size: 13px; color: #f59e0b;">
            Set your <b>Paper Trading Funds</b> in the sidebar to track capital usage, deployed amounts, and returns.
        </div>""", unsafe_allow_html=True)

    # -- Detailed stats row --
    if _pt_all_stats["total_trades"] > 0:
        _best = _pt_all_stats["best_trade"]
        _worst = _pt_all_stats["worst_trade"]
        _avg_pct = _pt_all_stats["avg_pnl_pct"]
        _avg_pct_c = "#34d399" if _avg_pct >= 0 else "#f87171"
        st.markdown(f"""<div style="display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 110px; text-align: center; padding: 12px; border-radius: 10px;
                background: rgba(16,185,129,0.06); border: 1px solid rgba(16,185,129,0.15);">
                <div style="font-size: 17px; font-weight: 800; color: #34d399;">+₹{_best:,.0f}</div>
                <div style="font-size: 11px; color: #7777a0; text-transform: uppercase; letter-spacing: 0.5px;">Best Trade</div>
            </div>
            <div style="flex: 1; min-width: 110px; text-align: center; padding: 12px; border-radius: 10px;
                background: rgba(239,68,68,0.06); border: 1px solid rgba(239,68,68,0.15);">
                <div style="font-size: 17px; font-weight: 800; color: #f87171;">₹{_worst:,.0f}</div>
                <div style="font-size: 11px; color: #7777a0; text-transform: uppercase; letter-spacing: 0.5px;">Worst Trade</div>
            </div>
            <div style="flex: 1; min-width: 110px; text-align: center; padding: 12px; border-radius: 10px;
                background: rgba(255,255,255,0.03); border: 1px solid #2a2a45;">
                <div style="font-size: 17px; font-weight: 800; color: {_avg_pct_c};">{"+" if _avg_pct >= 0 else ""}{_avg_pct:.2f}%</div>
                <div style="font-size: 11px; color: #7777a0; text-transform: uppercase; letter-spacing: 0.5px;">Avg P&L %</div>
            </div>
            <div style="flex: 1; min-width: 110px; text-align: center; padding: 12px; border-radius: 10px;
                background: rgba(255,255,255,0.03); border: 1px solid #2a2a45;">
                <div style="font-size: 17px; font-weight: 800; color: #e0e0f0;">{_pt_all_stats['wins']}/{_pt_all_stats['losses']}</div>
                <div style="font-size: 11px; color: #7777a0; text-transform: uppercase; letter-spacing: 0.5px;">Wins / Losses</div>
            </div>
        </div>""", unsafe_allow_html=True)

    # -- Action buttons row --
    _ptb1, _ptb2, _ptb3 = st.columns([4, 1, 1])
    with _ptb2:
        if st.button("Check Trades", type="primary", use_container_width=True, key="pt_check"):
            result = check_open_trades()
            if result["updated"]:
                st.cache_data.clear()
                st.rerun()
            else:
                st.info(f"Checked {result['checked']} — no changes")
    with _ptb3:
        if st.button("Reset All", use_container_width=True, key="pt_reset"):
            count = clear_all_trades()
            st.rerun()

    # -- Helper: generate badge HTML for a price level --
    def _level_badge(label, price_str, is_hit, hit_color="#34d399", pending_color="#555578"):
        if is_hit:
            _rgb = "16,185,129" if hit_color == "#34d399" else "239,68,68"
            return (f"<span style='display:inline-block; padding:4px 12px; border-radius:8px; margin:3px 4px; "
                    f"font-size:12px; font-weight:700; letter-spacing:0.3px; "
                    f"background:rgba({_rgb},0.15); "
                    f"border:1px solid {hit_color}; color:{hit_color};'>"
                    f"{label} ₹{price_str}</span>")
        else:
            return (f"<span style='display:inline-block; padding:4px 12px; border-radius:8px; margin:3px 4px; "
                    f"font-size:12px; font-weight:600; letter-spacing:0.3px; "
                    f"background:rgba(255,255,255,0.03); border:1px solid #2a2a4a; color:{pending_color};'>"
                    f"{label} ₹{price_str}</span>")

    # -- Open positions (PENDING + ACTIVE) --
    _open_all = get_open_trades(None)
    if not _open_all.empty:
        st.markdown("<div class='section-label'>Open Positions</div>", unsafe_allow_html=True)
        for _, row in _open_all.iterrows():
            _is_long = row["direction"] in ("BUY", "LONG")
            _dir_color = "#10b981" if _is_long else "#ef4444"
            _dir_icon = "▲" if _is_long else "▼"
            _type_badge = "SWING" if row["trade_type"] == "swing" else "SCALP"
            _badge_bg = "rgba(0,212,255,0.1)" if row["trade_type"] == "swing" else "rgba(168,85,247,0.1)"
            _badge_border = "rgba(0,212,255,0.3)" if row["trade_type"] == "swing" else "rgba(168,85,247,0.3)"
            _badge_color = "#00d4ff" if row["trade_type"] == "swing" else "#a855f7"
            _opened_short = row["opened_at"][:16].replace("T", " ") if row["opened_at"] else ""
            _is_active = row.get("status") == "ACTIVE"
            _status_label = "ACTIVE" if _is_active else "PENDING"
            _status_color = "#10b981" if _is_active else "#f59e0b"
            _status_rgb = "16,185,129" if _is_active else "245,158,11"

            # Read hit flags (with fallback for old data)
            _entry_hit = bool(row.get("entry_hit", 0))
            _t1_hit = bool(row.get("t1_hit", 0))
            _t2_hit = bool(row.get("t2_hit", 0))
            _t3_hit = bool(row.get("t3_hit", 0))
            _sl_hit = bool(row.get("sl_hit", 0))

            # Build badges
            _badges = _level_badge("ENTRY", f"{row['entry_price']:,.2f}", _entry_hit, "#00d4ff")
            _badges += _level_badge("T1", f"{row['target_1']:,.2f}", _t1_hit)
            if row.get("target_2"):
                _badges += _level_badge("T2", f"{row['target_2']:,.2f}", _t2_hit)
            if row.get("target_3"):
                _badges += _level_badge("T3", f"{row['target_3']:,.2f}", _t3_hit)
            _badges += _level_badge("SL", f"{row['stop_loss']:,.2f}", _sl_hit, "#ef4444", "#ef4444" if not _sl_hit else "#ef4444")

            st.markdown(f"""<div style="
                background: linear-gradient(135deg, #12122a, #141432);
                border: 1px solid #1e1e3a; border-left: 4px solid {_dir_color};
                border-radius: 14px; padding: 18px 22px; margin-bottom: 10px;">
                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 20px; font-weight: 800; color: {_dir_color};">{_dir_icon}</span>
                        <span style="font-size: 17px; font-weight: 800; color: #e0e0f0;">
                            {row['ticker'].replace('.NS','')}</span>
                        <span style="display: inline-block; font-size: 10px; font-weight: 700;
                            text-transform: uppercase; letter-spacing: 0.8px; padding: 3px 10px;
                            border-radius: 20px; background: {_badge_bg}; border: 1px solid {_badge_border};
                            color: {_badge_color};">{_type_badge}</span>
                        <span style="display: inline-block; font-size: 10px; font-weight: 700;
                            text-transform: uppercase; letter-spacing: 0.8px; padding: 3px 10px;
                            border-radius: 20px; background: rgba({_status_rgb},0.1);
                            border: 1px solid {_status_color}; color: {_status_color};">{_status_label}</span>
                    </div>
                    <div style="font-size: 13px; color: #6b6b80;">
                        Qty {row['quantity']} | ₹{row.get('capital_used', 0) or 0:,.0f} capital | {_opened_short}</div>
                </div>
                <div style="margin-top: 14px; display: flex; flex-wrap: wrap; gap: 2px;">
                    {_badges}
                </div>
            </div>""", unsafe_allow_html=True)
            if _is_active:
                # Active trade: only close at market
                if st.button("Close at market price", key=f"close_pt_{row['id']}"):
                    try:
                        import yfinance as yf
                        from src.data_fetcher import _yf_retry
                        stock = yf.Ticker(row["ticker"])
                        hist = _yf_retry(lambda: stock.history(period="1d"))
                        cp = hist["Close"].iloc[-1] if not hist.empty else row["entry_price"]
                    except Exception:
                        cp = row["entry_price"]
                    result = close_trade(row["id"], cp)
                    if "error" not in result:
                        st.rerun()
            else:
                # Pending trade: close or delete
                _btn_c1, _btn_c2, _btn_spacer = st.columns([1, 1, 4])
                with _btn_c1:
                    if st.button("Close at market", key=f"close_pt_{row['id']}", use_container_width=True):
                        try:
                            import yfinance as yf
                            from src.data_fetcher import _yf_retry
                            stock = yf.Ticker(row["ticker"])
                            hist = _yf_retry(lambda: stock.history(period="1d"))
                            cp = hist["Close"].iloc[-1] if not hist.empty else row["entry_price"]
                        except Exception:
                            cp = row["entry_price"]
                        result = close_trade(row["id"], cp)
                        if "error" not in result:
                            st.rerun()
                with _btn_c2:
                    if st.button("🗑 Delete", key=f"del_pt_{row['id']}", use_container_width=True, type="secondary"):
                        result = delete_trade(row["id"])
                        if "error" not in result:
                            st.rerun()
                        else:
                            st.error(result["error"])
    elif _pt_all_stats["total_trades"] == 0 and _pt_all_stats["open_trades"] == 0:
        st.markdown("""<div class="empty-state">
            <div class="es-icon">📝</div>
            <div class="es-title">No paper trades yet</div>
            <div class="es-desc">Go to the <b>Swing</b> or <b>Scalping</b> tab and click
            <b>Paper Trade</b> on an active signal to start tracking.</div>
        </div>""", unsafe_allow_html=True)

    # -- Trade history --
    _hist_all = get_trade_history(None, limit=100)
    if not _hist_all.empty:
        st.markdown("<div class='section-label'>Trade History</div>", unsafe_allow_html=True)

        # Outcome summary bar
        _t_hit = len(_hist_all[_hist_all["status"].str.startswith("HIT")])
        _t_stop = len(_hist_all[_hist_all["status"] == "STOPPED_OUT"])
        _t_exp = len(_hist_all[_hist_all["status"] == "EXPIRED"])
        _t_closed = len(_hist_all[_hist_all["status"] == "CLOSED"])

        st.markdown(f"""<div style="display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 100px; text-align: center; padding: 12px; border-radius: 10px;
                background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.2);">
                <div style="font-size: 20px; font-weight: 800; color: #34d399;">{_t_hit}</div>
                <div style="font-size: 11px; color: #7777a0; text-transform: uppercase; letter-spacing: 0.6px;">Targets Hit</div>
            </div>
            <div style="flex: 1; min-width: 100px; text-align: center; padding: 12px; border-radius: 10px;
                background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.2);">
                <div style="font-size: 20px; font-weight: 800; color: #f87171;">{_t_stop}</div>
                <div style="font-size: 11px; color: #7777a0; text-transform: uppercase; letter-spacing: 0.6px;">Stopped Out</div>
            </div>
            <div style="flex: 1; min-width: 100px; text-align: center; padding: 12px; border-radius: 10px;
                background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.2);">
                <div style="font-size: 20px; font-weight: 800; color: #f59e0b;">{_t_exp}</div>
                <div style="font-size: 11px; color: #7777a0; text-transform: uppercase; letter-spacing: 0.6px;">Expired</div>
            </div>
            <div style="flex: 1; min-width: 100px; text-align: center; padding: 12px; border-radius: 10px;
                background: rgba(0,212,255,0.08); border: 1px solid rgba(0,212,255,0.2);">
                <div style="font-size: 20px; font-weight: 800; color: #00d4ff;">{_t_closed}</div>
                <div style="font-size: 11px; color: #7777a0; text-transform: uppercase; letter-spacing: 0.6px;">Manually Closed</div>
            </div>
        </div>""", unsafe_allow_html=True)

        # Trade history cards with badges
        for _, row in _hist_all.iterrows():
            _status = row["status"]
            if _status.startswith("HIT"):
                _outcome_color = "#34d399"
                _outcome_bg = "rgba(16,185,129,0.06)"
                _outcome_border = "rgba(16,185,129,0.25)"
                _outcome_label = _status.replace("_", " ")
            elif _status == "STOPPED_OUT":
                _outcome_color = "#f87171"
                _outcome_bg = "rgba(239,68,68,0.06)"
                _outcome_border = "rgba(239,68,68,0.25)"
                _outcome_label = "STOPPED OUT"
            elif _status == "EXPIRED":
                _outcome_color = "#f59e0b"
                _outcome_bg = "rgba(245,158,11,0.06)"
                _outcome_border = "rgba(245,158,11,0.25)"
                _outcome_label = "EXPIRED"
            else:
                _outcome_color = "#00d4ff"
                _outcome_bg = "rgba(0,212,255,0.06)"
                _outcome_border = "rgba(0,212,255,0.25)"
                _outcome_label = "CLOSED"

            _pnl_val = row.get("pnl", 0) or 0
            _pnl_pct = row.get("pnl_pct", 0) or 0
            _pnl_c = "#34d399" if _pnl_val >= 0 else "#f87171"
            _pnl_s = "+" if _pnl_val >= 0 else ""
            _type_badge = "SWING" if row["trade_type"] == "swing" else "SCALP"
            _badge_color = "#00d4ff" if row["trade_type"] == "swing" else "#a855f7"
            _opened = (row.get("opened_at") or "")[:10]
            _closed = (row.get("closed_at") or "")[:10]

            # Build hit badges for history
            _entry_hit = bool(row.get("entry_hit", 0))
            _t1_hit = bool(row.get("t1_hit", 0))
            _t2_hit = bool(row.get("t2_hit", 0))
            _t3_hit = bool(row.get("t3_hit", 0))
            _sl_hit = bool(row.get("sl_hit", 0))

            _h_badges = _level_badge("ENTRY", f"{row['entry_price']:,.2f}", _entry_hit, "#00d4ff")
            _h_badges += _level_badge("T1", f"{row['target_1']:,.2f}", _t1_hit)
            if row.get("target_2"):
                _h_badges += _level_badge("T2", f"{row['target_2']:,.2f}", _t2_hit)
            if row.get("target_3"):
                _h_badges += _level_badge("T3", f"{row['target_3']:,.2f}", _t3_hit)
            _h_badges += _level_badge("SL", f"{row['stop_loss']:,.2f}", _sl_hit, "#ef4444")

            st.markdown(f"""<div style="
                background: {_outcome_bg}; border: 1px solid {_outcome_border};
                border-radius: 12px; padding: 16px 20px; margin-bottom: 8px;">
                <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 16px; font-weight: 800; color: #e0e0f0;">
                            {row['ticker'].replace('.NS','')}</span>
                        <span style="font-size: 10px; font-weight: 700; color: {_badge_color};
                            text-transform: uppercase; letter-spacing: 0.6px;">{_type_badge}</span>
                        <span style="font-size: 10px; font-weight: 700; color: {_outcome_color};
                            text-transform: uppercase; letter-spacing: 0.6px; padding: 2px 8px;
                            border-radius: 6px; background: {_outcome_bg}; border: 1px solid {_outcome_border};">
                            {_outcome_label}</span>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 18px; font-weight: 800; color: {_pnl_c};">
                            {_pnl_s}₹{_pnl_val:,.2f}</span>
                        <span style="font-size: 13px; color: {_pnl_c}; margin-left: 6px;">
                            ({_pnl_s}{_pnl_pct:.2f}%)</span>
                    </div>
                </div>
                <div style="margin-top: 10px; display: flex; flex-wrap: wrap; gap: 2px;">
                    {_h_badges}
                </div>
                <div style="display: flex; gap: 20px; margin-top: 8px; font-size: 13px; color: #6b6b80; flex-wrap: wrap;">
                    <span>Qty {row['quantity']}</span>
                    <span>₹{row.get('capital_used', 0) or 0:,.0f} capital</span>
                    <span>{_opened} → {_closed}</span>
                </div>
            </div>""", unsafe_allow_html=True)

        # Swing vs Scalp comparison
        if _pt_swing_stats["total_trades"] > 0 and _pt_scalp_stats["total_trades"] > 0:
            st.markdown("<div class='section-label'>Swing vs Scalp Comparison</div>", unsafe_allow_html=True)
            _cmp1, _cmp2 = st.columns(2)
            with _cmp1:
                _sw_pnl_c = "#34d399" if _pt_swing_stats["total_pnl"] >= 0 else "#f87171"
                _sw_roi_line = ""
                if _fund_swing["initial_capital"] > 0:
                    _sw_roi = _fund_swing["realized_pnl"] / _fund_swing["initial_capital"] * 100
                    _sw_roi_c = "#34d399" if _sw_roi >= 0 else "#f87171"
                    _sw_roi_line = f'<div style="font-size: 13px; color: {_sw_roi_c}; margin-top: 4px;">ROI: {"+" if _sw_roi >= 0 else ""}{_sw_roi:.1f}%</div>'
                st.markdown(f"""<div style="background: linear-gradient(135deg, #12122a, #141432);
                    border: 1px solid rgba(0,212,255,0.2); border-radius: 14px; padding: 20px; text-align: center;">
                    <div style="font-size: 13px; font-weight: 700; color: #00d4ff; text-transform: uppercase;
                        letter-spacing: 1px;">Swing Trading</div>
                    <div style="font-size: 28px; font-weight: 800; color: {_sw_pnl_c}; margin: 8px 0;">
                        ₹{_pt_swing_stats['total_pnl']:,.2f}</div>
                    <div style="font-size: 13px; color: #8888a8;">
                        {_pt_swing_stats['total_trades']} trades | {_pt_swing_stats['win_rate']}% win rate</div>
                    {_sw_roi_line}
                </div>""", unsafe_allow_html=True)
            with _cmp2:
                _sc_pnl_c = "#34d399" if _pt_scalp_stats["total_pnl"] >= 0 else "#f87171"
                _sc_roi_line = ""
                if _fund_scalp["initial_capital"] > 0:
                    _sc_roi = _fund_scalp["realized_pnl"] / _fund_scalp["initial_capital"] * 100
                    _sc_roi_c = "#34d399" if _sc_roi >= 0 else "#f87171"
                    _sc_roi_line = f'<div style="font-size: 13px; color: {_sc_roi_c}; margin-top: 4px;">ROI: {"+" if _sc_roi >= 0 else ""}{_sc_roi:.1f}%</div>'
                st.markdown(f"""<div style="background: linear-gradient(135deg, #12122a, #141432);
                    border: 1px solid rgba(168,85,247,0.2); border-radius: 14px; padding: 20px; text-align: center;">
                    <div style="font-size: 13px; font-weight: 700; color: #a855f7; text-transform: uppercase;
                        letter-spacing: 1px;">Scalp Trading</div>
                    <div style="font-size: 28px; font-weight: 800; color: {_sc_pnl_c}; margin: 8px 0;">
                        ₹{_pt_scalp_stats['total_pnl']:,.2f}</div>
                    <div style="font-size: 13px; color: #8888a8;">
                        {_pt_scalp_stats['total_trades']} trades | {_pt_scalp_stats['win_rate']}% win rate</div>
                    {_sc_roi_line}
                </div>""", unsafe_allow_html=True)


# ========================================================================
# TAB 5: SENTIMENT
# ========================================================================
with tab_sentiment:
    help_box("Get AI-powered sentiment analysis for this stock. <b>Step 1:</b> Fetch news & generate a prompt. "
             "<b>Step 2:</b> Copy the prompt and paste it into ChatGPT or Claude. "
             "<b>Step 3:</b> Paste the AI's response back here to see the analysis.")

    sent_col1, sent_col2 = st.columns([3, 1])
    with sent_col1:
        fetch_news_btn = st.button("📰 Step 1: Fetch News & Generate Prompt", type="primary", use_container_width=True)
    with sent_col2:
        clear_sent = st.button("🗑️ Clear", use_container_width=True)

    if clear_sent:
        for k in ["sentiment_prompt", "sentiment_news", "sentiment_result"]:
            st.session_state.pop(k, None)
        st.rerun()

    if fetch_news_btn:
        with st.spinner("Fetching news..."):
            news = get_stock_news(ticker)
        if not news:
            st.warning("No recent news found for this stock. Try a different stock.")
        else:
            st.session_state["sentiment_news"] = news
            headlines = "\n".join(
                f"- {item['title']} (Source: {item['publisher']})"
                for item in news if item.get("title")
            )
            stock_name = info.get("name", ticker)
            prompt = f"""Analyze the sentiment of these stock news headlines for {stock_name} ({ticker}), an Indian stock market company.

Rate the overall sentiment and each headline individually.

Headlines:
{headlines}

Respond in this EXACT format (no markdown, no extra text):
OVERALL: [Bullish/Bearish/Neutral]
SCORE: [float from -1.0 to 1.0]
SUMMARY: [1-2 sentence analysis]
DETAILS:
- [headline]: [Bullish/Bearish/Neutral]"""
            st.session_state["sentiment_prompt"] = prompt

    if "sentiment_prompt" in st.session_state:
        st.markdown("#### Step 1 Complete — Copy this prompt:")
        st.markdown(f"<div class='prompt-box'>{st.session_state['sentiment_prompt']}</div>", unsafe_allow_html=True)
        st.code(st.session_state["sentiment_prompt"], language=None)
        help_box("Copy the prompt above and paste it into <b>ChatGPT</b>, <b>Claude</b>, or any AI chatbot. "
                 "Then paste the response back below.")

        st.markdown("---")
        st.markdown("#### Step 2: Paste the AI's response")
        ai_response = st.text_area(
            "Paste AI response here",
            height=200,
            placeholder="OVERALL: Bullish\nSCORE: 0.6\nSUMMARY: ...\nDETAILS:\n- headline: Bullish\n...",
            key="sentiment_response_input",
            label_visibility="collapsed",
        )

        if st.button("✨ Step 3: Analyze Response", type="primary", use_container_width=True):
            if ai_response.strip():
                from src.sentiment import _parse_sentiment_response
                news = st.session_state.get("sentiment_news", [])
                sentiment = _parse_sentiment_response(ai_response.strip(), news)
                st.session_state["sentiment_result"] = sentiment
            else:
                st.warning("Please paste the AI response first.")

    if "sentiment_result" in st.session_state:
        sentiment = st.session_state["sentiment_result"]
        st.markdown("---")
        st.markdown("#### Sentiment Analysis Result")
        sc = {"Bullish": "action-up", "Bearish": "action-down", "Neutral": "action-sideways"}
        action_card(f"Sentiment: {sentiment['overall']}", f"Score: {sentiment['score']:+.2f}",
                    sc.get(sentiment["overall"], "action-sideways"))
        if sentiment.get("summary"):
            verdict_box(f"<b>Summary:</b> {sentiment['summary']}", "good" if sentiment["overall"] == "Bullish" else "bad" if sentiment["overall"] == "Bearish" else "neutral")
        for d in sentiment.get("details", []):
            ds = "good" if d["sentiment"] == "Bullish" else "bad" if d["sentiment"] == "Bearish" else "neutral"
            de = {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "🟡"}.get(d["sentiment"], "⚪")
            verdict_box(f"{de} <b>{d['sentiment']}</b> — {d['headline']}", ds)


# ========================================================================
# TAB 7: SCREENER
# ========================================================================
with tab_screener:
    help_box("<b>Stock Screener</b> scans multiple stocks using a <b>multi-factor approach</b> inspired by BlackRock's SAE methodology. "
             "It combines <b>Technical Signals</b> (RSI, MACD, ADX, EMAs) with <b>Fundamental Momentum</b> (earnings growth, volume trends) "
             "to find the best stocks for swing trading and scalping.")

    scr_c1, scr_c2 = st.columns([2, 1])
    with scr_c1:
        scan_scope = st.selectbox("Scan scope", ["All Popular Stocks"] + list(SECTOR_STOCKS.keys()),
                                   help="Scan all 25 popular stocks or pick a sector")
    with scr_c2:
        scan_btn = st.button("🔍 Scan Now", type="primary", use_container_width=True)

    if scan_btn:
        if scan_scope == "All Popular Stocks":
            scan_tickers = list(POPULAR_INDIAN_STOCKS.keys())
        else:
            scan_tickers = SECTOR_STOCKS[scan_scope]

        total_steps = len(scan_tickers) * 2  # swing + scalp pass
        progress = st.progress(0, text="Scanning stocks...")
        with st.spinner("Analyzing each stock for swing & scalping..."):
            scan_results = []
            scalp_results = []
            for idx, t in enumerate(scan_tickers):
                # --- Swing scan ---
                progress.progress((idx * 2 + 1) / total_steps, text=f"Swing scan: {t.replace('.NS', '')}...")
                try:
                    t_df = fetch_stock_data(t, period_years=1)
                    t_df = add_technical_indicators(t_df)
                    t_signal = generate_swing_signals(t_df)
                    t_latest = t_df.iloc[-1]
                    t_setup = calculate_trade_setup(t_df, t_signal)
                    t_patterns = identify_swing_patterns(t_df)
                    # Multi-factor composite score (inspired by BlackRock SAE)
                    # Factor 1: Technical signal strength (40%)
                    tech_score = t_signal.confidence

                    # Factor 2: Momentum (20%) — price above key MAs + positive ROC
                    momentum_score = 0
                    if t_latest["Close"] > t_latest.get("SMA_20", t_latest["Close"]):
                        momentum_score += 0.25
                    if t_latest["Close"] > t_latest.get("SMA_50", t_latest["Close"]):
                        momentum_score += 0.25
                    if t_latest["Close"] > t_latest.get("SMA_200", t_latest["Close"]):
                        momentum_score += 0.25
                    roc = t_latest.get("ROC", 0)
                    if roc > 0:
                        momentum_score += min(0.25, roc / 20)

                    # Factor 3: Quality (20%) — healthy RSI, strong ADX, volume confirmation
                    quality_score = 0
                    rsi_val = t_latest.get("RSI", 50)
                    if 40 < rsi_val < 65:
                        quality_score += 0.4  # Healthy RSI zone
                    elif 30 < rsi_val <= 40:
                        quality_score += 0.3  # Oversold recovery zone
                    if t_latest.get("ADX", 0) > 25:
                        quality_score += 0.3  # Strong trend
                    if t_latest.get("Volume_Ratio", 1) > 1.0:
                        quality_score += 0.3  # Active volume

                    # Factor 4: Sentiment/Price Action (20%) — patterns + BB position
                    sentiment_score = 0
                    if t_patterns:
                        bullish_patterns = sum(1 for p in t_patterns if "bullish" in p["implication"].lower())
                        sentiment_score += min(0.5, bullish_patterns * 0.25)
                    # Price in lower half of BB = more upside potential
                    if "BB_Upper" in t_df.columns and "BB_Lower" in t_df.columns:
                        bb_range = t_latest["BB_Upper"] - t_latest["BB_Lower"]
                        if bb_range > 0:
                            bb_position = (t_latest["Close"] - t_latest["BB_Lower"]) / bb_range
                            if bb_position < 0.4:
                                sentiment_score += 0.5

                    composite = (0.4 * tech_score + 0.2 * momentum_score +
                                 0.2 * quality_score + 0.2 * sentiment_score)

                    scan_results.append({
                        "ticker": t,
                        "name": POPULAR_INDIAN_STOCKS.get(t, t.replace(".NS", "")),
                        "price": t_latest["Close"],
                        "signal": t_signal.signal,
                        "strength": t_signal.strength,
                        "confidence": t_signal.confidence,
                        "composite_score": round(composite, 3),
                        "reasons": t_signal.reasons,
                        "rsi": t_latest.get("RSI", 0),
                        "macd": t_latest.get("MACD", 0),
                        "macd_signal": t_latest.get("MACD_Signal", 0),
                        "adx": t_latest.get("ADX", 0),
                        "volume_ratio": t_latest.get("Volume_Ratio", 1),
                        "setup": t_setup,
                        "patterns": t_patterns,
                    })
                except Exception:
                    pass

                # --- Scalp scan ---
                progress.progress((idx * 2 + 2) / total_steps, text=f"Scalp scan: {t.replace('.NS', '')}...")
                try:
                    intra_df = fetch_intraday_data(t, interval="5m", period="5d")
                    intra_df = add_scalping_indicators(intra_df)
                    s_signal = generate_scalp_signal(intra_df)
                    s_micro = get_market_microstructure(intra_df)
                    s_levels = get_scalping_levels(intra_df)
                    scalp_results.append({
                        "ticker": t,
                        "name": POPULAR_INDIAN_STOCKS.get(t, t.replace(".NS", "")),
                        "price": intra_df["Close"].iloc[-1],
                        "signal": s_signal.signal,
                        "strength": s_signal.strength,
                        "confidence": s_signal.confidence,
                        "reasons": s_signal.reasons,
                        "entry": s_signal.entry_price,
                        "stop_loss": s_signal.stop_loss,
                        "target_1": s_signal.target_1,
                        "target_2": s_signal.target_2,
                        "risk_reward": s_signal.risk_reward,
                        "scalpability": s_micro["scalpability_score"],
                        "trend": s_micro["trend"],
                        "volatility": s_micro["volatility_regime"],
                        "vwap": s_levels["vwap"],
                    })
                except Exception:
                    pass

            scan_results.sort(key=lambda x: x["composite_score"], reverse=True)
            # Sort scalp results by combined score: scalpability (weight 0.6) + confidence (weight 0.4)
            # This ensures high-scalpability stocks rank higher than just high-confidence ones
            scalp_results.sort(
                key=lambda x: 0.6 * (x["scalpability"] / 100) + 0.4 * x["confidence"],
                reverse=True,
            )
            progress.progress(1.0, text="Scan complete!")

        st.session_state["scan_results"] = scan_results
        st.session_state["scalp_results"] = scalp_results

    if ("scan_results" not in st.session_state or not st.session_state["scan_results"]) and not scan_btn:
        st.markdown("""<div class="empty-state">
            <div class="es-icon">🔍</div>
            <div class="es-title">Ready to scan</div>
            <div class="es-desc">Click <b>Scan Now</b> to analyze stocks for swing and scalping opportunities.<br>
            Uses multi-factor scoring (Technical + Momentum + Quality + Sentiment).</div>
        </div>""", unsafe_allow_html=True)

    if "scan_results" in st.session_state and st.session_state["scan_results"]:
        results = st.session_state["scan_results"]
        scalp_res = st.session_state.get("scalp_results", [])

        # Top picks
        buys = [r for r in results if r["signal"] == "BUY"]
        sells = [r for r in results if r["signal"] == "SELL"]
        holds = [r for r in results if r["signal"] == "HOLD"]

        # ── SUMMARY BAR ──
        total_scanned = len(results)
        st.markdown(f"""<div class="quick-stats">
            <div class="quick-stat">
                <div class="qs-label">Scanned</div>
                <div class="qs-value">{total_scanned}</div>
                <div class="qs-delta" style="color: #7777a0;">stocks</div>
            </div>
            <div class="quick-stat">
                <div class="qs-label">Buy Signals</div>
                <div class="qs-value" style="color: #34d399;">{len(buys)}</div>
                <div class="qs-delta" style="color: #34d399;">{'🟢' * min(len(buys), 5)}</div>
            </div>
            <div class="quick-stat">
                <div class="qs-label">Sell Signals</div>
                <div class="qs-value" style="color: #f87171;">{len(sells)}</div>
                <div class="qs-delta" style="color: #f87171;">{'🔴' * min(len(sells), 5)}</div>
            </div>
            <div class="quick-stat">
                <div class="qs-label">Hold</div>
                <div class="qs-value" style="color: #fbbf24;">{len(holds)}</div>
                <div class="qs-delta" style="color: #fbbf24;">{'🟡' * min(len(holds), 5)}</div>
            </div>
        </div>""", unsafe_allow_html=True)

        # ── TOP SWING OPPORTUNITIES ──
        st.markdown("#### Top Swing Picks")
        if buys:
            top = buys[:3]
            tc = st.columns(len(top))
            for i, pick in enumerate(top):
                with tc[i]:
                    badge = "<span class='badge-best'>Best</span>" if i == 0 else (f"<span class='badge-rank'>#{i+1}</span>" if len(top) > 1 else "")
                    action_card(
                        f"BUY: {pick['name']}{badge}",
                        f"₹{pick['price']:,.2f} | {pick['strength']} | Score: {pick.get('composite_score', 0)*100:.0f}%",
                        "action-buy",
                    )
                    p_setup = pick.get("setup")
                    if p_setup:
                        sw_ch = calc_angel_one_charges(p_setup.entry_price, p_setup.target_1, 1)
                        pnl_class = "profit" if sw_ch['net_profit'] > 0 else "loss"
                        st.markdown(f"""<div class='setup-card'>
                            <b>Entry:</b> ₹{p_setup.entry_price:,.2f} &nbsp;|&nbsp; <b>SL:</b> ₹{p_setup.stop_loss:,.2f}<br>
                            <b>T1:</b> ₹{p_setup.target_1:,.2f} (1:{p_setup.risk_reward_1}) &nbsp;|&nbsp;
                            <b>T2:</b> ₹{p_setup.target_2:,.2f} (1:{p_setup.risk_reward_2})<br>
                            <span class='{pnl_class}'><b>Net:</b> ₹{sw_ch['net_profit']:.2f}</span>
                            <span class='charges-dim'>(charges ₹{sw_ch['total_charges']:.2f})</span>
                        </div>""", unsafe_allow_html=True)
                    # Action buttons: Paper Trade + Analyse
                    _sw_pt_col, _sw_link_col = st.columns(2)
                    with _sw_pt_col:
                        if st.button("📝 Paper Trade", key=f"scr_sw_pt_{i}", use_container_width=True):
                            p_setup = pick.get("setup")
                            if p_setup:
                                try:
                                    _tid = place_trade(
                                        ticker=pick["ticker"], trade_type="swing",
                                        direction=pick["signal"],
                                        entry_price=p_setup.entry_price,
                                        stop_loss=p_setup.stop_loss,
                                        target_1=p_setup.target_1,
                                        target_2=p_setup.target_2,
                                        target_3=p_setup.target_3,
                                        quantity=1,
                                        signal_strength=pick["strength"],
                                        confidence=pick["confidence"],
                                        reasons=", ".join(pick["reasons"]) if pick["reasons"] else "",
                                    )
                                    st.success(f"Trade #{_tid} placed!")
                                except ValueError as e:
                                    st.error(str(e))
                            else:
                                st.warning("No trade setup available")
                    with _sw_link_col:
                        st.markdown(
                            f'<a class="scr-btn scr-btn-analyse" href="?stock={pick["ticker"]}&tab=swing" target="_blank">'
                            f'📈 Swing Analysis</a>',
                            unsafe_allow_html=True,
                        )
        else:
            st.info("No BUY signals found.")

        # ── TOP SCALPING OPPORTUNITIES ──
        scalp_longs_all = [r for r in scalp_res if r["signal"] == "LONG"] if scalp_res else []
        if scalp_longs_all:
            st.markdown("#### Top Scalping Picks")
            top_sc = scalp_longs_all[:3]
            sc_top_cols = st.columns(len(top_sc))
            for i, pick in enumerate(top_sc):
                with sc_top_cols[i]:
                    badge = "<span class='badge-best'>Best</span>" if i == 0 else (f"<span class='badge-rank'>#{i+1}</span>" if len(top_sc) > 1 else "")
                    action_card(
                        f"LONG: {pick['name']}{badge}",
                        f"₹{pick['price']:,.2f} | {pick['strength']} | Scalp {pick['scalpability']}/100",
                        "action-buy",
                    )
                    sc_ch = calc_angel_one_charges(pick['entry'], pick['target_1'], 1)
                    pnl_class = "profit" if sc_ch['net_profit'] > 0 else "loss"
                    st.markdown(f"""<div class='setup-card'>
                        <b>Entry:</b> ₹{pick['entry']:,.2f} &nbsp;|&nbsp; <b>SL:</b> ₹{pick['stop_loss']:,.2f}<br>
                        <b>T1:</b> ₹{pick['target_1']:,.2f} &nbsp;|&nbsp; <b>T2:</b> ₹{pick['target_2']:,.2f} &nbsp;|&nbsp; <b>R:R:</b> {pick['risk_reward']}<br>
                        <span class='{pnl_class}'><b>Net:</b> ₹{sc_ch['net_profit']:.2f}</span>
                        <span class='charges-dim'>(charges ₹{sc_ch['total_charges']:.2f})</span>
                    </div>""", unsafe_allow_html=True)
                    # Action buttons: Paper Trade + Analyse
                    _sc_pt_col, _sc_link_col = st.columns(2)
                    with _sc_pt_col:
                        if st.button("📝 Paper Trade", key=f"scr_sc_pt_{i}", use_container_width=True):
                            try:
                                _tid = place_trade(
                                    ticker=pick["ticker"], trade_type="scalp",
                                    direction=pick["signal"],
                                    entry_price=pick["entry"],
                                    stop_loss=pick["stop_loss"],
                                    target_1=pick["target_1"],
                                    target_2=pick["target_2"],
                                    quantity=1,
                                    signal_strength=pick["strength"],
                                    confidence=pick["confidence"],
                                    reasons=", ".join(pick["reasons"]) if pick["reasons"] else "",
                                )
                                st.success(f"Trade #{_tid} placed!")
                            except ValueError as e:
                                st.error(str(e))
                    with _sc_link_col:
                        st.markdown(
                            f'<a class="scr-btn scr-btn-analyse" href="?stock={pick["ticker"]}&tab=scalp" target="_blank">'
                            f'⚡ Scalp Analysis</a>',
                            unsafe_allow_html=True,
                        )

        if sells:
            st.markdown("#### Stocks to Avoid")
            sell_top = sells[:3]
            sc_cols = st.columns(len(sell_top))
            for i, pick in enumerate(sell_top):
                with sc_cols[i]:
                    action_card(
                        f"SELL: {pick['name']}",
                        f"₹{pick['price']:,.2f} | {pick['strength']}",
                        "action-sell",
                    )

        # ── FULL RESULTS TABLES ──
        st.markdown("---")
        st.markdown("#### All Swing Signals")
        help_box("<b>Score</b> = Multi-factor composite (Technical 40% + Momentum 20% + Quality 20% + Sentiment 20%). "
                 "Inspired by BlackRock's SAE methodology.")
        table_rows = []
        for r in results:
            sig_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(r["signal"], "⚪")
            r_setup = r.get("setup")
            if r_setup and r["signal"] == "BUY":
                r_ch = calc_angel_one_charges(r_setup.entry_price, r_setup.target_1, 1)
                entry_str = f"{r_setup.entry_price:,.2f}"
                t1_str = f"{r_setup.target_1:,.2f}"
                sl_str = f"{r_setup.stop_loss:,.2f}"
                profit_str = f"{r_ch['net_profit']:.2f}"
            elif r_setup and r["signal"] == "SELL":
                r_ch = calc_angel_one_charges(r_setup.target_1, r_setup.entry_price, 1)
                entry_str = f"{r_setup.entry_price:,.2f}"
                t1_str = f"{r_setup.target_1:,.2f}"
                sl_str = f"{r_setup.stop_loss:,.2f}"
                profit_str = f"{r_ch['net_profit']:.2f}"
            else:
                entry_str = t1_str = sl_str = profit_str = "—"
            table_rows.append({
                "Stock": r["name"],
                "Price (₹)": f"{r['price']:,.2f}",
                "Signal": f"{sig_emoji} {r['signal']}",
                "Strength": r["strength"],
                "Entry (₹)": entry_str,
                "Target 1 (₹)": t1_str,
                "SL (₹)": sl_str,
                "Profit/share (₹)": profit_str,
                "Score": f"{r.get('composite_score', 0)*100:.0f}%",
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        if scalp_res:
            st.markdown("#### All Scalping Signals")
            scalp_table = []
            for r in scalp_res:
                sig_emoji = {"LONG": "🟢", "SHORT": "🔴", "NO_TRADE": "🟡"}.get(r["signal"], "⚪")
                if r["signal"] == "LONG":
                    sc_ch = calc_angel_one_charges(r['entry'], r['target_1'], 1)
                    profit_str = f"{sc_ch['net_profit']:.2f}"
                elif r["signal"] == "SHORT":
                    sc_ch = calc_angel_one_charges(r['target_1'], r['entry'], 1)
                    profit_str = f"{sc_ch['net_profit']:.2f}"
                else:
                    profit_str = "—"
                scalp_table.append({
                    "Stock": r["name"],
                    "Price (₹)": f"{r['price']:,.2f}",
                    "Signal": f"{sig_emoji} {r['signal']}",
                    "Entry (₹)": f"{r['entry']:,.2f}",
                    "Target 1 (₹)": f"{r['target_1']:,.2f}",
                    "SL (₹)": f"{r['stop_loss']:,.2f}",
                    "Profit/share (₹)": profit_str,
                    "R:R": f"{r['risk_reward']}",
                    "Scalpability": f"{r['scalpability']}/100",
                    "Trend": r["trend"],
                })
            st.dataframe(pd.DataFrame(scalp_table), use_container_width=True, hide_index=True)

        swing_summary = f"{len(buys)} BUY | {len(sells)} SELL | {len(holds)} HOLD"
        scalp_longs_count = len([r for r in scalp_res if r["signal"] == "LONG"])
        scalp_shorts_count = len([r for r in scalp_res if r["signal"] == "SHORT"])
        scalp_no_count = len([r for r in scalp_res if r["signal"] == "NO_TRADE"])
        scalp_summary = f"{scalp_longs_count} LONG | {scalp_shorts_count} SHORT | {scalp_no_count} NO_TRADE"
        st.markdown(f"**Swing:** {swing_summary} out of {len(results)} stocks | **Scalp:** {scalp_summary} out of {len(scalp_res)} stocks")


# ========================================================================
# TAB 8: FUNDAMENTALS
# ========================================================================
with tab_fund:
    help_box("<b>Fundamentals</b> show the financial health of a company — how profitable it is, "
             "how much debt it has, and whether the stock price is cheap or expensive relative to earnings.")

    with st.spinner("Loading fundamentals..."):
        try:
            fund = get_stock_fundamentals(ticker)
        except Exception as e:
            st.error(f"Could not load fundamentals: {e}")
            fund = None

    if fund:
        st.markdown(f"### {fund.get('name', ticker)}")

        # --- 52-week range ---
        w52_high = fund.get("52w_high")
        w52_low = fund.get("52w_low")
        cur_price = fund.get("current_price", 0)
        if w52_high and w52_low and w52_high > w52_low:
            pct_in_range = ((cur_price - w52_low) / (w52_high - w52_low)) * 100
            st.markdown(f"**52-Week Range:** ₹{w52_low:,.2f} — ₹{w52_high:,.2f} (currently at {pct_in_range:.0f}%)")
            st.progress(min(max(pct_in_range / 100, 0), 1.0))
            if pct_in_range > 90:
                verdict_box("Stock is near its <b>52-week high</b> — may be overextended.", "bad")
            elif pct_in_range < 10:
                verdict_box("Stock is near its <b>52-week low</b> — could be undervalued or in trouble.", "neutral")

        # --- VALUATION ---
        st.markdown("#### Valuation")
        help_box("<b>P/E Ratio</b> = price ÷ earnings. Low P/E may mean undervalued. High P/E means expensive or high growth. "
                 "<b>P/B</b> = price ÷ book value. <b>EV/EBITDA</b> compares enterprise value to operating profit.")
        v1, v2, v3, v4 = st.columns(4)
        pe = fund.get("trailing_pe")
        fpe = fund.get("forward_pe")
        pb = fund.get("price_to_book")
        ev_ebitda = fund.get("ev_to_ebitda")

        v1.metric("P/E Ratio", f"{pe:.1f}" if pe else "N/A")
        v2.metric("Forward P/E", f"{fpe:.1f}" if fpe else "N/A")
        v3.metric("Price/Book", f"{pb:.2f}" if pb else "N/A")
        v4.metric("EV/EBITDA", f"{ev_ebitda:.1f}" if ev_ebitda else "N/A")

        if pe:
            if pe < 15:
                verdict_box(f"P/E of <b>{pe:.1f}</b> — relatively <b>cheap</b>. Could be undervalued.", "good")
            elif pe < 30:
                verdict_box(f"P/E of <b>{pe:.1f}</b> — <b>fairly valued</b>. Reasonable for a quality company.", "neutral")
            else:
                verdict_box(f"P/E of <b>{pe:.1f}</b> — <b>expensive</b>. Market expects high growth.", "bad")

        peg = fund.get("peg_ratio")
        ps = fund.get("price_to_sales")
        v5, v6, v7, v8 = st.columns(4)
        v5.metric("PEG Ratio", f"{peg:.2f}" if peg else "N/A")
        v6.metric("Price/Sales", f"{ps:.2f}" if ps else "N/A")
        mcap = fund.get("market_cap")
        v7.metric("Market Cap", f"₹{mcap/1e7:,.0f} Cr" if mcap else "N/A")
        ev = fund.get("enterprise_value")
        v8.metric("Enterprise Value", f"₹{ev/1e7:,.0f} Cr" if ev else "N/A")

        # --- PROFITABILITY ---
        st.markdown("#### Profitability")
        help_box("<b>ROE</b> = return on equity — how well the company uses shareholder money. >15% is good. "
                 "<b>Profit Margin</b> = % of revenue kept as profit. Higher is better.")
        p1, p2, p3, p4, p5 = st.columns(5)
        pm = fund.get("profit_margin")
        om = fund.get("operating_margin")
        gm = fund.get("gross_margin")
        roe = fund.get("roe")
        roa = fund.get("roa")

        p1.metric("Profit Margin", f"{pm*100:.1f}%" if pm else "N/A")
        p2.metric("Operating Margin", f"{om*100:.1f}%" if om else "N/A")
        p3.metric("Gross Margin", f"{gm*100:.1f}%" if gm else "N/A")
        p4.metric("ROE", f"{roe*100:.1f}%" if roe else "N/A")
        p5.metric("ROA", f"{roa*100:.1f}%" if roa else "N/A")

        if roe:
            if roe > 0.20:
                verdict_box(f"ROE of <b>{roe*100:.1f}%</b> — <b>excellent</b>. Company generates strong returns.", "good")
            elif roe > 0.10:
                verdict_box(f"ROE of <b>{roe*100:.1f}%</b> — <b>decent</b>. Adequate returns for shareholders.", "neutral")
            else:
                verdict_box(f"ROE of <b>{roe*100:.1f}%</b> — <b>weak</b>. Low returns on invested capital.", "bad")

        # --- FINANCIAL HEALTH ---
        st.markdown("#### Financial Health")
        help_box("<b>Debt/Equity</b> = how much borrowed vs owned. Lower is safer. "
                 "<b>Current Ratio</b> = can the company pay short-term bills? >1.5 is healthy.")
        h1, h2, h3, h4 = st.columns(4)
        de = fund.get("debt_to_equity")
        cr = fund.get("current_ratio")
        td = fund.get("total_debt")
        tc = fund.get("total_cash")

        h1.metric("Debt/Equity", f"{de:.1f}" if de else "N/A")
        h2.metric("Current Ratio", f"{cr:.2f}" if cr else "N/A")
        h3.metric("Total Debt", f"₹{td/1e7:,.0f} Cr" if td else "N/A")
        h4.metric("Total Cash", f"₹{tc/1e7:,.0f} Cr" if tc else "N/A")

        if de is not None:
            if de < 50:
                verdict_box(f"Debt/Equity of <b>{de:.1f}</b> — <b>low debt</b>. Financially strong.", "good")
            elif de < 150:
                verdict_box(f"Debt/Equity of <b>{de:.1f}</b> — <b>moderate debt</b>. Manageable but watch it.", "neutral")
            else:
                verdict_box(f"Debt/Equity of <b>{de:.1f}</b> — <b>high debt</b>. Risk if earnings drop.", "bad")

        # --- DIVIDENDS & GROWTH ---
        st.markdown("#### Dividends & Growth")
        help_box("<b>Dividend Yield</b> = annual dividend ÷ stock price. Higher yield = more income. "
                 "<b>Revenue Growth</b> and <b>Earnings Growth</b> show how fast the company is expanding.")
        dg1, dg2, dg3, dg4 = st.columns(4)
        dy = fund.get("dividend_yield")
        pr = fund.get("payout_ratio")
        rg = fund.get("revenue_growth")
        eg = fund.get("earnings_growth")

        dg1.metric("Dividend Yield", f"{dy*100:.2f}%" if dy else "N/A")
        dg2.metric("Payout Ratio", f"{pr*100:.1f}%" if pr else "N/A")
        dg3.metric("Revenue Growth", f"{rg*100:.1f}%" if rg else "N/A")
        dg4.metric("Earnings Growth", f"{eg*100:.1f}%" if eg else "N/A")

        eps = fund.get("eps")
        feps = fund.get("forward_eps")
        rev = fund.get("revenue")
        earn = fund.get("earnings")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("EPS", f"₹{eps:.2f}" if eps else "N/A")
        e2.metric("Forward EPS", f"₹{feps:.2f}" if feps else "N/A")
        e3.metric("Revenue", f"₹{rev/1e7:,.0f} Cr" if rev else "N/A")
        e4.metric("Net Income", f"₹{earn/1e7:,.0f} Cr" if earn else "N/A")

        if rg and eg:
            if rg > 0.15 and eg > 0.15:
                verdict_box(f"Revenue growing <b>{rg*100:.1f}%</b> and earnings <b>{eg*100:.1f}%</b> — <b>strong growth</b>.", "good")
            elif rg > 0 and eg > 0:
                verdict_box(f"Revenue growing <b>{rg*100:.1f}%</b> and earnings <b>{eg*100:.1f}%</b> — <b>steady growth</b>.", "neutral")
            else:
                verdict_box("Growth is <b>declining</b> — company may be struggling.", "bad")

        # --- MOVING AVERAGES ---
        st.markdown("#### Price vs Averages")
        avg50 = fund.get("50d_avg")
        avg200 = fund.get("200d_avg")
        beta_val = fund.get("beta")
        ma1, ma2, ma3 = st.columns(3)
        ma1.metric("50-Day Avg", f"₹{avg50:,.2f}" if avg50 else "N/A",
                    f"{'Above' if cur_price > avg50 else 'Below'}" if avg50 else None)
        ma2.metric("200-Day Avg", f"₹{avg200:,.2f}" if avg200 else "N/A",
                    f"{'Above' if cur_price > avg200 else 'Below'}" if avg200 else None)
        ma3.metric("Beta", f"{beta_val:.2f}" if beta_val else "N/A")
        if beta_val:
            if beta_val < 0.8:
                verdict_box(f"Beta of <b>{beta_val:.2f}</b> — <b>defensive</b>. Less volatile than the market.", "good")
            elif beta_val <= 1.2:
                verdict_box(f"Beta of <b>{beta_val:.2f}</b> — <b>market-aligned</b>. Moves with the index.", "neutral")
            else:
                verdict_box(f"Beta of <b>{beta_val:.2f}</b> — <b>aggressive</b>. More volatile than the market.", "bad")


# ========================================================================
# TAB 9: SECTOR COMPARISON
# ========================================================================
with tab_sector:
    help_box("<b>Sector Analysis</b> compares your selected stock against peers in the same sector. "
             "See which stocks outperform, how correlated they are, and key metrics side-by-side.")

    sector = get_sector_for_stock(ticker)
    if not sector:
        st.warning(f"{ticker} is not in any predefined sector. Try a stock from the popular list.")
    else:
        peers = SECTOR_STOCKS[sector]
        st.markdown(f"### Sector: {sector}")
        st.markdown(f"**Peers:** {', '.join(p.replace('.NS','') for p in peers)}")

        with st.spinner("Loading sector data..."):
            try:
                prices_df = fetch_multiple_stocks(peers, period_years=1)
            except Exception as e:
                st.error(f"Error loading sector data: {e}")
                prices_df = pd.DataFrame()

        if not prices_df.empty:
            # --- Normalized Performance Chart ---
            st.markdown("#### Price Performance (1 Year, Indexed to 100)")
            help_box("All stocks start at 100. A stock at 130 means it gained 30%. "
                     "Compare which stock performed best over the past year.")

            normalized = (prices_df / prices_df.iloc[0]) * 100
            fig_perf = go.Figure()
            perf_colors = [C_CYAN, C_AMBER, C_PURPLE, C_GREEN, C_RED]
            for i, col in enumerate(normalized.columns):
                lw = 2.5 if col == ticker else 1.2
                fig_perf.add_trace(go.Scatter(
                    x=normalized.index, y=normalized[col],
                    name=col.replace(".NS", ""),
                    line=dict(width=lw, color=perf_colors[i % len(perf_colors)]),
                ))
            fig_perf.add_hline(y=100, line_dash="dot", line_color="rgba(150,150,150,0.3)")
            fig_perf.update_layout(height=420, **CHART_LAYOUT)
            st.plotly_chart(fig_perf, use_container_width=True)

            # --- YTD Returns Bar Chart ---
            st.markdown("#### Returns Comparison")
            returns_1y = ((prices_df.iloc[-1] / prices_df.iloc[0]) - 1) * 100
            returns_sorted = returns_1y.sort_values(ascending=True)
            bar_colors = [C_GREEN if v >= 0 else C_RED for v in returns_sorted.values]
            highlight = [C_CYAN if idx == ticker else c for idx, c in zip(returns_sorted.index, bar_colors)]

            fig_ret = go.Figure(go.Bar(
                x=returns_sorted.values,
                y=[t.replace(".NS", "") for t in returns_sorted.index],
                orientation="h",
                marker_color=highlight,
                text=[f"{v:+.1f}%" for v in returns_sorted.values],
                textposition="outside",
            ))
            fig_ret.update_layout(height=max(250, len(peers) * 60), **CHART_LAYOUT)
            st.plotly_chart(fig_ret, use_container_width=True)

            # --- Key Metrics Table ---
            st.markdown("#### Key Metrics Comparison")
            help_box("Compare valuation and profitability across sector peers. "
                     "Lower P/E may mean better value. Higher ROE means better returns on investment.")

            metrics_rows = []
            for p in peers:
                if p in prices_df.columns:
                    try:
                        p_fund = get_stock_fundamentals(p)
                        p_ret = ((prices_df[p].iloc[-1] / prices_df[p].iloc[0]) - 1) * 100
                        metrics_rows.append({
                            "Stock": p.replace(".NS", ""),
                            "Price (₹)": f"{p_fund.get('current_price', 0):,.2f}",
                            "Mkt Cap (Cr)": f"{p_fund['market_cap']/1e7:,.0f}" if p_fund.get("market_cap") else "N/A",
                            "P/E": f"{p_fund['trailing_pe']:.1f}" if p_fund.get("trailing_pe") else "N/A",
                            "ROE %": f"{p_fund['roe']*100:.1f}" if p_fund.get("roe") else "N/A",
                            "Div Yield %": f"{p_fund['dividend_yield']*100:.2f}" if p_fund.get("dividend_yield") else "N/A",
                            "1Y Return %": f"{p_ret:+.1f}",
                            "Beta": f"{p_fund['beta']:.2f}" if p_fund.get("beta") else "N/A",
                        })
                    except Exception:
                        continue

            if metrics_rows:
                st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True, hide_index=True)

            # --- Correlation Matrix ---
            st.markdown("#### Correlation Matrix")
            help_box("Shows how closely stocks move together. <b>1.0</b> = move exactly the same. "
                     "<b>0</b> = no relation. High correlation means they react to similar market forces.")

            corr_matrix = calculate_correlation_matrix(prices_df)
            corr_labels = [t.replace(".NS", "") for t in corr_matrix.columns]
            fig_corr = go.Figure(go.Heatmap(
                z=corr_matrix.values,
                x=corr_labels, y=corr_labels,
                colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                textfont=dict(size=12),
            ))
            fig_corr.update_layout(height=400, **CHART_LAYOUT)
            st.plotly_chart(fig_corr, use_container_width=True)

            # --- Risk Metrics ---
            with st.expander("Risk & Return Metrics"):
                help_box("<b>Sharpe Ratio</b> = return per unit of risk. >1 is good, >2 is excellent. "
                         "<b>Max Drawdown</b> = worst peak-to-trough drop.")
                risk_rows = []
                for p in peers:
                    if p in prices_df.columns:
                        try:
                            p_returns = prices_df[p].pct_change().dropna()
                            ratios = calculate_sharpe_ratio(p_returns)
                            risk_rows.append({
                                "Stock": p.replace(".NS", ""),
                                "Sharpe": f"{ratios['sharpe_ratio']:.2f}",
                                "Sortino": f"{ratios['sortino_ratio']:.2f}",
                                "Ann. Return %": f"{ratios['annualized_return']:.1f}",
                                "Ann. Vol %": f"{ratios['annualized_volatility']:.1f}",
                                "Max Drawdown %": f"{ratios['max_drawdown']:.1f}",
                            })
                        except Exception:
                            continue
                if risk_rows:
                    st.dataframe(pd.DataFrame(risk_rows), use_container_width=True, hide_index=True)


# ============================================================
# TAB 10: ACCURACY TRACKING
# ============================================================
with tab_accuracy:
    st.subheader("🎯 Prediction Accuracy Tracker")
    help_box("Track how accurate past AI predictions were. Predictions are logged when you run them, "
             "and validated automatically against actual prices the next day.")

    # Ticker filter
    tracked = get_tracked_tickers()

    if not tracked:
        st.markdown("""<div class="empty-state">
            <div class="es-icon">🎯</div>
            <div class="es-title">No predictions tracked yet</div>
            <div class="es-desc">Go to the <b>Predictions</b> tab and run a prediction first.<br>
            Predictions are automatically logged and validated against actual prices.</div>
        </div>""", unsafe_allow_html=True)
    else:
        acc_col1, acc_col2 = st.columns([1, 3])
        with acc_col1:
            acc_ticker_options = ["All Tickers"] + tracked
            acc_ticker = st.selectbox("Filter by ticker", acc_ticker_options, key="acc_ticker")
            filter_ticker = None if acc_ticker == "All Tickers" else acc_ticker

            if st.button("🔄 Validate Now", help="Fetch actual prices for pending predictions"):
                with st.spinner("Validating predictions..."):
                    val_result = validate_pending_predictions(filter_ticker)
                st.success(f"Validated {val_result['validated_count']} predictions")
                if val_result["errors"]:
                    for err in val_result["errors"]:
                        st.warning(err)
                st.cache_data.clear()
                st.rerun()

        # Summary metrics
        metrics = get_accuracy_metrics(filter_ticker)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Logged", metrics["total_predictions"])
        m2.metric("Validated", metrics["validated_count"])
        m3.metric("Direction Accuracy",
                  f"{metrics['direction_accuracy_pct']}%",
                  help="% of predictions where predicted direction (up/down) matched actual")
        m4.metric("Avg Error",
                  f"{metrics['avg_error_pct']}%",
                  help="Average absolute error between predicted and actual price")
        m5.metric("Within Confidence",
                  f"{metrics['within_confidence_pct']}%",
                  help="% of predictions where actual price fell within the confidence band")

        if metrics["direction_accuracy_pct"] >= 60:
            verdict_box(f"Strong prediction accuracy ({metrics['direction_accuracy_pct']}% direction correct)", "bullish")
        elif metrics["direction_accuracy_pct"] >= 50:
            verdict_box(f"Moderate accuracy ({metrics['direction_accuracy_pct']}% direction correct)", "neutral")
        elif metrics["validated_count"] > 0:
            verdict_box(f"Low accuracy ({metrics['direction_accuracy_pct']}% direction correct) — consider relearning", "bearish")

        # Prediction history table
        history_df = get_prediction_history(filter_ticker, limit=100)

        if not history_df.empty:
            st.markdown("### Prediction History")

            # Charts for validated predictions
            validated_df = history_df[history_df["actual_price"].notna()].copy()

            if not validated_df.empty:
                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    st.markdown("#### Predicted vs Actual")
                    fig_pva = go.Figure()
                    fig_pva.add_trace(go.Scatter(
                        x=validated_df["target_date"], y=validated_df["actual_price"],
                        name="Actual", line=dict(color=C_CYAN, width=2),
                    ))
                    fig_pva.add_trace(go.Scatter(
                        x=validated_df["target_date"], y=validated_df["predicted_price"],
                        name="Predicted", line=dict(color=C_AMBER, width=2, dash="dash"),
                    ))
                    # Confidence band
                    if "confidence_upper" in validated_df.columns:
                        conf_df = validated_df[validated_df["confidence_upper"].notna()]
                        if not conf_df.empty:
                            fig_pva.add_trace(go.Scatter(
                                x=conf_df["target_date"], y=conf_df["confidence_upper"],
                                line=dict(width=0), showlegend=False,
                            ))
                            fig_pva.add_trace(go.Scatter(
                                x=conf_df["target_date"], y=conf_df["confidence_lower"],
                                fill="tonexty", fillcolor="rgba(245, 158, 11, 0.1)",
                                line=dict(width=0), name="Confidence Band",
                            ))
                    fig_pva.update_layout(height=350, **CHART_LAYOUT)
                    st.plotly_chart(fig_pva, use_container_width=True)

                with chart_col2:
                    st.markdown("#### Rolling Direction Accuracy")
                    if len(validated_df) >= 5:
                        validated_df = validated_df.sort_values("target_date")
                        validated_df["rolling_acc"] = validated_df["direction_correct"].rolling(
                            window=min(10, len(validated_df)), min_periods=3
                        ).mean() * 100

                        fig_roll = go.Figure()
                        fig_roll.add_trace(go.Scatter(
                            x=validated_df["target_date"], y=validated_df["rolling_acc"],
                            name="Rolling Accuracy %", line=dict(color=C_GREEN, width=2),
                            fill="tozeroy", fillcolor="rgba(16, 185, 129, 0.1)",
                        ))
                        fig_roll.add_hline(y=50, line_dash="dash", line_color=C_RED,
                                           annotation_text="50% (random)")
                        fig_roll.update_layout(height=350, yaxis_title="Accuracy %", **CHART_LAYOUT)
                        st.plotly_chart(fig_roll, use_container_width=True)
                    else:
                        st.info("Need at least 5 validated predictions for rolling accuracy chart.")

                # Per-model comparison
                if "lstm_price" in validated_df.columns and "xgb_price" in validated_df.columns:
                    model_df = validated_df[validated_df["lstm_price"].notna() & validated_df["xgb_price"].notna()]
                    if not model_df.empty:
                        with st.expander("📊 LSTM vs XGBoost Comparison"):
                            lstm_err = (model_df["lstm_price"] - model_df["actual_price"]).abs() / model_df["actual_price"] * 100
                            xgb_err = (model_df["xgb_price"] - model_df["actual_price"]).abs() / model_df["actual_price"] * 100
                            ens_err = (model_df["predicted_price"] - model_df["actual_price"]).abs() / model_df["actual_price"] * 100

                            mc1, mc2, mc3 = st.columns(3)
                            mc1.metric("LSTM Avg Error", f"{lstm_err.mean():.2f}%")
                            mc2.metric("XGBoost Avg Error", f"{xgb_err.mean():.2f}%")
                            mc3.metric("Ensemble Avg Error", f"{ens_err.mean():.2f}%")

                            better_model = "XGBoost" if xgb_err.mean() < lstm_err.mean() else "LSTM"
                            verdict_box(f"{better_model} has been more accurate recently", "neutral")

            # Display history table
            display_df = history_df.copy()
            display_df["Status"] = display_df["actual_price"].apply(
                lambda x: "✅ Validated" if pd.notna(x) else "⏳ Pending"
            )
            display_df["Direction"] = display_df["direction_correct"].apply(
                lambda x: "✅ Correct" if x == 1 else ("❌ Wrong" if x == 0 else "—")
            )
            show_cols = ["ticker", "prediction_date", "target_date", "predicted_price",
                         "actual_price", "error_pct", "Direction", "Status"]
            available_cols = [c for c in show_cols if c in display_df.columns]
            st.dataframe(
                display_df[available_cols].rename(columns={
                    "ticker": "Ticker", "prediction_date": "Predicted On",
                    "target_date": "Target Date", "predicted_price": "Predicted ₹",
                    "actual_price": "Actual ₹", "error_pct": "Error %",
                }),
                use_container_width=True, hide_index=True,
            )

        # Relearning controls
        st.markdown("---")
        st.markdown("### 🔄 Model Relearning")
        help_box("When prediction accuracy drops, you can retrain the model using recent data. "
                 "This fine-tunes LSTM with a low learning rate and retrains XGBoost from scratch. "
                 "Ensemble weights are automatically adjusted based on which model performed better.")

        relearn_ticker = st.selectbox("Select ticker to relearn", tracked, key="relearn_ticker")

        if relearn_ticker:
            version = get_current_model_version(relearn_ticker)
            can_relearn, reason = should_relearn(relearn_ticker)

            rl1, rl2, rl3 = st.columns(3)
            rl1.metric("Model Version", version if version > 0 else "Initial")
            rl2.metric("Adaptive XGB Weight", f"{compute_adaptive_weights(relearn_ticker):.2f}")

            relearn_metrics = get_accuracy_metrics(relearn_ticker)
            rl3.metric("Direction Accuracy", f"{relearn_metrics['direction_accuracy_pct']}%")

            if can_relearn:
                verdict_box(f"Relearning recommended: {reason}", "bearish")
                if st.button(f"🔄 Relearn {relearn_ticker}", type="primary"):
                    progress = st.empty()
                    status = st.empty()
                    def relearn_progress(msg, pct):
                        progress.progress(pct, text=msg)
                    with st.spinner("Relearning..."):
                        result = relearn_models(relearn_ticker, progress_callback=relearn_progress)
                    progress.progress(1.0, text="Relearning complete!")
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success(
                            f"Model updated: v{result['old_version']} → v{result['new_version']}. "
                            f"New XGBoost weight: {result['new_xgb_weight']:.2f}"
                        )
                        st.cache_data.clear()
                        st.rerun()
            else:
                verdict_box(f"No relearning needed: {reason}", "neutral")

            # Manual relearn override
            with st.expander("Force relearn (override criteria)"):
                if st.button(f"Force relearn {relearn_ticker}"):
                    progress = st.empty()
                    def force_progress(msg, pct):
                        progress.progress(pct, text=msg)
                    with st.spinner("Relearning..."):
                        result = relearn_models(relearn_ticker, progress_callback=force_progress)
                    progress.progress(1.0, text="Done!")
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success(
                            f"Model updated: v{result['old_version']} → v{result['new_version']}. "
                            f"New XGBoost weight: {result['new_xgb_weight']:.2f}"
                        )
                        st.cache_data.clear()
                        st.rerun()


# ============================================================
# FOOTER
# ============================================================
st.markdown("""<div class="footer-card">
    <div style="display: flex; align-items: center; justify-content: center; gap: 8px; flex-wrap: wrap;">
        <span style="font-size: 16px;">⚠️</span>
        <span><b>Disclaimer:</b> Educational purposes only. Not financial advice. Markets are inherently risky — always do your own research before investing.</span>
    </div>
    <div style="margin-top: 8px; font-size: 11px; color: #444460;">
        Built with Streamlit &bull; LSTM + XGBoost AI Models &bull; Data from Yahoo Finance
    </div>
</div>""", unsafe_allow_html=True)
