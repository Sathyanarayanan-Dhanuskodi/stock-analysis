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
from src.charges import calc_angel_one_charges
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

st.set_page_config(page_title="Indian Stock Predictor",
                   page_icon="📈", layout="wide")

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
# PAGE STATE
# ============================================================
if "page" not in st.session_state:
    st.session_state["page"] = "home"
if "analysis_ticker" not in st.session_state:
    st.session_state["analysis_ticker"] = None


# ============================================================
# MODERN GRADIENT THEME CSS
# ============================================================
st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Sora:wght@300;400;500;600;700&display=swap');

    :root {
        --bg: #111318;
        --bg-raised: #16181e;
        --bg-hover: #1b1e25;
        --bg-active: #1f222b;
        --border: rgba(255, 255, 255, 0.05);
        --border-hover: rgba(255, 255, 255, 0.09);
        --text: #e8e4de;
        --text-2: rgba(255, 255, 255, 0.45);
        --text-3: rgba(255, 255, 255, 0.22);
        --accent: #d4a054;
        --accent-dim: rgba(212, 160, 84, 0.12);
        --green: #5eb88a;
        --green-dim: rgba(94, 184, 138, 0.1);
        --red: #d45d5d;
        --red-dim: rgba(212, 93, 93, 0.1);
        --blue: #c9b89c;
        --mono: 'JetBrains Mono', monospace;
        --sans: 'Sora', sans-serif;
    }

    /* === BASE — apply Sora globally === */
    .block-container { max-width: 1440px; }
    html, body, [class*="css"] { font-family: var(--sans) !important; font-size: 14px !important; }
    .stApp { font-family: var(--sans) !important; font-size: 14px; }
    .stMarkdown p, .stMarkdown li { font-size: 13px; }
    h4 { font-size: 15px !important; }
    h3 { font-size: 17px !important; }
    h2 { font-size: 19px !important; }
    .stMarkdown, .stMarkdown *, .stTabs,
    button, input, select, textarea, label,
    p, li, div, h1, h2, h3, h4, h5, h6, a {
        font-family: var(--sans) !important;
    }
    /* Restore Material Symbols for Streamlit icons (expander arrows, etc.) */
    [data-testid="stExpanderToggleIcon"],
    [data-testid="stExpanderToggleIcon"] * {
        font-family: 'Material Symbols Rounded' !important;
    }
    h4, h3, h2 { color: var(--text) !important; font-weight: 700 !important; letter-spacing: -0.3px; }
    p, span, li, div { color: var(--text); }

    /* === MAIN BACKGROUND === */
    .stApp { background: var(--bg) !important; }
    header[data-testid="stHeader"] { background: var(--bg) !important; }

    /* === ACTION CARDS === */
    .action-card {
        padding: 20px 24px; border-radius: 10px; text-align: center;
        font-size: 20px; font-weight: 700; color: #fff; margin: 10px 0;
        border: none; letter-spacing: 0.2px;
        position: relative; overflow: hidden;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .action-card:hover { transform: translateY(-1px); }
    .action-card span { display: block; font-size: 13px; font-weight: 500; opacity: 0.9; margin-top: 6px; line-height: 1.5; }
    .action-buy, .action-long, .action-up   { background: linear-gradient(135deg, var(--green-dim), rgba(94,184,138,0.05)); border: 1px solid rgba(94,184,138,0.15); color: var(--green); }
    .action-sell, .action-short, .action-down { background: linear-gradient(135deg, var(--red-dim), rgba(212,93,93,0.05)); border: 1px solid rgba(212,93,93,0.15); color: var(--red); }
    .action-hold, .action-sideways           { background: linear-gradient(135deg, var(--accent-dim), rgba(212,160,84,0.05)); border: 1px solid rgba(212,160,84,0.15); color: var(--accent); }
    .action-notrade                          { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); color: var(--text-2); }

    /* === BADGE === */
    .badge-best {
        display: inline-block; background: linear-gradient(135deg, var(--accent), #c4883c);
        color: #1a1610; font-size: 11px; font-weight: 800; text-transform: uppercase;
        letter-spacing: 0.8px; padding: 4px 12px; border-radius: 20px;
        margin-left: 8px; vertical-align: middle; line-height: 1;
    }
    .badge-rank {
        display: inline-block; background: rgba(255,255,255,0.06);
        color: var(--text-2); font-size: 11px; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.6px; padding: 4px 12px; border-radius: 20px;
        margin-left: 8px; vertical-align: middle; line-height: 1;
        border: 1px solid rgba(255,255,255,0.06);
    }

    /* === PROFIT / LOSS TEXT === */
    .profit, .setup-card .profit, .setup-card .profit b { color: var(--green) !important; font-weight: 700; }
    .loss, .setup-card .loss, .setup-card .loss b { color: var(--red) !important; font-weight: 700; }
    .charges-dim, .setup-card .charges-dim { color: var(--text-3) !important; font-size: 13px; }

    /* === HELP BOX === */
    .help-box {
        background: rgba(212,160,84,0.06); border-left: 3px solid var(--accent);
        padding: 12px 16px; border-radius: 0 10px 10px 0; margin: 18px 0;
        font-size: 13px; color: var(--text-2); line-height: 1.7;
    }
    .help-box b { color: var(--text); }

    /* === SETUP CARD === */
    .setup-card {
        background: var(--bg-raised); padding: 18px 20px; border-radius: 10px;
        border: 1px solid var(--border); margin: 8px 0;
        color: var(--text-2); line-height: 1.8; font-size: 14px;
        transition: border-color 0.2s ease;
    }
    .setup-card:hover { border-color: var(--border-hover); }
    .setup-card b { color: var(--text); font-size: 14px; }

    /* === PAPER TRADE CONTAINER (st.container with border) === */
    .paper-trade-box [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--bg-raised) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }
    .paper-trade-box [data-testid="stVerticalBlockBorderWrapper"] > div {
        background: var(--bg-raised) !important;
    }

    /* === GREEN PAPER TRADE BUTTON === */
    .paper-trade-btn + div button[data-testid="stBaseButton-primary"],
    .paper-trade-btn ~ div button[data-testid="stBaseButton-primary"],
    .paper-trade-btn button,
    .paper-trade-btn + div button {
        background: var(--green) !important;
        color: #000 !important;
        border: none !important;
        font-weight: 700 !important;
    }
    .paper-trade-btn + div button[data-testid="stBaseButton-primary"]:hover,
    .paper-trade-btn ~ div button[data-testid="stBaseButton-primary"]:hover,
    .paper-trade-btn button:hover,
    .paper-trade-btn + div button:hover {
        box-shadow: 0 4px 16px rgba(94,184,138,0.25) !important;
        background: #4a9e74 !important;
    }

    /* === VERDICT BOX === */
    .verdict-box {
        padding: 12px 16px; border-radius: 8px; margin: 6px 0;
        font-size: 13px; line-height: 1.7;
    }
    .verdict-box b { color: var(--text); }
    .good    { background: rgba(94,184,138,0.08); border: 1px solid rgba(94,184,138,0.12); border-left: 3px solid var(--green); color: rgba(94,184,138,0.9); }
    .bad     { background: rgba(212,93,93,0.08); border: 1px solid rgba(212,93,93,0.12); border-left: 3px solid var(--red); color: rgba(212,93,93,0.9); }
    .neutral { background: rgba(212,160,84,0.08); border: 1px solid rgba(212,160,84,0.12); border-left: 3px solid var(--accent); color: rgba(212,160,84,0.9); }
    .bullish { background: rgba(94,184,138,0.08); border: 1px solid rgba(94,184,138,0.12); border-left: 3px solid var(--green); color: rgba(94,184,138,0.9); }
    .bearish { background: rgba(212,93,93,0.08); border: 1px solid rgba(212,93,93,0.12); border-left: 3px solid var(--red); color: rgba(212,93,93,0.9); }

    /* === CHECKLIST === */
    .checklist {
        background: var(--bg-raised); padding: 14px 18px; border-radius: 10px; margin: 8px 0;
        color: var(--text-2); line-height: 1.9; font-size: 13px; border: 1px solid var(--border);
    }
    .checklist b { color: var(--text); font-size: 14px; }

    /* === METRIC CARDS === */
    [data-testid="stMetric"] {
        background: var(--bg-raised); border: 1px solid var(--border); border-radius: 10px; padding: 14px;
        transition: border-color 0.2s ease;
    }
    [data-testid="stMetric"]:hover { border-color: var(--border-hover); }
    [data-testid="stMetricLabel"] { color: var(--text-3) !important; font-size: 9px !important; text-transform: uppercase; letter-spacing: 1.2px; font-weight: 600 !important; }
    [data-testid="stMetricValue"] { color: var(--text) !important; font-weight: 700 !important; font-size: 1.2rem !important; font-family: var(--mono) !important; }
    [data-testid="stMetricDelta"] > div { font-size: 12px !important; }

    /* === TABS === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px; background: var(--bg-raised) !important; border-radius: 10px; padding: 4px;
        border: 1px solid var(--border) !important;
        overflow-x: auto;
    }
    .stTabs [data-baseweb="tab-list"] button {
        border-radius: 8px !important; padding: 8px 16px !important;
        color: #b8975a !important; font-weight: 500 !important; font-size: 13px !important;
        transition: all 0.2s ease; white-space: nowrap;
        border: none !important; background: transparent !important;
        font-family: var(--sans) !important;
    }
    .stTabs [data-baseweb="tab-list"] button:hover {
        color: var(--text) !important; background: var(--bg-hover) !important;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background: rgba(212,160,84,0.1) !important;
        color: var(--accent) !important; font-weight: 600 !important;
        border: 1px solid rgba(212,160,84,0.4) !important;
        box-shadow: 0 0 8px rgba(212,160,84,0.08);
    }
    /* Hide the tab highlight/indicator bar */
    .stTabs [data-baseweb="tab-highlight"] { display: none !important; }
    .stTabs [data-baseweb="tab-border"] { display: none !important; }

    /* === BUTTONS === */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--green), #4a9e73) !important; color: #1a1610 !important;
        border: none !important; border-radius: 8px !important; font-weight: 700 !important;
        font-size: 12px !important; padding: 9px 20px !important;
        transition: all 0.25s ease; letter-spacing: 0.02em;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 4px 16px rgba(94,184,138,0.25) !important;
        transform: translateY(-1px);
    }
    .stButton > button:not([kind="primary"]) {
        border-radius: 8px !important;
        border: 1px solid var(--border) !important;
        background: var(--bg-raised) !important;
        color: var(--text-2) !important;
        transition: all 0.2s ease;
    }
    .stButton > button:not([kind="primary"]):hover {
        border-color: var(--border-hover) !important;
        color: var(--text) !important;
        background: var(--bg-hover) !important;
    }

    /* === SCREENER PICK CARD === */
    .screener-pick {
        background: var(--bg-raised); padding: 18px; border-radius: 10px;
        border: 1px solid var(--border); margin: 8px 0;
        color: var(--text-2); line-height: 1.8; font-size: 13px;
        transition: border-color 0.2s ease;
    }
    .screener-pick:hover { border-color: var(--border-hover); }
    .screener-pick b { color: var(--text); }
    .screener-pick .price { font-size: 20px; font-weight: 700; color: var(--accent); font-family: var(--mono); }
    .screener-pick .detail { font-size: 12px; color: var(--text-3); }

    /* === SECTION DIVIDER === */
    .section-label {
        font-size: 9px; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px;
        color: var(--text-3); margin: 18px 0 10px 0; padding-bottom: 8px;
        border-bottom: 1px solid var(--border);
    }

    /* === STOCK HEADER === */
    .stock-header {
        background: var(--bg-raised); border: 1px solid var(--border); border-radius: 10px;
        padding: 20px 24px; margin-bottom: 16px;
    }

    /* === QUICK STATS ROW === */
    .quick-stats {
        display: flex; gap: 10px; flex-wrap: wrap;
    }
    .quick-stat {
        background: var(--bg-raised); border: 1px solid var(--border); border-radius: 10px;
        padding: 14px 16px; flex: 1; min-width: 120px; text-align: center;
        transition: border-color 0.2s ease;
    }
    .quick-stat:hover { border-color: var(--border-hover); }
    .quick-stat .qs-label { font-size: 9px; color: var(--text-3); text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
    .quick-stat .qs-value { font-size: 20px; font-weight: 700; color: var(--text); margin-top: 4px; font-family: var(--mono); }
    .quick-stat .qs-delta { font-size: 11px; font-weight: 500; margin-top: 2px; }

    /* === DATAFRAME STYLING === */
    .stDataFrame { font-size: 12px !important; }
    .stDataFrame [data-testid="stDataFrameResizable"] {
        border-radius: 10px; overflow: hidden;
        border: 1px solid var(--border) !important;
    }

    /* === FOOTER === */
    .footer-card {
        background: var(--bg-raised); border: 1px solid var(--border); border-radius: 10px;
        padding: 14px 18px; margin-top: 20px; text-align: center;
        font-size: 12px; color: var(--text-3); line-height: 1.6;
    }
    .footer-card b { color: var(--accent); }

    /* === PROMPT BOX === */
    .prompt-box {
        background: var(--bg); border: 1px solid var(--border); border-radius: 10px;
        padding: 18px; margin: 10px 0; font-family: var(--mono);
        font-size: 12px; color: var(--text-2); line-height: 1.7;
        white-space: pre-wrap; word-wrap: break-word;
        max-height: 400px; overflow-y: auto;
    }
    .prompt-box b { color: var(--accent); }

    /* === EXPANDER === */
    .streamlit-expanderHeader {
        font-size: 13px !important; font-weight: 600 !important; color: var(--text-2) !important;
        border-radius: 8px !important;
    }
    .streamlit-expanderContent { border-color: var(--border) !important; }
    details { background: var(--bg-raised) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }

    /* === SIDEBAR === */
    [data-testid="stSidebar"] {
        background: var(--bg) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text); }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stNumberInput label {
        font-size: 12px !important; font-weight: 600 !important; color: var(--text-2) !important;
    }
    [data-testid="stSidebar"] .stDivider { border-color: var(--border) !important; }

    /* Sidebar inputs */
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] select,
    [data-testid="stSidebar"] [data-baseweb="select"] {
        background: var(--bg-raised) !important; border-color: var(--border) !important;
        color: var(--text) !important;
    }
    [data-testid="stSidebar"] input:focus {
        border-color: var(--accent) !important;
    }

    /* Sidebar slider */
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {
        background: var(--accent) !important;
    }

    /* Slider track fill — override Streamlit blue */
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background: var(--accent) !important;
    }
    .stSlider [data-baseweb="slider"] div[data-testid="stThumbValue"] {
        color: var(--accent) !important;
    }
    .stSlider div[data-baseweb="slider"] > div:first-child > div {
        background: var(--accent) !important;
    }
    .stSlider div[data-baseweb="slider"] > div > div[role="progressbar"] {
        background-color: var(--accent) !important;
    }
    /* All possible slider track selectors */
    .stSlider [data-testid="stSliderTrack"] > div:first-child,
    .stSlider div[data-baseweb="slider"] div[style*="background-color: rgb"] {
        background-color: var(--accent) !important;
        background: var(--accent) !important;
    }

    /* Primary button — force green over Streamlit default */
    .stButton button[kind="primary"],
    .stButton button[data-testid="stBaseButton-primary"],
    button[data-testid="stBaseButton-primary"] {
        background-color: var(--green) !important;
        background: linear-gradient(135deg, var(--green), #4a9e73) !important;
        color: #000 !important;
        border: none !important;
    }
    button[data-testid="stBaseButton-primary"]:hover {
        box-shadow: 0 4px 16px rgba(94,184,138,0.25) !important;
        background: linear-gradient(135deg, #4a9e73, var(--green)) !important;
    }
    button[data-testid="stBaseButton-primary"]:active,
    button[data-testid="stBaseButton-primary"]:focus {
        background: linear-gradient(135deg, var(--accent), #c4883c) !important;
        color: #000 !important;
    }

    /* === EMPTY STATE === */
    .empty-state { text-align: center; padding: 36px 24px; color: var(--text-3); }
    .empty-state .es-icon { font-size: 36px; margin-bottom: 10px; opacity: 0.5; }
    .empty-state .es-title { font-size: 15px; font-weight: 600; color: var(--text-2); margin-bottom: 5px; }
    .empty-state .es-desc { font-size: 13px; color: var(--text-3); line-height: 1.6; }

    /* === DIVIDERS === */
    hr { border-color: var(--border) !important; opacity: 1; margin: 16px 0 !important; }

    /* === SCROLLBAR === */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.15); }

    /* === NUMBER INPUT / SELECT POLISH === */
    .stNumberInput input, .stTextInput input {
        border-radius: 8px !important;
        background: var(--bg-raised) !important;
        border-color: var(--border) !important;
        color: var(--text) !important;
        font-family: var(--mono) !important;
        transition: border-color 0.2s ease !important;
    }
    .stNumberInput input:focus, .stTextInput input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px rgba(212,160,84,0.15) !important;
    }

    /* === SELECT BOXES === */
    [data-baseweb="select"] > div {
        background: var(--bg-raised) !important;
        border-color: var(--border) !important;
        color: var(--text) !important;
        border-radius: 8px !important;
    }
    [data-baseweb="select"] > div:focus-within {
        border-color: var(--accent) !important;
    }

    /* === PROGRESS BAR === */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--accent), #c4883c) !important;
        border-radius: 3px;
    }
    .stProgress > div > div { background: rgba(255,255,255,0.04) !important; }

    /* === SCREENER ACTION BUTTONS === */
    .scr-btn {
        display: inline-flex; align-items: center; justify-content: center; gap: 4px;
        padding: 6px 12px; border-radius: 6px; font-size: 11px; font-weight: 600;
        text-decoration: none; cursor: pointer; border: none; transition: all 0.2s;
        flex: 1; text-align: center;
    }
    .scr-btn-trade {
        background: var(--green-dim); color: var(--green); border: 1px solid rgba(94,184,138,0.2);
    }
    .scr-btn-trade:hover { background: rgba(94,184,138,0.18); }
    .scr-btn-analyse {
        background: var(--accent-dim); color: var(--accent); border: 1px solid rgba(212,160,84,0.2);
    }
    .scr-btn-analyse:hover { background: rgba(212,160,84,0.15); color: var(--accent); text-decoration: none; }

    /* === SCREENER CARD-BUTTON SIDE LAYOUT === */
    .scr-btn-col { display: flex; flex-direction: column; gap: 6px; justify-content: center; height: 100%; padding-top: 4px; }
    .scr-btn-col .stElementContainer,
    .scr-btn-col .stMarkdown,
    .scr-btn-col .stMarkdown p,
    .scr-btn-col .stButton { width: 100% !important; }
    .scr-btn-col .stButton > button {
        font-size: 12px !important; padding: 8px 0 !important; border-radius: 6px !important;
        min-height: 0 !important; height: auto !important; width: 100% !important;
    }
    .scr-btn-col a.scr-btn {
        width: 100% !important; box-sizing: border-box; display: block; text-align: center;
        padding: 8px 0 !important; font-size: 12px !important;
    }

    /* === TAB CONTENT SPACING === */
    .stTabs [data-baseweb="tab-panel"] { padding-top: 10px; }

    /* === POPOVER / DROPDOWN === */
    [data-baseweb="popover"] > div { background: var(--bg-hover) !important; border: 1px solid var(--border-hover) !important; }
    [data-baseweb="menu"] { background: var(--bg-hover) !important; }
    [data-baseweb="menu"] li { color: var(--text) !important; }
    [data-baseweb="menu"] li:hover { background: var(--bg-active) !important; }

    /* === CHECKBOX / RADIO === */
    .stCheckbox label span { color: var(--text-2) !important; }
    .stRadio label span { color: var(--text-2) !important; }

    /* === TEXT AREA === */
    .stTextArea textarea {
        background: #16181e !important; border-color: rgba(255,255,255,0.05) !important;
        color: #e8e4de !important; border-radius: 8px !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* === ALERTS === */
    .stAlert { border-radius: 8px !important; }
</style>""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

# Handle query param stock — auto-route to analysis page
_qp_stock = st.query_params.get("stock")
if _qp_stock:
    st.session_state["page"] = "analysis"
    _qp_upper = _qp_stock.strip().upper()
    st.session_state["analysis_ticker"] = _qp_upper

with st.sidebar:
    st.markdown("""<div style="display: flex; align-items: center; gap: 10px; padding: 4px 0 16px 0;">
        <div style="width: 30px; height: 30px; border-radius: 8px;
            background: linear-gradient(135deg, #d4a054, #c4883c);
            display: flex; align-items: center; justify-content: center;
            font-size: 12px; font-weight: 800; color: #1a1610;
            font-family: 'JetBrains Mono', monospace;">SP</div>
        <div>
            <div style="font-size: 15px; font-weight: 700; color: #e8e4de; letter-spacing: -0.02em;">StockPredictor</div>
            <div style="font-size: 10px; color: rgba(255,255,255,0.22); font-family: 'JetBrains Mono', monospace; letter-spacing: 0.04em;">AI Analysis</div>
        </div>
    </div>""", unsafe_allow_html=True)

    if st.session_state["page"] == "analysis":
        if st.button("\u2190 Home", key="sidebar_home_btn", use_container_width=True):
            st.session_state["page"] = "home"
            st.session_state["analysis_ticker"] = None
            st.query_params.clear()
            st.rerun()

    st.markdown("##### Select Stock")
    stock_options = {f"{name} ({ticker})": ticker for ticker,
                     name in POPULAR_INDIAN_STOCKS.items()}
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

    if st.button("Analyze Stock", key="sidebar_analyze_btn", type="primary", use_container_width=True):
        st.session_state["page"] = "analysis"
        st.session_state["analysis_ticker"] = ticker
        st.rerun()

    if st.session_state["page"] == "analysis":
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
            capital = st.number_input("Capital (\u20b9)", 10000, 10000000, 100000, step=10000,
                                      help="How much money you plan to invest")
            max_risk_pct = st.slider("Max risk per trade (%)", 0.5, 5.0, 2.0, 0.5,
                                     help="Maximum % of capital to risk. 2% is safe.")
            sr_lookback = st.slider("Analysis lookback (days)", 30, 180, 90,
                                    help="How far back for support/resistance levels")
            pivot_method = st.selectbox(
                "Pivot calculation", ["standard", "fibonacci", "camarilla", "woodie"])

        with st.expander("Paper Trading Funds", expanded=False):
            st.caption(
                "Set capital for paper trading. Scalp trades get 5\u00d7 leverage (buy \u20b95L with \u20b91L).")
            _cur_swing_fund = get_fund_balance("swing")

            _fund_input = st.number_input(
                "Trading Capital (\u20b9)", 0, 50000000,
                int(_cur_swing_fund["initial_capital"]),
                step=50000, key="sidebar_trading_fund",
                help="Capital for paper trading. Swing uses full amount, scalp gets 5\u00d7 leverage.",
            )
            if st.button("Update Funds", key="sidebar_update_funds", use_container_width=True):
                set_fund_capital("swing", _fund_input)
                set_fund_capital("scalp", _fund_input)
                st.success("Funds updated!")
                st.rerun()

        st.divider()
        st.markdown("""<div style="background: rgba(212,160,84,0.08); border: 1px solid rgba(212,160,84,0.12);
            border-radius: 8px; padding: 10px 12px; font-size: 11px; color: rgba(255,255,255,0.45); line-height: 1.6;">
            \U0001f4a1 Start with the <b style="color: #e8e4de;">Chart</b> tab to see
            the stock's current health, then check <b style="color: #d4a054;">Swing</b> or
            <b style="color: #d4a054;">Scalping</b> for trading signals.
        </div>""", unsafe_allow_html=True)

    if st.session_state["page"] == "home":
        st.divider()
        st.markdown("""<div style="background: rgba(212,160,84,0.08); border: 1px solid rgba(212,160,84,0.12);
            border-radius: 8px; padding: 10px 12px; font-size: 11px; color: rgba(255,255,255,0.45); line-height: 1.6;">
            \U0001f4a1 Select a stock above and click <b style="color: #d4a054;">Analyze Stock</b> to view
            charts, predictions, swing & scalp signals, fundamentals, and more.
        </div>""", unsafe_allow_html=True)

# ============================================================
# SCORE HELPERS (display-only, does not affect scan/sort)
# ============================================================
def _safe_num(val, default=0):
    """Return val as float if it's a valid number, else default."""
    import math
    try:
        f = float(val)
        return default if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return default


def compute_tech_score(latest_row, signal_confidence: float) -> int:
    """0-100 technical score from indicators."""
    s = 0
    s += _safe_num(signal_confidence) * 30
    rsi = _safe_num(latest_row.get("RSI", 50) if hasattr(latest_row, 'get') else 50, 50)
    if 40 <= rsi <= 60:
        s += 15
    elif 30 <= rsi < 40 or 60 < rsi <= 70:
        s += 10
    elif rsi < 30:
        s += 12
    adx = _safe_num(latest_row.get("ADX", 0) if hasattr(latest_row, 'get') else 0)
    s += min(15, adx / 50 * 15)
    macd = _safe_num(latest_row.get("MACD", 0) if hasattr(latest_row, 'get') else 0)
    macd_sig = _safe_num(latest_row.get("MACD_Signal", 0) if hasattr(latest_row, 'get') else 0)
    if macd > macd_sig:
        s += 10
    close = _safe_num(latest_row.get("Close", 0) if hasattr(latest_row, 'get') else 0)
    for ma in ["SMA_20", "SMA_50", "SMA_200"]:
        val = _safe_num(latest_row.get(ma, 0) if hasattr(latest_row, 'get') else 0)
        if val and close > val:
            s += 5
    vr = _safe_num(latest_row.get("Volume_Ratio", 1) if hasattr(latest_row, 'get') else 1, 1)
    s += min(10, max(0, (vr - 0.5) / 1.5) * 10)
    bb_u = _safe_num(latest_row.get("BB_Upper", 0) if hasattr(latest_row, 'get') else 0)
    bb_l = _safe_num(latest_row.get("BB_Lower", 0) if hasattr(latest_row, 'get') else 0)
    if bb_u > bb_l > 0:
        bb_pos = (close - bb_l) / (bb_u - bb_l)
        s += max(0, (1 - bb_pos)) * 5
    return max(0, min(100, int(round(s))))


def compute_fundamental_score(fund: dict) -> int:
    """0-100 fundamental score from key ratios."""
    if not fund:
        return 0
    s = 0
    pe = _safe_num(fund.get("trailing_pe")) or _safe_num(fund.get("forward_pe"))
    if pe and pe > 0:
        if pe < 15:
            s += 15
        elif pe < 25:
            s += 12
        elif pe < 40:
            s += 6
        else:
            s += 2
    roe = _safe_num(fund.get("roe"))
    if roe > 0:
        s += min(15, roe * 60)
    pm = _safe_num(fund.get("profit_margin"))
    if pm > 0:
        s += min(15, pm * 75)
    de_raw = fund.get("debt_to_equity")
    de = _safe_num(de_raw) if de_raw is not None else None
    if de is not None:
        if de < 0.5:
            s += 15
        elif de < 1.0:
            s += 10
        elif de < 2.0:
            s += 5
    else:
        s += 8
    rg = _safe_num(fund.get("revenue_growth"))
    if rg > 0.20:
        s += 15
    elif rg > 0.10:
        s += 12
    elif rg > 0:
        s += 7
    eg = _safe_num(fund.get("earnings_growth"))
    if eg > 0.20:
        s += 15
    elif eg > 0.10:
        s += 12
    elif eg > 0:
        s += 7
    cr = _safe_num(fund.get("current_ratio"))
    if cr > 0:
        if cr > 1.5:
            s += 10
        elif cr > 1.0:
            s += 6
        elif cr > 0.5:
            s += 3
    return max(0, min(100, int(round(s))))


# ============================================================
# FETCH DATA (analysis page only)
# ============================================================
if st.session_state["page"] == "analysis":
    if st.session_state.get("analysis_ticker"):
        ticker = st.session_state["analysis_ticker"]

    try:
        with st.spinner("Fetching stock data..."):
            raw_df = fetch_stock_data(ticker)
            df = add_technical_indicators(raw_df)
            info = get_stock_info(ticker)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

    # ============================================================
    # PRECOMPUTE — stock banner + quick stats data (used in analysis tabs)
    # ============================================================
    price_change = raw_df["Close"].iloc[-1] - raw_df["Close"].iloc[-2]
    pct_change = (price_change / raw_df["Close"].iloc[-2]) * 100
    change_arrow = "▲" if pct_change >= 0 else "▼"
    mcap_str = f"₹{info['market_cap'] / 1e7:,.0f} Cr" if info.get(
        "market_cap") else "N/A"

    # 52-week range for context
    w52_high = info.get("year_high", raw_df["High"].tail(252).max())
    w52_low = info.get("year_low", raw_df["Low"].tail(252).min())
    cur_price = info.get("current_price") or raw_df["Close"].iloc[-1]
    w52_pct = ((cur_price - w52_low) / (w52_high - w52_low)
               * 100) if w52_high > w52_low else 50

    # Today's OHLC range
    today_open = raw_df["Open"].iloc[-1]
    today_high = raw_df["High"].iloc[-1]
    today_low = raw_df["Low"].iloc[-1]
    today_close = raw_df["Close"].iloc[-1]
    today_range = today_high - today_low if today_high > today_low else 1
    today_open_pct = ((today_open - today_low) / today_range) * 100
    today_close_pct = ((today_close - today_low) / today_range) * 100

    # 5-day trend mini indicator
    recent_5d = raw_df["Close"].tail(5)
    trend_5d_pct = ((recent_5d.iloc[-1] - recent_5d.iloc[0]) /
                    recent_5d.iloc[0] * 100) if len(recent_5d) >= 5 else 0
    trend_5d_arrow = "▲" if trend_5d_pct >= 0 else "▼"

    # Volume context
    avg_vol = raw_df["Volume"].tail(20).mean()
    today_vol = raw_df["Volume"].iloc[-1]
    vol_ratio_banner = today_vol / avg_vol if avg_vol > 0 else 1

    change_color_light = "#5eb88a" if pct_change >= 0 else "#d45d5d"
    change_bg_light = "rgba(94,184,138,0.1)" if pct_change >= 0 else "rgba(212,93,93,0.1)"
    trend_5d_color_light = "#5eb88a" if trend_5d_pct >= 0 else "#d45d5d"

    # Quick stats data
    latest_row = df.iloc[-1]
    qs_rsi = latest_row.get("RSI", 50)
    qs_rsi_color = "#d45d5d" if qs_rsi > 70 else "#5eb88a" if qs_rsi < 30 else "#d4a054"
    qs_rsi_label = "Overbought" if qs_rsi > 70 else "Oversold" if qs_rsi < 30 else "Neutral"
    qs_macd = latest_row.get("MACD", 0)
    qs_macd_sig = latest_row.get("MACD_Signal", 0)
    qs_macd_color = "#5eb88a" if qs_macd > qs_macd_sig else "#d45d5d"
    qs_macd_label = "Bullish" if qs_macd > qs_macd_sig else "Bearish"
    qs_adx = latest_row.get("ADX", 0)
    qs_adx_label = "Strong" if qs_adx > 25 else "Weak"
    qs_vol = latest_row.get("Volume_Ratio", 1)
    qs_vol_color = "#5eb88a" if qs_vol > 1.2 else "#d45d5d" if qs_vol < 0.7 else "#9b8ec4"

    # Display-only scores for stock header
    try:
        _hdr_signal = generate_swing_signals(df)
        _hdr_tech = compute_tech_score(latest_row, _hdr_signal.confidence)
    except Exception:
        _hdr_tech = 0
    try:
        _hdr_fund_data = get_stock_fundamentals(ticker)
        _hdr_fund = compute_fundamental_score(_hdr_fund_data)
    except Exception:
        _hdr_fund = 0
    _hdr_tech_color = "#5eb88a" if _hdr_tech >= 60 else "#d4a054" if _hdr_tech >= 40 else "#d45d5d"
    _hdr_fund_color = "#5eb88a" if _hdr_fund >= 60 else "#d4a054" if _hdr_fund >= 40 else "#d45d5d"
    _hdr_tech_label = "Strong" if _hdr_tech >= 60 else "Moderate" if _hdr_tech >= 40 else "Weak"
    _hdr_fund_label = "Strong" if _hdr_fund >= 60 else "Moderate" if _hdr_fund >= 40 else "Weak"

    def _render_stock_header():
        """Render stock banner + indicators row inside any tab."""
        st.markdown(f"""<div class="stock-header">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 16px;">
            <div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 18px; font-weight: 700; color: #e8e4de;">{info['name']}</span>
                    <span style="font-size: 11px; color: rgba(255,255,255,0.22); font-weight: 600; font-family: 'JetBrains Mono', monospace;">{ticker}</span>
                    <span style="background: rgba(212,160,84,0.1); border: 1px solid rgba(212,160,84,0.15); color: #d4a054; font-size: 10px; font-weight: 700;
                           padding: 2px 8px; border-radius: 6px; letter-spacing: 0.04em;">{info['sector']}</span>
                </div>
                <div style="display: flex; align-items: center; gap: 10px; margin-top: 4px; flex-wrap: wrap;">
                    <span style="font-size: 22px; font-weight: 700; color: #e8e4de; font-family: 'JetBrains Mono', monospace;">₹{cur_price:,.2f}</span>
                    <span style="background: {change_bg_light}; padding: 3px 10px; border-radius: 6px;">
                        <span style="font-size: 12px; font-weight: 600; color: {change_color_light}; font-family: 'JetBrains Mono', monospace;">
                            {change_arrow} ₹{abs(price_change):,.2f} ({pct_change:+.2f}%)</span>
                    </span>
                    <span style="background: rgba(212,160,84,0.1); border: 1px solid rgba(212,160,84,0.15); padding: 3px 10px; border-radius: 6px;">
                        <span style="font-size: 10px; color: rgba(255,255,255,0.4); font-weight: 600;">O </span>
                        <span style="font-size: 12px; font-weight: 700; color: #d4a054; font-family: 'JetBrains Mono', monospace;">₹{today_open:,.2f}</span>
                    </span>
                    <span style="background: rgba(232,228,222,0.08); border: 1px solid rgba(232,228,222,0.12); padding: 3px 10px; border-radius: 6px;">
                        <span style="font-size: 10px; color: rgba(255,255,255,0.4); font-weight: 600;">C </span>
                        <span style="font-size: 12px; font-weight: 700; color: #e8e4de; font-family: 'JetBrains Mono', monospace;">₹{today_close:,.2f}</span>
                    </span>
                </div>
            </div>
            <div style="display: flex; gap: 20px; align-items: center; flex-wrap: wrap;">
                <div style="text-align: center;">
                    <div style="font-size: 9px; color: rgba(255,255,255,0.22); text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">Market Cap</div>
                    <div style="font-size: 14px; font-weight: 700; color: #e8e4de; margin-top: 3px; font-family: 'JetBrains Mono', monospace;">{mcap_str}</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 9px; color: rgba(255,255,255,0.22); text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">5-Day</div>
                    <div style="font-size: 13px; font-weight: 700; color: {trend_5d_color_light}; margin-top: 3px; font-family: 'JetBrains Mono', monospace;">{trend_5d_arrow} {trend_5d_pct:+.1f}%</div>
                </div>
            </div>
        </div>
        <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.04);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-size: 9px; color: rgba(255,255,255,0.22); text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">52-Week Range</span>
                <span style="font-size: 10px; color: rgba(255,255,255,0.35); font-family: 'JetBrains Mono', monospace;">{w52_pct:.0f}% from low</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 11px; font-weight: 600; color: #d45d5d; font-family: 'JetBrains Mono', monospace; white-space: nowrap;">₹{w52_low:,.0f}</span>
                <div style="flex: 1; position: relative; height: 6px; background: rgba(255,255,255,0.06); border-radius: 3px; overflow: visible;">
                    <div style="width: {max(min(w52_pct, 100), 0):.1f}%; height: 100%; border-radius: 3px;
                                background: linear-gradient(90deg, #d45d5d, #d4a054, #5eb88a);"></div>
                    <div style="position: absolute; top: -3px; left: calc({max(min(w52_pct, 100), 0):.1f}% - 6px); width: 12px; height: 12px;
                                background: #e8e4de; border-radius: 50%; border: 2px solid #0e0e1c; box-shadow: 0 0 4px rgba(232,228,222,0.3);"></div>
                </div>
                <span style="font-size: 11px; font-weight: 600; color: #5eb88a; font-family: 'JetBrains Mono', monospace; white-space: nowrap;">₹{w52_high:,.0f}</span>
            </div>
        </div>
        <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.04);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-size: 9px; color: rgba(255,255,255,0.22); text-transform: uppercase; letter-spacing: 1px; font-weight: 600;">Today's Range</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 11px; font-weight: 600; color: #d45d5d; font-family: 'JetBrains Mono', monospace; white-space: nowrap;">₹{today_low:,.2f}</span>
                <div style="flex: 1; position: relative; height: 6px; background: rgba(255,255,255,0.06); border-radius: 3px; overflow: visible;">
                    <div style="position: absolute; left: {max(min(today_open_pct, 100), 0):.1f}%; right: {max(100 - max(min(today_close_pct, 100), 0), 0):.1f}%;
                                height: 100%; border-radius: 3px; background: {'#5eb88a' if today_close >= today_open else '#d45d5d'}; opacity: 0.5;"></div>
                    <div style="position: absolute; top: -4px; left: calc({max(min(today_open_pct, 100), 0):.1f}% - 1px); width: 3px; height: 14px;
                                background: #d4a054; border-radius: 1px;"></div>
                    <div style="position: absolute; top: -3px; left: calc({max(min(today_close_pct, 100), 0):.1f}% - 6px); width: 12px; height: 12px;
                                background: #e8e4de; border-radius: 50%; border: 2px solid #0e0e1c; box-shadow: 0 0 4px rgba(232,228,222,0.3);"></div>
                </div>
                <span style="font-size: 11px; font-weight: 600; color: #5eb88a; font-family: 'JetBrains Mono', monospace; white-space: nowrap;">₹{today_high:,.2f}</span>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    def _render_indicators_row():
        """Render RSI/MACD/ADX/Volume indicator cards."""
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
            <div class="qs-label">ADX</div>
            <div class="qs-value">{qs_adx:.0f}</div>
            <div class="qs-delta" style="color: {'#c9b89c' if qs_adx > 25 else 'rgba(255,255,255,0.3)'};">{qs_adx_label} Trend</div>
        </div>
        <div class="quick-stat">
            <div class="qs-label">Volume</div>
            <div class="qs-value" style="color: {qs_vol_color};">{qs_vol:.1f}x</div>
            <div class="qs-delta" style="color: {qs_vol_color};">vs 20d avg</div>
        </div>
        <div class="quick-stat">
            <div class="qs-label">Tech Score</div>
            <div class="qs-value" style="color: {_hdr_tech_color};">{_hdr_tech}</div>
            <div class="qs-delta" style="color: {_hdr_tech_color};">{_hdr_tech_label}</div>
        </div>
        <div class="quick-stat">
            <div class="qs-label">Fund Score</div>
            <div class="qs-value" style="color: {_hdr_fund_color};">{_hdr_fund}</div>
            <div class="qs-delta" style="color: {_hdr_fund_color};">{_hdr_fund_label}</div>
        </div>
    </div>""", unsafe_allow_html=True)


# ============================================================
# TABS — page-dependent tab bar
# ============================================================
if st.session_state["page"] == "home":
    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
    _HOME_TAB_NAMES = ["Screener", "Paper Trading", "Accuracy"]
    _HOME_TAB_KEYS = ["screener", "paper", "accuracy"]
    tab_screener, tab_paper, tab_accuracy = st.tabs(_HOME_TAB_NAMES)

    # Auto-switch tab from URL query param (?tab=screener)
    _qp_tab = st.query_params.get("tab")
    if _qp_tab and _qp_tab in _HOME_TAB_KEYS:
        _tab_idx = _HOME_TAB_KEYS.index(_qp_tab)
        import streamlit.components.v1 as _components
        _components.html(f"""
            <script>
                const tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
                if (tabs.length > {_tab_idx}) tabs[{_tab_idx}].click();
            </script>
        """, height=0)

if st.session_state["page"] == "analysis":
    _ANALYSIS_TAB_NAMES = ["Chart & Predictions", "Swing",
                           "Scalping", "Fundamentals", "Sector", "Sentiment"]
    _ANALYSIS_TAB_KEYS = ["chart", "swing", "scalp",
                          "fundamentals", "sector", "sentiment"]
    tab_chart, tab_swing, tab_scalp, tab_fund, tab_sector, tab_sentiment = st.tabs(
        _ANALYSIS_TAB_NAMES)

    # Auto-switch tab from URL query param (?tab=swing)
    _qp_tab = st.query_params.get("tab")
    if _qp_tab and _qp_tab in _ANALYSIS_TAB_KEYS:
        _tab_idx = _ANALYSIS_TAB_KEYS.index(_qp_tab)
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
    paper_bgcolor="#16181e",
    plot_bgcolor="#111318",
    font=dict(color="rgba(255,255,255,0.5)",
              family="Sora, sans-serif", size=11),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)",
               zerolinecolor="rgba(255,255,255,0.04)", gridwidth=1),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)",
               zerolinecolor="rgba(255,255,255,0.04)", gridwidth=1),
    margin=dict(l=10, r=10, t=30, b=10),
    hoverlabel=dict(bgcolor="#1b1e25", font_size=12,
                    font_color="#e8e4de", bordercolor="rgba(255,255,255,0.08)"),
)

# colors
C_CYAN = "#c9b89c"
C_AMBER = "#d4a054"
C_PURPLE = "#9b8ec4"
C_GREEN = "#5eb88a"
C_RED = "#d45d5d"
VOL_UP, VOL_DOWN = "#5eb88a", "#d45d5d"


# ============================================================
# HELPERS
# ============================================================
def help_box(text: str):
    st.markdown(
        f"<div class='help-box'>💡 <b>What does this mean?</b> {text}</div>", unsafe_allow_html=True)


def verdict_box(text: str, sentiment: str = "neutral"):
    st.markdown(
        f"<div class='verdict-box {sentiment}'>{text}</div>", unsafe_allow_html=True)


def action_card(action: str, subtitle: str, css_class: str):
    st.markdown(
        f"<div class='action-card {css_class}'>{action}<span>{subtitle}</span></div>", unsafe_allow_html=True)


def checklist(items: list[tuple[str, bool]]):
    html = "<div class='checklist'><b>Checklist — Why this signal?</b><br>"
    for text, positive in items:
        html += f"{'✅' if positive else '❌'} {text}<br>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)




def interpret_rsi(rsi):
    if rsi > 70:
        return "Overbought — stock may be overpriced, could drop soon", "bad"
    if rsi < 30:
        return "Oversold — stock may be underpriced, could bounce up", "good"
    return "Neutral — no extreme buying or selling pressure", "neutral"


def interpret_macd(macd, signal):
    if macd > signal:
        return "Bullish — upward momentum is building", "good"
    return "Bearish — downward momentum is building", "bad"


# ========================================================================
# TAB 1: CHART
# ========================================================================
if st.session_state["page"] == "analysis":
    with tab_chart:
        _render_stock_header()
        _render_indicators_row()

        help_box("Green candles = price went up. Red candles = price went down. "
                 "Lines show average prices over different time periods. Use the checkboxes to customize the chart.")

        cc1, cc2, cc3 = st.columns(3)
        show_sma = cc1.checkbox("Moving Averages", value=True)
        show_bollinger = cc2.checkbox("Bollinger Bands", value=False)
        show_volume = cc3.checkbox("Volume", value=True)

        rows = 2 if show_volume else 1
        row_heights = [0.75, 0.25] if show_volume else [1]
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                            vertical_spacing=0.03, row_heights=row_heights)

        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price",
        ), row=1, col=1)

        if show_sma:
            for col, color, label in [("SMA_20", C_CYAN, "20-Day"), ("SMA_50", C_AMBER, "50-Day"), ("SMA_200", C_PURPLE, "200-Day")]:
                if col in df.columns:
                    fig.add_trace(go.Scatter(x=df.index, y=df[col], name=label, line=dict(
                        width=1, color=color)), row=1, col=1)

        if show_bollinger and "BB_Upper" in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper", line=dict(
                width=1, color="rgba(150,150,150,0.5)")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower", line=dict(
                width=1, color="rgba(150,150,150,0.5)"), fill="tonexty", fillcolor="rgba(150,150,150,0.07)"), row=1, col=1)

        if show_volume:
            vc = [VOL_UP if c >= o else VOL_DOWN for o,
                  c in zip(df["Open"], df["Close"])]
            fig.add_trace(go.Bar(
                x=df.index, y=df["Volume"], name="Volume", marker_color=vc, opacity=0.5), row=2, col=1)

        fig.update_layout(height=550, xaxis_rangeslider_visible=False, legend=dict(
            orientation="h", yanchor="bottom", y=1.02), **CHART_LAYOUT)
        if show_volume:
            fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)", row=2, col=1)
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
            verdict_box(
                "📊 <b>Overall: Healthy.</b> Most indicators are positive — trend and momentum favor buyers.", "good")
        elif bullish_count <= 1:
            verdict_box(
                "📊 <b>Overall: Weak.</b> Most indicators are negative — be cautious, sellers have control.", "bad")
        else:
            verdict_box(
                "📊 <b>Overall: Mixed.</b> No clear direction — best to wait for a clearer setup.", "neutral")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("RSI", f"{rsi_val:.0f}/100")
            verdict_box(rsi_text, rsi_sent)
        with c2:
            st.metric("MACD", f"{macd_val:.2f}")
            verdict_box(macd_text, macd_sent)
        with c3:
            st.metric("ADX (Trend)", f"{adx_val:.0f}/100")
            verdict_box("Strong trend — stock moving decisively." if adx_val >
                        25 else "Weak trend — stock moving sideways.", "good" if adx_val > 25 else "neutral")
        with c4:
            st.metric("Volume", f"{vol_ratio:.1f}x avg")
            if vol_ratio > 1.5:
                verdict_box("High volume — price moves are reliable.", "good")
            elif vol_ratio < 0.7:
                verdict_box("Low volume — price moves may not stick.", "bad")
            else:
                verdict_box("Normal volume.", "neutral")

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
        st.markdown("<div class='section-label'>AI Price Predictions</div>",
                    unsafe_allow_html=True)
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
            fig_pred.add_trace(go.Scatter(
                x=hist_df.index, y=hist_df["Close"], name="Actual Price", line=dict(color=C_CYAN)))
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
            st.dataframe(pd.DataFrame(pred_table),
                         use_container_width=True, hide_index=True)

            # Accuracy
            if "backtest_metrics" in results:
                bm = results["backtest_metrics"]
                ed = bm["ensemble"]["directional_accuracy"]
                er = bm["ensemble"]["rmse"]
                if ed >= 55:
                    verdict_box(
                        f"🎯 <b>Directional Accuracy: {ed:.1f}%</b> — correctly predicts up/down ~{ed:.0f}/100 times. Avg error: ₹{er:.2f}", "good")
                else:
                    verdict_box(
                        f"⚠️ <b>Directional Accuracy: {ed:.1f}%</b> — modest. Use as one input, not sole decision maker. Avg error: ₹{er:.2f}", "neutral")
                if bm.get("improvement_pct", 0) > 0:
                    st.success(
                        f"Ensemble is {bm['improvement_pct']:.1f}% better than LSTM alone")

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
                            st.write(
                                f"Direction: {bm[key]['directional_accuracy']:.1f}%")

            if "actual_vs_predicted" in results:
                with st.expander("Backtest Chart"):
                    eval_df = results["actual_vs_predicted"].tail(120)
                    fig_e = go.Figure()
                    fig_e.add_trace(go.Scatter(
                        x=eval_df.index, y=eval_df["Actual"], name="Actual", line=dict(color=C_CYAN)))
                    fig_e.add_trace(go.Scatter(
                        x=eval_df.index, y=eval_df["Predicted"], name="Predicted", line=dict(color=C_AMBER)))
                    fig_e.update_layout(height=320, **CHART_LAYOUT)
                    st.plotly_chart(fig_e, use_container_width=True)

            if "feature_importance" in results and results["feature_importance"]:
                with st.expander("Feature Importance"):
                    fi = results["feature_importance"]
                    top = dict(list(fi.items())[:10])
                    fig_fi = go.Figure(go.Bar(x=list(top.values()), y=list(
                        top.keys()), orientation="h", marker_color=C_CYAN))
                    fig_fi.update_layout(height=320, **CHART_LAYOUT)
                    fig_fi.update_yaxes(autorange="reversed")
                    st.plotly_chart(fig_fi, use_container_width=True)

            if "train_metrics" in results:
                with st.expander("Training Details"):
                    m1, m2 = st.columns(2)
                    with m1:
                        f1 = go.Figure()
                        f1.add_trace(go.Scatter(
                            y=results["train_metrics"]["loss"], name="Train"))
                        f1.add_trace(go.Scatter(
                            y=results["train_metrics"]["val_loss"], name="Val"))
                        f1.update_layout(
                            height=280, title="Loss", **CHART_LAYOUT)
                        st.plotly_chart(f1, use_container_width=True)
                    with m2:
                        f2 = go.Figure()
                        f2.add_trace(go.Scatter(
                            y=results["train_metrics"]["mae"], name="Train"))
                        f2.add_trace(go.Scatter(
                            y=results["train_metrics"]["val_mae"], name="Val"))
                        f2.update_layout(
                            height=280, title="MAE", **CHART_LAYOUT)
                        st.plotly_chart(f2, use_container_width=True)


# ========================================================================
# TAB 3: SWING
# ========================================================================
if st.session_state["page"] == "analysis":
    with tab_swing:
        _render_stock_header()
        _render_indicators_row()
        help_box("<b>Swing trading</b> = hold a stock for days to weeks. Buy at support, sell at resistance. "
                 "This tab tells you: <b>buy, sell, or wait?</b> With exact entry, stop loss, and targets.")

        signal = generate_swing_signals(df)
        atr_data = calculate_atr_stop_loss(df)
        setup = calculate_trade_setup(
            df, signal, capital=capital, max_risk_pct=max_risk_pct / 100)
        sr_data = calculate_support_resistance(df, lookback=sr_lookback)
        fib_data = calculate_fibonacci_retracements(df, lookback=sr_lookback)
        pivot_data = calculate_pivot_points(df, method=pivot_method)
        patterns = identify_swing_patterns(df)

        # --- 2-Column Layout: Left (Signal + Checklist) | Right (Trade Setup + Targets + Paper Trade) ---
        sig_class = {"BUY": "action-buy",
                     "SELL": "action-sell", "HOLD": "action-hold"}
        sig_advice = {"BUY": "Indicators suggest buying.",
                      "SELL": "Indicators suggest selling.", "HOLD": "Mixed signals — wait."}

        sw_sig_left, sw_sig_right = st.columns([2, 1])

        with sw_sig_left:
            action_card(f"Signal: {signal.signal} — {signal.strength}",
                        f"Confidence: {signal.confidence * 100:.0f}% — {sig_advice.get(signal.signal, '')}", sig_class.get(signal.signal, "action-hold"))

            if signal.reasons:
                checklist([(r, any(w in r.lower() for w in ["bullish", "above", "bounce",
                          "oversold", "upward", "positive"])) for r in signal.reasons])

            # --- KEY LEVELS CARD ---
            _sw_cmp = df["Close"].iloc[-1]
            _sw_high = df["High"].tail(sr_lookback).max()
            _sw_low = df["Low"].tail(sr_lookback).min()
            _sw_vwap_vol = df["Volume"].tail(sr_lookback)
            _sw_vwap_tp = ((df["High"] + df["Low"] +
                           df["Close"]) / 3).tail(sr_lookback)
            _sw_vwap = (_sw_vwap_tp * _sw_vwap_vol).sum() / \
                _sw_vwap_vol.sum() if _sw_vwap_vol.sum() > 0 else _sw_cmp
            _sw_above_vwap = _sw_cmp >= _sw_vwap
            st.markdown(f"""<div class='setup-card' style='border-left: none; padding: 14px 16px; margin-top: 12px;'>
                <div style="font-size: 10px; font-weight: 700; color: rgba(255,255,255,0.3); text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 12px;">Key Levels</div>
                <div style="display: flex; gap: 0; justify-content: space-between;">
                    <div style="text-align: center; flex: 1;">
                        <div style="font-size: 10px; color: rgba(255,255,255,0.4); text-transform: uppercase; margin-bottom: 4px;">VWAP</div>
                        <div style="font-size: 16px; font-weight: 700; color: #e8e4de; font-family: var(--mono);">₹{_sw_vwap:,.2f}</div>
                        <div style="font-size: 10px; color: {'#5eb88a' if _sw_above_vwap else '#d45d5d'}; margin-top: 2px;">{'▲ Above' if _sw_above_vwap else '▼ Below'}</div>
                    </div>
                    <div style="text-align: center; flex: 1;">
                        <div style="font-size: 10px; color: rgba(255,255,255,0.4); text-transform: uppercase; margin-bottom: 4px;">Pivot</div>
                        <div style="font-size: 16px; font-weight: 700; color: #e8e4de; font-family: var(--mono);">₹{pivot_data['PP']:,.2f}</div>
                    </div>
                    <div style="text-align: center; flex: 1;">
                        <div style="font-size: 10px; color: rgba(255,255,255,0.4); text-transform: uppercase; margin-bottom: 4px;">{sr_lookback}D High</div>
                        <div style="font-size: 16px; font-weight: 700; color: #e8e4de; font-family: var(--mono);">₹{_sw_high:,.2f}</div>
                    </div>
                    <div style="text-align: center; flex: 1;">
                        <div style="font-size: 10px; color: rgba(255,255,255,0.4); text-transform: uppercase; margin-bottom: 4px;">{sr_lookback}D Low</div>
                        <div style="font-size: 16px; font-weight: 700; color: #e8e4de; font-family: var(--mono);">₹{_sw_low:,.2f}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            # --- SUPPORT & RESISTANCE BAR CARD ---
            _sw_pivot_levels = [
                ("S3", pivot_data["S3"], "#5eb88a"),
                ("S2", pivot_data["S2"], "#5eb88a"),
                ("S1", pivot_data["S1"], "#5eb88a"),
                ("PIVOT", pivot_data["PP"], "#d4a054"),
                ("R1", pivot_data["R1"], "#d45d5d"),
                ("R2", pivot_data["R2"], "#d45d5d"),
                ("R3", pivot_data["R3"], "#d45d5d"),
            ]
            _sw_pmin = min(l[1] for l in _sw_pivot_levels)
            _sw_pmax = max(l[1] for l in _sw_pivot_levels)
            _sw_prange = _sw_pmax - _sw_pmin if _sw_pmax != _sw_pmin else 1
            _sr_rows = ""
            for label, price, color in _sw_pivot_levels:
                pct = ((price - _sw_pmin) / _sw_prange) * 100
                bar_w = max(pct, 3)
                _sr_rows += f"""<div style="display: flex; align-items: center; padding: 5px 0; border-bottom: 1px solid rgba(255,255,255,0.03);">
                    <span style="font-size: 11px; color: {color}; width: 45px; font-weight: 600;">● {label}</span>
                    <div style="flex: 1; margin: 0 10px; height: 4px; background: rgba(255,255,255,0.05); border-radius: 2px; overflow: hidden;">
                        <div style="width: {bar_w}%; height: 100%; background: {color}; opacity: 0.5; border-radius: 2px;"></div>
                    </div>
                    <span style="font-size: 12px; font-weight: 700; color: {color}; font-family: var(--mono); width: 80px; text-align: right;">₹{price:,.2f}</span>
                </div>"""
            st.markdown(f"""<div class='setup-card' style='border-left: none; padding: 14px 16px; margin-top: 8px;'>
                <div style="font-size: 10px; font-weight: 700; color: rgba(255,255,255,0.3); text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 10px;">Support & Resistance</div>
                {_sr_rows}
            </div>""", unsafe_allow_html=True)

        with sw_sig_right:
            # Quantity input first so charges reflect user selection
            swing_qty = st.number_input(
                "Quantity", min_value=1, value=10, step=5, key="swing_qty_trade")
            _sw_capital_needed = setup.entry_price * swing_qty

            if signal.signal == "SELL":
                sw_t1 = calc_angel_one_charges(
                    setup.target_1, setup.entry_price, swing_qty, "delivery")
                sw_t2 = calc_angel_one_charges(
                    setup.target_2, setup.entry_price, swing_qty, "delivery")
                sw_t3 = calc_angel_one_charges(
                    setup.target_3, setup.entry_price, swing_qty, "delivery")
                sw_sl = calc_angel_one_charges(
                    setup.stop_loss, setup.entry_price, swing_qty, "delivery")
            else:
                sw_t1 = calc_angel_one_charges(
                    setup.entry_price, setup.target_1, swing_qty, "delivery")
                sw_t2 = calc_angel_one_charges(
                    setup.entry_price, setup.target_2, swing_qty, "delivery")
                sw_t3 = calc_angel_one_charges(
                    setup.entry_price, setup.target_3, swing_qty, "delivery")
                sw_sl = calc_angel_one_charges(
                    setup.entry_price, setup.stop_loss, swing_qty, "delivery")

            _sw_risk_ps = abs(setup.entry_price - setup.stop_loss)
            t1_cls = "profit" if sw_t1["net_profit"] > 0 else "loss"
            t2_cls = "profit" if sw_t2["net_profit"] > 0 else "loss"
            t3_cls = "profit" if sw_t3["net_profit"] > 0 else "loss"

            # --- TRADE SETUP CARD ---
            st.markdown(f"""<div class='setup-card' style='border-left: none; padding: 14px 16px;'>
                <div style="font-size: 10px; font-weight: 700; color: rgba(255,255,255,0.3); text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 10px;">Trade Setup</div>
                <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.04);">
                    <span style="font-size: 12px; color: rgba(255,255,255,0.5);">Entry</span>
                    <span style="font-size: 13px; font-weight: 700; color: #e8e4de; font-family: var(--mono);">₹{setup.entry_price:,.2f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.04);">
                    <span style="font-size: 12px; color: rgba(255,255,255,0.5);">Stop Loss</span>
                    <span style="font-size: 13px; font-weight: 700; color: #d45d5d; font-family: var(--mono);">₹{setup.stop_loss:,.2f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.04);">
                    <span style="font-size: 12px; color: rgba(255,255,255,0.5);">Risk per share</span>
                    <span style="font-size: 13px; font-weight: 700; color: #e8e4de; font-family: var(--mono);">₹{_sw_risk_ps:,.2f} (1:{setup.risk_reward_1})</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.04);">
                    <span style="font-size: 12px; color: rgba(255,255,255,0.5);">Amount needed ({swing_qty} qty)</span>
                    <span style="font-size: 13px; font-weight: 700; color: #d4a054; font-family: var(--mono);">₹{_sw_capital_needed:,.2f}</span>
                </div>
                <div style="background: rgba(212,93,93,0.08); border: 1px solid rgba(212,93,93,0.1); border-radius: 6px; padding: 6px 10px; margin-top: 8px; text-align: center;">
                    <span style="font-size: 12px; color: #d45d5d; font-family: var(--mono);">If stopped ({swing_qty} qty): -₹{abs(sw_sl['net_profit']):,.2f}</span>
                </div>
            </div>""", unsafe_allow_html=True)

            # --- TARGETS CARD ---
            st.markdown(f"""<div class='setup-card' style='border-left: none; padding: 14px 16px;'>
                <div style="font-size: 10px; font-weight: 700; color: rgba(255,255,255,0.3); text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 10px;">Targets</div>
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.04);">
                    <span style="font-size: 12px; color: rgba(255,255,255,0.5);">Target 1</span>
                    <span style="font-size: 13px; font-weight: 700; color: #5eb88a; font-family: var(--mono);">₹{setup.target_1:,.2f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 2px 0 6px 0;">
                    <span style="font-size: 11px; color: rgba(255,255,255,0.28);">+₹{abs(setup.target_1 - setup.entry_price):,.2f}/share × {swing_qty}</span>
                    <span style="font-size: 11px; color: rgba(255,255,255,0.28);">Net: <span class='{t1_cls}' style="font-size: 11px;">₹{sw_t1['net_profit']:,.2f}</span> <span style="color: rgba(255,255,255,0.2);">(charges ₹{sw_t1['total_charges']:.2f})</span></span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.04);">
                    <span style="font-size: 12px; color: rgba(255,255,255,0.5);">Target 2</span>
                    <span style="font-size: 13px; font-weight: 700; color: #5eb88a; font-family: var(--mono);">₹{setup.target_2:,.2f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 2px 0 6px 0;">
                    <span style="font-size: 11px; color: rgba(255,255,255,0.28);">+₹{abs(setup.target_2 - setup.entry_price):,.2f}/share × {swing_qty}</span>
                    <span style="font-size: 11px; color: rgba(255,255,255,0.28);">Net: <span class='{t2_cls}' style="font-size: 11px;">₹{sw_t2['net_profit']:,.2f}</span> <span style="color: rgba(255,255,255,0.2);">(charges ₹{sw_t2['total_charges']:.2f})</span></span>
                </div>
            </div>""", unsafe_allow_html=True)

            # --- PAPER TRADE SECTION ---
            if signal.signal in ("BUY", "SELL"):
                with st.container(border=True):
                    st.markdown("""<div style="font-size: 10px; font-weight: 700; color: rgba(255,255,255,0.3); text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 4px;">Paper Trade</div>""", unsafe_allow_html=True)
                    _sw_fund = get_fund_balance("swing")
                    _sw_fund_active = _sw_fund["initial_capital"] > 0
                    _sw_exit_opts = [f"Target 1 — ₹{setup.target_1:,.1f}"]
                    if setup.target_2:
                        _sw_exit_opts.append(
                            f"Target 2 — ₹{setup.target_2:,.1f}")
                    if setup.target_3:
                        _sw_exit_opts.append(
                            f"Target 3 — ₹{setup.target_3:,.1f}")
                    _sw_exit_idx = st.selectbox("Exit at", range(len(_sw_exit_opts)), index=0, key="swing_exit_target",
                                                format_func=lambda i: _sw_exit_opts[i],
                                                help="Trade auto-closes when this target is hit.")
                    _sw_exit = ["T1", "T2", "T3"][_sw_exit_idx]
                    if _sw_fund_active:
                        st.markdown(f"""<div style="display: flex; justify-content: space-between; padding: 6px 0;">
                            <span style="font-size: 11px; color: rgba(255,255,255,0.35); font-style: italic;">Margin required</span>
                            <span style="font-size: 12px; font-weight: 700; color: #e8e4de; font-family: var(--mono);">₹{_sw_capital_needed:,.0f}</span>
                        </div>""", unsafe_allow_html=True)
                    if _sw_fund_active and _sw_fund["available"] < _sw_capital_needed:
                        st.warning(
                            f"Insufficient funds. Need ₹{_sw_capital_needed:,.0f}, available ₹{_sw_fund['available']:,.0f}")
                    _sw_btn_label = f"📝 Paper Trade — {signal.signal} {ticker}"
                    st.markdown('<div class="paper-trade-btn">',
                                unsafe_allow_html=True)
                    if st.button(_sw_btn_label, key="swing_paper_btn", type="primary", use_container_width=True):
                        try:
                            tid = place_trade(
                                ticker=ticker, trade_type="swing", direction=signal.signal,
                                entry_price=setup.entry_price, stop_loss=setup.stop_loss,
                                target_1=setup.target_1, target_2=setup.target_2, target_3=setup.target_3,
                                quantity=swing_qty, signal_strength=signal.strength, confidence=signal.confidence,
                                reasons=", ".join(signal.reasons) if signal.reasons else "", exit_target=_sw_exit,
                            )
                            st.success(
                                f"Trade #{tid} placed! Exit at {_sw_exit} | ₹{setup.entry_price:,.2f} × {swing_qty}")
                        except ValueError as e:
                            st.error(str(e))
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown("""<div class="empty-state" style="padding: 16px;">
                    <div class="es-icon" style="font-size: 24px;">⏸️</div>
                    <div class="es-title" style="font-size: 13px;">No active signal</div>
                    <div class="es-desc" style="font-size: 12px;">Wait for a BUY or SELL signal to place a paper trade.</div>
                </div>""", unsafe_allow_html=True)

        # --- Full width: Support & Resistance Chart ---
        st.markdown("#### Support & Resistance")
        chart_data = df.tail(sr_lookback)
        fig_sr = go.Figure()
        fig_sr.add_trace(go.Candlestick(x=chart_data.index, open=chart_data["Open"], high=chart_data["High"],
                                        low=chart_data["Low"], close=chart_data["Close"], name="Price"))
        for lv in sr_data["support_levels"]:
            fig_sr.add_hline(y=lv["price"], line_dash="dash", line_color="rgba(16,185,129,0.6)",
                             annotation_text=f"S ₹{lv['price']:.0f} ({lv['touches']}x)")
        for lv in sr_data["resistance_levels"]:
            fig_sr.add_hline(y=lv["price"], line_dash="dash", line_color="rgba(239,68,68,0.6)",
                             annotation_text=f"R ₹{lv['price']:.0f} ({lv['touches']}x)")
        fig_sr.add_hline(y=pivot_data["PP"], line_dash="dot", line_color=C_AMBER,
                         annotation_text=f"Pivot ₹{pivot_data['PP']:.0f}")
        fig_sr.update_layout(
            height=450, xaxis_rangeslider_visible=False, **CHART_LAYOUT)
        st.plotly_chart(fig_sr, use_container_width=True)

        # S/R Levels summary
        _sup_levels = sr_data["support_levels"][:3]
        _res_levels = sr_data["resistance_levels"][:3]
        _sr_html = '<div style="display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px;">'
        for lv in _sup_levels:
            _sr_html += f'<div style="background: rgba(94,184,138,0.08); border: 1px solid rgba(94,184,138,0.15); border-radius: 8px; padding: 8px 14px;">'
            _sr_html += f'<div style="font-size: 10px; color: rgba(255,255,255,0.45); text-transform: uppercase; letter-spacing: 0.5px;">Support</div>'
            _sr_html += f'<div style="font-size: 14px; font-weight: 700; color: #5eb88a; font-family: \'JetBrains Mono\', monospace;">₹{lv["price"]:,.0f}</div>'
            _sr_html += f'<div style="font-size: 10px; color: rgba(255,255,255,0.22);">{lv["touches"]}x touches</div></div>'
        for lv in _res_levels:
            _sr_html += f'<div style="background: rgba(212,93,93,0.08); border: 1px solid rgba(212,93,93,0.15); border-radius: 8px; padding: 8px 14px;">'
            _sr_html += f'<div style="font-size: 10px; color: rgba(255,255,255,0.45); text-transform: uppercase; letter-spacing: 0.5px;">Resistance</div>'
            _sr_html += f'<div style="font-size: 14px; font-weight: 700; color: #d45d5d; font-family: \'JetBrains Mono\', monospace;">₹{lv["price"]:,.0f}</div>'
            _sr_html += f'<div style="font-size: 10px; color: rgba(255,255,255,0.22);">{lv["touches"]}x touches</div></div>'
        _sr_html += '</div>'
        st.markdown(_sr_html, unsafe_allow_html=True)

        with st.expander("Fibonacci Levels"):
            fig_fib = go.Figure()
            fig_fib.add_trace(go.Candlestick(x=chart_data.index, open=chart_data["Open"], high=chart_data["High"],
                                             low=chart_data["Low"], close=chart_data["Close"], name="Price"))
            fib_colors = [C_RED, C_AMBER, C_AMBER, C_GREEN, C_CYAN]
            for i, (label, level) in enumerate(fib_data["retracements"].items()):
                fig_fib.add_hline(y=level, line_dash="dash", line_color=fib_colors[i % len(fib_colors)],
                                  annotation_text=f"Fib {label}: ₹{level:.0f}")
            fig_fib.update_layout(
                height=400, xaxis_rangeslider_visible=False, **CHART_LAYOUT)
            st.plotly_chart(fig_fib, use_container_width=True)

        with st.expander("Pivot Points"):
            pivot_cols = st.columns(7)
            for i, key in enumerate(["S3", "S2", "S1", "PP", "R1", "R2", "R3"]):
                pivot_cols[i].metric(key, f"₹{pivot_data[key]:,.2f}")

        if patterns:
            st.markdown("#### Detected Patterns")
            for p in patterns:
                s = "good" if "bullish" in p["implication"].lower(
                ) else "bad" if "bearish" in p["implication"].lower() else "neutral"
                verdict_box(
                    f"<b>{p['pattern']}</b> ({p['confidence']*100:.0f}%) — {p['implication']}", s)
        else:
            st.markdown("""<div class="empty-state" style="padding: 24px;">
                <div class="es-icon" style="font-size: 32px;">🔎</div>
                <div class="es-title" style="font-size: 15px;">No chart patterns detected</div>
                <div class="es-desc" style="font-size: 13px;">No significant bullish or bearish patterns found in recent data.</div>
            </div>""", unsafe_allow_html=True)


# ========================================================================
# TAB 4: SCALPING
# ========================================================================
if st.session_state["page"] == "analysis":
    with tab_scalp:
        _render_stock_header()
        _render_indicators_row()
        help_box(
            "<b>Scalping</b> = buy & sell within minutes on 5-min candles. Always use stop losses!")

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
            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
            scalp_btn = st.button("Load Intraday Data",
                                  type="primary", use_container_width=True)

        # Clear stale scalp data if ticker changed
        if "scalp_data_ticker" in st.session_state and st.session_state["scalp_data_ticker"] != ticker:
            st.session_state.pop("scalp_data", None)
            st.session_state.pop("scalp_data_ticker", None)

        if scalp_btn:
            with st.spinner("Fetching 5-min candles..."):
                try:
                    idf = fetch_intraday_data(
                        ticker, interval="5m", period=scalp_period)
                    idf = add_scalping_indicators(idf)
                    st.session_state["scalp_data"] = idf
                    st.session_state["scalp_data_ticker"] = ticker
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.info(
                        "Intraday data only available during market hours (Mon-Fri, 9:15 AM - 3:30 PM IST)")

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

            # --- Signal + scalpability ---
            scalpability = micro["scalpability_score"]
            scalp_class = {"LONG": "action-long",
                           "SHORT": "action-short", "NO_TRADE": "action-notrade"}
            scalp_label = {
                "LONG": "BUY (Long)", "SHORT": "SELL (Short)", "NO_TRADE": "NO TRADE — Wait"}
            scalp_advice = {"LONG": "Price likely going UP.",
                            "SHORT": "Price likely going DOWN.", "NO_TRADE": "Signals weak — don't trade."}
            scalp_score_color = "#5eb88a" if scalpability >= 65 else "#d4a054" if scalpability >= 40 else "#d45d5d"

            # --- MARKET TIME WARNING ---
            ist = pytz.timezone("Asia/Kolkata")
            now_ist = datetime.now(ist)
            market_open = now_ist.replace(
                hour=9, minute=15, second=0, microsecond=0)
            market_close = now_ist.replace(
                hour=15, minute=30, second=0, microsecond=0)
            is_weekday = now_ist.weekday() < 5

            risk_ps = abs(ss.entry_price - ss.stop_loss)
            rew_ps = abs(ss.target_1 - ss.entry_price)

            # --- 2-Column Layout: Left (Signal + Checklist + Levels) | Right (Trade Setup + Targets + Paper Trade) ---
            sc_sig_left, sc_sig_right = st.columns([2, 1])

            with sc_sig_left:
                action_card(
                    scalp_label.get(ss.signal, "NO TRADE"),
                    f"{ss.strength} | {ss.confidence*100:.0f}% — {scalp_advice.get(ss.signal, '')} "
                    f"| Scalpability: <span style='color:{scalp_score_color}'>{scalpability}/100</span>",
                    scalp_class.get(ss.signal, "action-notrade"),
                )

                # Market time warnings
                if is_weekday and market_open <= now_ist <= market_close:
                    mins_left = int(
                        (market_close - now_ist).total_seconds() / 60)
                    if mins_left <= 30:
                        verdict_box(
                            f"<b>Market closes in {mins_left} min!</b> Avoid new scalp trades — "
                            f"not enough time for targets to be reached. Exit open positions.", "bad")
                    elif mins_left <= 60:
                        verdict_box(
                            f"<b>{mins_left} min to market close.</b> Use tighter targets only. "
                            f"Target 2 unlikely — focus on Target 1 or skip.", "neutral")
                    elif mins_left <= 90:
                        verdict_box(
                            f"<b>{mins_left} min to close.</b> Be selective — only high-confidence setups.", "neutral")
                elif is_weekday and now_ist > market_close:
                    verdict_box(
                        "<b>Market closed.</b> Data from last trading session.", "neutral")
                elif not is_weekday:
                    verdict_box(
                        "<b>Weekend — market closed.</b> Data from last trading session.", "neutral")

                # Screener consistency
                if screener_scalp_match and screener_scalp_match["signal"] != ss.signal:
                    verdict_box(
                        f"Screener showed <b>{screener_scalp_match['signal']}</b>, now <b>{ss.signal}</b> — signal changed. Trust latest.",
                        "neutral")

                if ss.reasons:
                    checklist([(r, any(w in r.lower() for w in ["bullish", "above", "bounce",
                              "oversold", "positive", "up candle", "upward"])) for r in ss.reasons])

                # --- INTRADAY LEVELS CARD ---
                _sc_cmp = idf["Close"].iloc[-1]
                _sc_above_vwap = _sc_cmp >= levels['vwap']
                st.markdown(f"""<div class='setup-card' style='border-left: none; padding: 14px 16px; margin-top: 12px;'>
                    <div style="font-size: 10px; font-weight: 700; color: rgba(255,255,255,0.3); text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 12px;">Intraday Levels</div>
                    <div style="display: flex; gap: 0; justify-content: space-between;">
                        <div style="text-align: center; flex: 1;">
                            <div style="font-size: 10px; color: rgba(255,255,255,0.4); text-transform: uppercase; margin-bottom: 4px;">VWAP</div>
                            <div style="font-size: 16px; font-weight: 700; color: #e8e4de; font-family: var(--mono);">₹{levels['vwap']:,.2f}</div>
                            <div style="font-size: 10px; color: {'#5eb88a' if _sc_above_vwap else '#d45d5d'}; margin-top: 2px;">{'▲ Above' if _sc_above_vwap else '▼ Below'}</div>
                        </div>
                        <div style="text-align: center; flex: 1;">
                            <div style="font-size: 10px; color: rgba(255,255,255,0.4); text-transform: uppercase; margin-bottom: 4px;">Pivot</div>
                            <div style="font-size: 16px; font-weight: 700; color: #e8e4de; font-family: var(--mono);">₹{levels['pivot']:,.2f}</div>
                        </div>
                        <div style="text-align: center; flex: 1;">
                            <div style="font-size: 10px; color: rgba(255,255,255,0.4); text-transform: uppercase; margin-bottom: 4px;">Today High</div>
                            <div style="font-size: 16px; font-weight: 700; color: #e8e4de; font-family: var(--mono);">₹{levels['today_high']:,.2f}</div>
                        </div>
                        <div style="text-align: center; flex: 1;">
                            <div style="font-size: 10px; color: rgba(255,255,255,0.4); text-transform: uppercase; margin-bottom: 4px;">Today Low</div>
                            <div style="font-size: 16px; font-weight: 700; color: #e8e4de; font-family: var(--mono);">₹{levels['today_low']:,.2f}</div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

                # --- SUPPORT & RESISTANCE BAR CARD (Camarilla) ---
                _sc_cam_levels = [
                    ("S3", levels["cam_s3"], "#5eb88a"),
                    ("S2", levels["cam_s2"], "#5eb88a"),
                    ("S1", levels["cam_s1"], "#5eb88a"),
                    ("PIVOT", levels["pivot"], "#d4a054"),
                    ("R1", levels["cam_r1"], "#d45d5d"),
                    ("R2", levels["cam_r2"], "#d45d5d"),
                    ("R3", levels["cam_r3"], "#d45d5d"),
                ]
                _sc_pmin = min(l[1] for l in _sc_cam_levels)
                _sc_pmax = max(l[1] for l in _sc_cam_levels)
                _sc_prange = _sc_pmax - _sc_pmin if _sc_pmax != _sc_pmin else 1
                _sc_sr_rows = ""
                for label, price, color in _sc_cam_levels:
                    pct = ((price - _sc_pmin) / _sc_prange) * 100
                    bar_w = max(pct, 3)
                    _sc_sr_rows += f"""<div style="display: flex; align-items: center; padding: 5px 0; border-bottom: 1px solid rgba(255,255,255,0.03);">
                        <span style="font-size: 11px; color: {color}; width: 45px; font-weight: 600;">● {label}</span>
                        <div style="flex: 1; margin: 0 10px; height: 4px; background: rgba(255,255,255,0.05); border-radius: 2px; overflow: hidden;">
                            <div style="width: {bar_w}%; height: 100%; background: {color}; opacity: 0.5; border-radius: 2px;"></div>
                        </div>
                        <span style="font-size: 12px; font-weight: 700; color: {color}; font-family: var(--mono); width: 80px; text-align: right;">₹{price:,.2f}</span>
                    </div>"""
                st.markdown(f"""<div class='setup-card' style='border-left: none; padding: 14px 16px; margin-top: 8px;'>
                    <div style="font-size: 10px; font-weight: 700; color: rgba(255,255,255,0.3); text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 10px;">Support & Resistance</div>
                    {_sc_sr_rows}
                </div>""", unsafe_allow_html=True)

            with sc_sig_right:
                scalp_qty = st.number_input(
                    "Quantity", min_value=1, value=100, step=25, key="scalp_qty_trade")
                _sc_position_value = ss.entry_price * scalp_qty
                _sc_margin_needed = _sc_position_value / 5  # 5x leverage
                # Calculate charges
                if ss.signal == "SHORT":
                    buy_p, sell_p = ss.target_1, ss.entry_price
                    buy_p2, sell_p2 = ss.target_2, ss.entry_price
                    sl_buy, sl_sell = ss.stop_loss, ss.entry_price
                else:
                    buy_p, sell_p = ss.entry_price, ss.target_1
                    buy_p2, sell_p2 = ss.entry_price, ss.target_2
                    sl_buy, sl_sell = ss.entry_price, ss.stop_loss
                t1_charges = calc_angel_one_charges(buy_p, sell_p, scalp_qty)
                t2_charges = calc_angel_one_charges(buy_p2, sell_p2, scalp_qty)
                loss_charges = calc_angel_one_charges(
                    sl_buy, sl_sell, scalp_qty)
                p1_class = "profit" if t1_charges["net_profit"] > 0 else "loss"
                p2_class = "profit" if t2_charges["net_profit"] > 0 else "loss"

                # --- TRADE SETUP CARD ---
                st.markdown(f"""<div class='setup-card' style='border-left: none; padding: 14px 16px;'>
                    <div style="font-size: 10px; font-weight: 700; color: rgba(255,255,255,0.3); text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 10px;">Trade Setup</div>
                    <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.04);">
                        <span style="font-size: 12px; color: rgba(255,255,255,0.5);">Entry</span>
                        <span style="font-size: 13px; font-weight: 700; color: #e8e4de; font-family: var(--mono);">₹{ss.entry_price:,.2f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.04);">
                        <span style="font-size: 12px; color: rgba(255,255,255,0.5);">Stop Loss</span>
                        <span style="font-size: 13px; font-weight: 700; color: #d45d5d; font-family: var(--mono);">₹{ss.stop_loss:,.2f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.04);">
                        <span style="font-size: 12px; color: rgba(255,255,255,0.5);">Risk per share</span>
                        <span style="font-size: 13px; font-weight: 700; color: #e8e4de; font-family: var(--mono);">₹{risk_ps:,.2f} (1:{ss.risk_reward})</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.04);">
                        <span style="font-size: 12px; color: rgba(255,255,255,0.5);">Amount needed ({scalp_qty} qty)</span>
                        <span style="font-size: 13px; font-weight: 700; color: #d4a054; font-family: var(--mono);">₹{_sc_position_value:,.2f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.04);">
                        <span style="font-size: 12px; color: rgba(255,255,255,0.5);">Margin (5x leverage)</span>
                        <span style="font-size: 13px; font-weight: 700; color: #d4a054; font-family: var(--mono);">₹{_sc_margin_needed:,.2f}</span>
                    </div>
                    <div style="background: rgba(212,93,93,0.08); border: 1px solid rgba(212,93,93,0.1); border-radius: 6px; padding: 6px 10px; margin-top: 8px; text-align: center;">
                        <span style="font-size: 12px; color: #d45d5d; font-family: var(--mono);">If stopped ({scalp_qty} qty): -₹{abs(loss_charges['net_profit']):,.2f}</span>
                    </div>
                </div>""", unsafe_allow_html=True)

                # --- TARGETS CARD ---
                st.markdown(f"""<div class='setup-card' style='border-left: none; padding: 14px 16px;'>
                    <div style="font-size: 10px; font-weight: 700; color: rgba(255,255,255,0.3); text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 10px;">Targets</div>
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.04);">
                        <span style="font-size: 12px; color: rgba(255,255,255,0.5);">Target 1</span>
                        <span style="font-size: 13px; font-weight: 700; color: #5eb88a; font-family: var(--mono);">₹{ss.target_1:,.2f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 2px 0 6px 0; border-bottom: 1px solid rgba(255,255,255,0.04);">
                        <span style="font-size: 11px; color: rgba(255,255,255,0.28);">+₹{rew_ps:.2f}/share × {scalp_qty}</span>
                        <span style="font-size: 11px; color: rgba(255,255,255,0.28);">Net: <span class='{p1_class}' style="font-size: 11px;">₹{t1_charges['net_profit']:,.2f}</span> <span style="color: rgba(255,255,255,0.2);">(charges ₹{t1_charges['total_charges']:.2f})</span></span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.04);">
                        <span style="font-size: 12px; color: rgba(255,255,255,0.5);">Target 2</span>
                        <span style="font-size: 13px; font-weight: 700; color: #5eb88a; font-family: var(--mono);">₹{ss.target_2:,.2f}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 2px 0 6px 0;">
                        <span style="font-size: 11px; color: rgba(255,255,255,0.28);">+₹{abs(ss.target_2 - ss.entry_price):.2f}/share × {scalp_qty}</span>
                        <span style="font-size: 11px; color: rgba(255,255,255,0.28);">Net: <span class='{p2_class}' style="font-size: 11px;">₹{t2_charges['net_profit']:,.2f}</span> <span style="color: rgba(255,255,255,0.2);">(charges ₹{t2_charges['total_charges']:.2f})</span></span>
                    </div>
                </div>""", unsafe_allow_html=True)

                # --- PAPER TRADE SECTION ---
                if ss.signal in ("LONG", "SHORT"):
                    with st.container(border=True):
                        st.markdown(
                            """<div style="font-size: 10px; font-weight: 700; color: rgba(255,255,255,0.3); text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 4px;">Paper Trade</div>""", unsafe_allow_html=True)
                        _sc_fund = get_fund_balance("scalp")
                        _sc_fund_active = _sc_fund["initial_capital"] > 0
                        _sc_exit_opts = [f"Target 1 — ₹{ss.target_1:,.1f}"]
                        if ss.target_2:
                            _sc_exit_opts.append(
                                f"Target 2 — ₹{ss.target_2:,.1f}")
                        _sc_exit_idx = st.selectbox("Exit at", range(len(_sc_exit_opts)), index=0, key="scalp_exit_target",
                                                    format_func=lambda i: _sc_exit_opts[i],
                                                    help="Trade auto-closes at this target. Squared off at 3:30 PM if not hit.")
                        _sc_exit = ["T1", "T2"][_sc_exit_idx]

                        if _sc_fund_active:
                            st.markdown(f"""<div style="display: flex; justify-content: space-between; padding: 6px 0;">
                                <span style="font-size: 11px; color: rgba(255,255,255,0.35); font-style: italic;">Margin required</span>
                                <span style="font-size: 12px; font-weight: 700; color: #e8e4de; font-family: var(--mono);">₹{_sc_margin_needed:,.0f}</span>
                            </div>""", unsafe_allow_html=True)

                        if _sc_fund_active and _sc_fund["available"] < _sc_margin_needed:
                            st.warning(
                                f"Insufficient funds. Need ₹{_sc_margin_needed:,.0f} margin, available ₹{_sc_fund['available']:,.0f}")

                        _sc_btn_label = f"📝 Paper Trade — {ss.signal} {ticker}"
                        st.markdown(
                            '<div class="paper-trade-btn">', unsafe_allow_html=True)
                        if st.button(_sc_btn_label, key="scalp_paper_btn", type="primary", use_container_width=True):
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
                                    reasons=", ".join(
                                        ss.reasons) if ss.reasons else "",
                                    exit_target=_sc_exit,
                                )
                                st.success(
                                    f"Trade #{tid} placed! Exit at {_sc_exit} | ₹{ss.entry_price:,.2f} × {scalp_qty}")
                            except ValueError as e:
                                st.error(str(e))
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown("""<div class="empty-state" style="padding: 16px;">
                        <div class="es-icon" style="font-size: 24px;">⏸️</div>
                        <div class="es-title" style="font-size: 13px;">No active signal</div>
                        <div class="es-desc" style="font-size: 12px;">Wait for a LONG or SHORT signal to place a paper trade.</div>
                    </div>""", unsafe_allow_html=True)

            # --- Full width: Intraday Chart ---
            st.markdown("#### Intraday Chart")
            fig_s = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                  vertical_spacing=0.03, row_heights=[0.75, 0.25])
            fig_s.add_trace(go.Candlestick(
                x=idf.index, open=idf["Open"], high=idf["High"], low=idf["Low"], close=idf["Close"], name="Price"), row=1, col=1)
            fig_s.add_trace(go.Scatter(x=idf.index, y=idf["EMA_5"], name="EMA 5", line=dict(
                width=1, color=C_CYAN)), row=1, col=1)
            fig_s.add_trace(go.Scatter(x=idf.index, y=idf["EMA_9"], name="EMA 9", line=dict(
                width=1, color=C_AMBER)), row=1, col=1)
            fig_s.add_trace(go.Scatter(x=idf.index, y=idf["EMA_21"], name="EMA 21", line=dict(
                width=1, color=C_PURPLE)), row=1, col=1)
            fig_s.add_trace(go.Scatter(x=idf.index, y=idf["VWAP"], name="VWAP", line=dict(
                width=2, color=C_AMBER, dash="dash")), row=1, col=1)
            fig_s.add_trace(go.Scatter(x=idf.index, y=idf["BB_Upper_Scalp"], name="BB+", line=dict(
                width=1, color="rgba(150,150,150,0.4)")), row=1, col=1)
            fig_s.add_trace(go.Scatter(x=idf.index, y=idf["BB_Lower_Scalp"], name="BB-", line=dict(
                width=1, color="rgba(150,150,150,0.4)"), fill="tonexty", fillcolor="rgba(150,150,150,0.06)"), row=1, col=1)
            fig_s.add_hline(y=levels["pivot"], line_dash="dot",
                            line_color=C_CYAN, annotation_text="Pivot", row=1, col=1)
            fig_s.add_hline(y=levels["prev_high"], line_dash="dot",
                            line_color=C_RED, annotation_text="Prev High", row=1, col=1)
            fig_s.add_hline(y=levels["prev_low"], line_dash="dot",
                            line_color=C_GREEN, annotation_text="Prev Low", row=1, col=1)
            svc = [VOL_UP if c >= o else VOL_DOWN for o,
                   c in zip(idf["Open"], idf["Close"])]
            fig_s.add_trace(go.Bar(
                x=idf.index, y=idf["Volume"], name="Vol", marker_color=svc, opacity=0.5), row=2, col=1)
            fig_s.update_layout(height=500, xaxis_rangeslider_visible=False, legend=dict(
                orientation="h", yanchor="bottom", y=1.02), **CHART_LAYOUT)
            fig_s.update_yaxes(
                gridcolor="rgba(255,255,255,0.04)", row=2, col=1)
            st.plotly_chart(fig_s, use_container_width=True)

            with st.expander("Market Microstructure"):
                mi1, mi2, mi3, mi4 = st.columns(4)
                mi1.metric("ATR", f"₹{micro['atr']:.2f}")
                mi2.metric("Candle Size", f"{micro['avg_candle_range']:.3f}%")
                mi3.metric(
                    "Consecutive", f"{micro['consecutive_candles']} {micro['consecutive_direction']}")
                mi4.metric("Trend Slope", f"{micro['trend_slope']:.4f}")


# ========================================================================
# TAB: PAPER TRADING DASHBOARD
# ========================================================================
if st.session_state["page"] == "home":
    with tab_paper:
        st.markdown("""<div style="margin-bottom: 12px;">
            <span style="font-size: 20px; font-weight: 700; color: #e8e4de;">Paper Trading</span>
            <div style="font-size: 12px; color: rgba(255,255,255,0.35); margin-top: 2px;">Track &amp; manage your simulated trades</div>
        </div>""", unsafe_allow_html=True)
        # Aggregate stats for the header (all trades)
        _pt_all_stats = get_paper_stats(None)
        _pt_swing_stats = get_paper_stats("swing")
        _pt_scalp_stats = get_paper_stats("scalp")

        # -- Performance summary banner --
        _pnl = _pt_all_stats["total_pnl"]
        _pnl_color = "#5eb88a" if _pnl >= 0 else "#d45d5d"
        _pnl_sign = "+" if _pnl >= 0 else ""
        _total_profit = _pt_all_stats["total_profit"]
        _total_loss = _pt_all_stats["total_loss"]
        _total_charges = _pt_all_stats.get("total_charges", 0)
        _wr = _pt_all_stats["win_rate"]
        _wr_color = "#5eb88a" if _wr >= 50 else "#d4a054" if _wr >= 30 else "#d45d5d"

        # Compute combined funds available
        _fund_swing = get_fund_balance("swing")
        _fund_scalp = get_fund_balance("scalp")
        _any_funds = _fund_swing["initial_capital"] > 0 or _fund_scalp["initial_capital"] > 0
        _total_available = _fund_swing["available"] + _fund_scalp["available"]
        _total_deployed = _fund_swing["deployed"] + _fund_scalp["deployed"]
        _total_initial = _fund_swing["initial_capital"] + \
            _fund_scalp["initial_capital"]

        _funds_html = ""
        if _any_funds:
            _funds_html = (
                f'<div style="margin-top: 16px; padding-top: 14px; border-top: 1px solid rgba(255,255,255,0.05);'
                f' display: flex; gap: 24px; flex-wrap: wrap; align-items: center;">'
                f'<div>'
                f'<div style="font-size: 11px; color: rgba(255,255,255,0.45); text-transform: uppercase; letter-spacing: 0.5px;">Funds Available</div>'
                f'<div style="font-size: 22px; font-weight: 800; color: #5eb88a;">₹{_total_available:,.0f}</div>'
                f'</div>'
                f'<div>'
                f'<div style="font-size: 11px; color: rgba(255,255,255,0.45); text-transform: uppercase; letter-spacing: 0.5px;">Deployed</div>'
                f'<div style="font-size: 22px; font-weight: 800; color: #d4a054;">₹{_total_deployed:,.0f}</div>'
                f'</div>'
                f'<div>'
                f'<div style="font-size: 11px; color: rgba(255,255,255,0.45); text-transform: uppercase; letter-spacing: 0.5px;">Total Capital</div>'
                f'<div style="font-size: 22px; font-weight: 800; color: #e8e4de;">₹{_total_initial:,.0f}</div>'
                f'</div>'
                f'</div>'
            )

        st.markdown(f"""<div style="
            background: #16181e;
            border: 1px solid rgba(255,255,255,0.05); border-radius: 18px;
            padding: 24px 28px; margin-bottom: 14px; position: relative; overflow: hidden;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 16px;">
                <div>
                    <div style="font-size: 13px; color: rgba(255,255,255,0.45); text-transform: uppercase; letter-spacing: 1px; font-weight: 700;">
                        Paper Trading — Net P&L</div>
                    <div style="font-size: 32px; font-weight: 800; color: {_pnl_color}; margin-top: 4px;">
                        {_pnl_sign}₹{_pnl:,.2f}</div>
                    <div style="font-size: 13px; color: rgba(255,255,255,0.45); margin-top: 2px;">
                        {_pt_all_stats['total_trades']} closed trades | {_pt_all_stats['open_trades']} open</div>
                </div>
                <div style="display: flex; gap: 24px; flex-wrap: wrap;">
                    <div style="text-align: center;">
                        <div style="font-size: 20px; font-weight: 800; color: #5eb88a;">+₹{_total_profit:,.0f}</div>
                        <div style="font-size: 11px; color: rgba(255,255,255,0.45); text-transform: uppercase; letter-spacing: 0.6px; font-weight: 600;">Total Profit</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 20px; font-weight: 800; color: #d45d5d;">₹{_total_loss:,.0f}</div>
                        <div style="font-size: 11px; color: rgba(255,255,255,0.45); text-transform: uppercase; letter-spacing: 0.6px; font-weight: 600;">Total Loss</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 20px; font-weight: 800; color: {_wr_color};">{_wr}%</div>
                        <div style="font-size: 11px; color: rgba(255,255,255,0.45); text-transform: uppercase; letter-spacing: 0.6px; font-weight: 600;">Win Rate</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 20px; font-weight: 800; color: #e8e4de;">{_pt_all_stats['profit_factor']}</div>
                        <div style="font-size: 11px; color: rgba(255,255,255,0.45); text-transform: uppercase; letter-spacing: 0.6px; font-weight: 600;">Profit Factor</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 20px; font-weight: 800; color: rgba(255,255,255,0.55);">₹{_total_charges:,.0f}</div>
                        <div style="font-size: 11px; color: rgba(255,255,255,0.45); text-transform: uppercase; letter-spacing: 0.6px; font-weight: 600;">Total Charges</div>
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
                _return_pct = ((_current_val - _total) /
                               _total * 100) if _total > 0 else 0
                _ret_color = "#5eb88a" if _return_pct >= 0 else "#d45d5d"
                _ret_sign = "+" if _return_pct >= 0 else ""
                _util_pct = (_deployed / _total * 100) if _total > 0 else 0
                _bar_width = min(_util_pct, 100)
                _rpnl_color = "#5eb88a" if _rpnl >= 0 else "#d45d5d"
                _rpnl_sign = "+" if _rpnl >= 0 else ""
                _leverage_badge = (
                    f'<span style="font-size: 10px; background: rgba(155,142,196,0.1); color: #9b8ec4; '
                    f'padding: 2px 6px; border-radius: 4px; margin-left: 6px; font-weight: 600;">'
                    f'{leverage}\u00d7 leverage</span>'
                ) if leverage > 1 else ""
                # Build metric cells
                _cells = (
                    f'<div style="flex: 1; min-width: 90px;">'
                    f'<div style="font-size: 11px; color: rgba(255,255,255,0.45); text-transform: uppercase; letter-spacing: 0.5px;">Available</div>'
                    f'<div style="font-size: 18px; font-weight: 800; color: #5eb88a;">\u20b9{_avail:,.0f}</div></div>'
                )
                if leverage > 1:
                    _cells += (
                        f'<div style="flex: 1; min-width: 90px;">'
                        f'<div style="font-size: 11px; color: rgba(255,255,255,0.45); text-transform: uppercase; letter-spacing: 0.5px;">Buying Power</div>'
                        f'<div style="font-size: 18px; font-weight: 800; color: #9b8ec4;">\u20b9{_buying_power:,.0f}</div></div>'
                    )
                _cells += (
                    f'<div style="flex: 1; min-width: 90px;">'
                    f'<div style="font-size: 11px; color: rgba(255,255,255,0.45); text-transform: uppercase; letter-spacing: 0.5px;">Deployed</div>'
                    f'<div style="font-size: 18px; font-weight: 800; color: #d4a054;">\u20b9{_deployed:,.0f}</div></div>'
                    f'<div style="flex: 1; min-width: 90px;">'
                    f'<div style="font-size: 11px; color: rgba(255,255,255,0.45); text-transform: uppercase; letter-spacing: 0.5px;">Realized P&L</div>'
                    f'<div style="font-size: 18px; font-weight: 800; color: {_rpnl_color};">'
                    f'{_rpnl_sign}\u20b9{_rpnl:,.0f}</div></div>'
                )
                return (
                    f'<div style="background: #16181e;'
                    f' border: 1px solid {border_color}; border-radius: 10px;'
                    f' padding: 18px 20px; margin-bottom: 12px;">'
                    f'<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px;">'
                    f'<span style="font-size: 13px; font-weight: 700; color: {accent};'
                    f' text-transform: uppercase; letter-spacing: 1px;">{label}{_leverage_badge}</span>'
                    f'<span style="font-size: 12px; color: {_ret_color}; font-weight: 700;">'
                    f'{_ret_sign}{_return_pct:.1f}% return</span></div>'
                    f'<div style="display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 12px;">{_cells}</div>'
                    f'<div style="background: rgba(255,255,255,0.04); border-radius: 4px; height: 6px; overflow: hidden;">'
                    f'<div style="background: {accent}; width: {_bar_width}%; height: 100%; border-radius: 6px;'
                    f' transition: width 0.3s;"></div></div>'
                    f'<div style="font-size: 11px; color: rgba(255,255,255,0.22); margin-top: 4px;">'
                    f'{_util_pct:.0f}% deployed of \u20b9{_total:,.0f} capital</div></div>'
                )

            with _fc1:
                if _fund_swing["initial_capital"] > 0:
                    st.markdown(_fund_card("Swing", _fund_swing, "#d4a054", "rgba(212,160,84,0.15)"),
                                unsafe_allow_html=True)
            with _fc2:
                if _fund_scalp["initial_capital"] > 0:
                    st.markdown(_fund_card("Scalp", _fund_scalp, "#9b8ec4", "rgba(155,142,196,0.15)", leverage=5),
                                unsafe_allow_html=True)
        else:
            st.markdown("""<div style="background: rgba(212,160,84,0.06); border: 1px solid rgba(212,160,84,0.15);
                border-radius: 10px; padding: 12px 16px; margin-bottom: 12px; font-size: 13px; color: #d4a054;">
                Set your <b>Paper Trading Funds</b> in the sidebar to track capital usage, deployed amounts, and returns.
            </div>""", unsafe_allow_html=True)

        # -- Detailed stats row --
        if _pt_all_stats["total_trades"] > 0:
            _best = _pt_all_stats["best_trade"]
            _worst = _pt_all_stats["worst_trade"]
            _avg_pct = _pt_all_stats["avg_pnl_pct"]
            _avg_pct_c = "#5eb88a" if _avg_pct >= 0 else "#d45d5d"
            st.markdown(f"""<div style="display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 110px; text-align: center; padding: 12px; border-radius: 10px;
                    background: rgba(94,184,138,0.06); border: 1px solid rgba(94,184,138,0.12);">
                    <div style="font-size: 17px; font-weight: 800; color: #5eb88a;">+₹{_best:,.0f}</div>
                    <div style="font-size: 11px; color: rgba(255,255,255,0.45); text-transform: uppercase; letter-spacing: 0.5px;">Best Trade</div>
                </div>
                <div style="flex: 1; min-width: 110px; text-align: center; padding: 12px; border-radius: 10px;
                    background: rgba(212,93,93,0.06); border: 1px solid rgba(212,93,93,0.12);">
                    <div style="font-size: 17px; font-weight: 800; color: #d45d5d;">₹{_worst:,.0f}</div>
                    <div style="font-size: 11px; color: rgba(255,255,255,0.45); text-transform: uppercase; letter-spacing: 0.5px;">Worst Trade</div>
                </div>
                <div style="flex: 1; min-width: 110px; text-align: center; padding: 12px; border-radius: 10px;
                    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05);">
                    <div style="font-size: 17px; font-weight: 800; color: {_avg_pct_c};">{"+" if _avg_pct >= 0 else ""}{_avg_pct:.2f}%</div>
                    <div style="font-size: 11px; color: rgba(255,255,255,0.45); text-transform: uppercase; letter-spacing: 0.5px;">Avg P&L %</div>
                </div>
                <div style="flex: 1; min-width: 110px; text-align: center; padding: 12px; border-radius: 10px;
                    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05);">
                    <div style="font-size: 17px; font-weight: 800; color: #e8e4de;">{_pt_all_stats['wins']}/{_pt_all_stats['losses']}</div>
                    <div style="font-size: 11px; color: rgba(255,255,255,0.45); text-transform: uppercase; letter-spacing: 0.5px;">Wins / Losses</div>
                </div>
            </div>""", unsafe_allow_html=True)

        # -- Action buttons row --
        _ptb1, _ptb2, _ptb3 = st.columns([3, 1, 1])
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
        def _level_badge(label, price_str, is_hit, hit_color="#5eb88a", pending_color="rgba(255,255,255,0.3)"):
            if is_hit:
                _rgb = "94,184,138" if hit_color == "#5eb88a" else "212,93,93"
                return (f"<span style='display:inline-block; padding:4px 12px; border-radius:8px; margin:3px 4px; "
                        f"font-size:12px; font-weight:700; letter-spacing:0.3px; "
                        f"background:rgba({_rgb},0.15); "
                        f"border:1px solid {hit_color}; color:{hit_color};'>"
                        f"{label} ₹{price_str}</span>")
            else:
                return (f"<span style='display:inline-block; padding:4px 12px; border-radius:8px; margin:3px 4px; "
                        f"font-size:12px; font-weight:600; letter-spacing:0.3px; "
                        f"background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.05); color:{pending_color};'>"
                        f"{label} ₹{price_str}</span>")

        # -- Open positions (PENDING + ACTIVE) --
        _open_all = get_open_trades(None)
        if not _open_all.empty:
            st.markdown("<div class='section-label'>Open Positions</div>",
                        unsafe_allow_html=True)
            for _, row in _open_all.iterrows():
                _is_long = row["direction"] in ("BUY", "LONG")
                _dir_color = "#5eb88a" if _is_long else "#d45d5d"
                _dir_icon = "▲" if _is_long else "▼"
                _type_badge = "SWING" if row["trade_type"] == "swing" else "SCALP"
                _badge_bg = "rgba(212,160,84,0.1)" if row["trade_type"] == "swing" else "rgba(155,142,196,0.1)"
                _badge_border = "rgba(212,160,84,0.2)" if row["trade_type"] == "swing" else "rgba(155,142,196,0.2)"
                _badge_color = "#d4a054" if row["trade_type"] == "swing" else "#9b8ec4"
                _opened_short = row["opened_at"][:16].replace(
                    "T", " ") if row["opened_at"] else ""
                _is_active = row.get("status") == "ACTIVE"
                _status_label = "ACTIVE" if _is_active else "PENDING"
                _status_color = "#5eb88a" if _is_active else "#d4a054"
                _status_rgb = "94,184,138" if _is_active else "212,160,84"

                # Read hit flags (with fallback for old data)
                _entry_hit = bool(row.get("entry_hit", 0))
                _t1_hit = bool(row.get("t1_hit", 0))
                _t2_hit = bool(row.get("t2_hit", 0))
                _t3_hit = bool(row.get("t3_hit", 0))
                _sl_hit = bool(row.get("sl_hit", 0))

                # Build badges
                _badges = _level_badge(
                    "ENTRY", f"{row['entry_price']:,.2f}", _entry_hit, "#d4a054")
                _badges += _level_badge("T1",
                                        f"{row['target_1']:,.2f}", _t1_hit)
                if row.get("target_2"):
                    _badges += _level_badge("T2",
                                            f"{row['target_2']:,.2f}", _t2_hit)
                if row.get("target_3"):
                    _badges += _level_badge("T3",
                                            f"{row['target_3']:,.2f}", _t3_hit)
                _badges += _level_badge("SL", f"{row['stop_loss']:,.2f}",
                                        _sl_hit, "#d45d5d", "#d45d5d" if not _sl_hit else "#d45d5d")

                st.markdown(f"""<div style="
                    background: #16181e;
                    border: 1px solid rgba(255,255,255,0.05); border-left: 3px solid {_dir_color};
                    border-radius: 10px; padding: 18px 22px; margin-bottom: 10px;">
                    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span style="font-size: 20px; font-weight: 800; color: {_dir_color};">{_dir_icon}</span>
                            <span style="font-size: 17px; font-weight: 800; color: #e8e4de;">
                                {row['ticker'].replace('.NS', '')}</span>
                            <span style="display: inline-block; font-size: 10px; font-weight: 700;
                                text-transform: uppercase; letter-spacing: 0.8px; padding: 3px 10px;
                                border-radius: 20px; background: {_badge_bg}; border: 1px solid {_badge_border};
                                color: {_badge_color};">{_type_badge}</span>
                            <span style="display: inline-block; font-size: 10px; font-weight: 700;
                                text-transform: uppercase; letter-spacing: 0.8px; padding: 3px 10px;
                                border-radius: 20px; background: rgba({_status_rgb},0.1);
                                border: 1px solid {_status_color}; color: {_status_color};">{_status_label}</span>
                        </div>
                        <div style="font-size: 13px; color: rgba(255,255,255,0.22);">
                            Qty {row['quantity']} | ₹{row.get('capital_used', 0) or 0:,.0f} capital | Exit: {row.get('exit_target', 'T1') or 'T1'} | {_opened_short}</div>
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
                            hist = _yf_retry(
                                lambda: stock.history(period="1d"))
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
                                hist = _yf_retry(
                                    lambda: stock.history(period="1d"))
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
            st.markdown("<div class='section-label'>Trade History</div>",
                        unsafe_allow_html=True)

            # Outcome summary bar
            _t_hit = len(_hist_all[_hist_all["status"].str.startswith("HIT")])
            _t_stop = len(_hist_all[_hist_all["status"] == "STOPPED_OUT"])
            _t_exp = len(_hist_all[_hist_all["status"] == "EXPIRED"])
            _t_closed = len(_hist_all[_hist_all["status"] == "CLOSED"])

            st.markdown(f"""<div style="display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 100px; text-align: center; padding: 12px; border-radius: 10px;
                    background: rgba(94,184,138,0.08); border: 1px solid rgba(94,184,138,0.15);">
                    <div style="font-size: 20px; font-weight: 800; color: #5eb88a;">{_t_hit}</div>
                    <div style="font-size: 11px; color: rgba(255,255,255,0.45); text-transform: uppercase; letter-spacing: 0.6px;">Targets Hit</div>
                </div>
                <div style="flex: 1; min-width: 100px; text-align: center; padding: 12px; border-radius: 10px;
                    background: rgba(212,93,93,0.08); border: 1px solid rgba(212,93,93,0.15);">
                    <div style="font-size: 20px; font-weight: 800; color: #d45d5d;">{_t_stop}</div>
                    <div style="font-size: 11px; color: rgba(255,255,255,0.45); text-transform: uppercase; letter-spacing: 0.6px;">Stopped Out</div>
                </div>
                <div style="flex: 1; min-width: 100px; text-align: center; padding: 12px; border-radius: 10px;
                    background: rgba(212,160,84,0.08); border: 1px solid rgba(212,160,84,0.15);">
                    <div style="font-size: 20px; font-weight: 800; color: #d4a054;">{_t_exp}</div>
                    <div style="font-size: 11px; color: rgba(255,255,255,0.45); text-transform: uppercase; letter-spacing: 0.6px;">Expired</div>
                </div>
                <div style="flex: 1; min-width: 100px; text-align: center; padding: 12px; border-radius: 10px;
                    background: rgba(212,160,84,0.08); border: 1px solid rgba(212,160,84,0.15);">
                    <div style="font-size: 20px; font-weight: 800; color: #d4a054;">{_t_closed}</div>
                    <div style="font-size: 11px; color: rgba(255,255,255,0.45); text-transform: uppercase; letter-spacing: 0.6px;">Manually Closed</div>
                </div>
            </div>""", unsafe_allow_html=True)

            # Trade history cards with badges
            for _, row in _hist_all.iterrows():
                _status = row["status"]
                if _status.startswith("HIT"):
                    _outcome_color = "#5eb88a"
                    _outcome_bg = "rgba(5,150,105,0.06)"
                    _outcome_border = "rgba(5,150,105,0.25)"
                    _outcome_label = _status.replace("_", " ")
                elif _status == "STOPPED_OUT":
                    _outcome_color = "#d45d5d"
                    _outcome_bg = "rgba(220,38,38,0.06)"
                    _outcome_border = "rgba(220,38,38,0.25)"
                    _outcome_label = "STOPPED OUT"
                elif _status == "EXPIRED":
                    _outcome_color = "#d4a054"
                    _outcome_bg = "rgba(245,158,11,0.06)"
                    _outcome_border = "rgba(245,158,11,0.25)"
                    _outcome_label = "EXPIRED"
                else:
                    _outcome_color = "#d4a054"
                    _outcome_bg = "rgba(212,160,84,0.06)"
                    _outcome_border = "rgba(212,160,84,0.2)"
                    _outcome_label = "CLOSED"

                _gross_pnl = row.get("pnl", 0) or 0
                _charges_val = row.get("charges", 0) or 0
                _pnl_val = round(_gross_pnl - _charges_val, 2)  # net P&L
                _pnl_pct = row.get("pnl_pct", 0) or 0
                _pnl_c = "#5eb88a" if _pnl_val >= 0 else "#d45d5d"
                _pnl_s = "+" if _pnl_val >= 0 else ""
                _type_badge = "SWING" if row["trade_type"] == "swing" else "SCALP"
                _badge_color = "#d4a054" if row["trade_type"] == "swing" else "#9b8ec4"
                _opened = (row.get("opened_at") or "")[:10]
                _closed = (row.get("closed_at") or "")[:10]

                # Build hit badges for history
                _entry_hit = bool(row.get("entry_hit", 0))
                _t1_hit = bool(row.get("t1_hit", 0))
                _t2_hit = bool(row.get("t2_hit", 0))
                _t3_hit = bool(row.get("t3_hit", 0))
                _sl_hit = bool(row.get("sl_hit", 0))

                _exit_tgt = row.get("exit_target", "T1") or "T1"
                _h_badges = _level_badge(
                    "ENTRY", f"{row['entry_price']:,.2f}", _entry_hit, "#d4a054")
                _t1_label = "T1 (exit)" if _exit_tgt == "T1" else "T1"
                _h_badges += _level_badge(_t1_label,
                                          f"{row['target_1']:,.2f}", _t1_hit)
                if row.get("target_2"):
                    _t2_label = "T2 (exit)" if _exit_tgt == "T2" else "T2"
                    _h_badges += _level_badge(_t2_label,
                                              f"{row['target_2']:,.2f}", _t2_hit)
                if row.get("target_3"):
                    _t3_label = "T3 (exit)" if _exit_tgt == "T3" else "T3"
                    _h_badges += _level_badge(_t3_label,
                                              f"{row['target_3']:,.2f}", _t3_hit)
                _h_badges += _level_badge("SL",
                                          f"{row['stop_loss']:,.2f}", _sl_hit, "#d45d5d")
                # Peak price info — show what the stock reached
                _peak = row.get("highest_price", 0) or 0
                _trough = row.get("lowest_price", 0) or 0

                st.markdown(f"""<div style="
                    background: {_outcome_bg}; border: 1px solid {_outcome_border};
                    border-radius: 12px; padding: 16px 20px; margin-bottom: 8px;">
                    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span style="font-size: 16px; font-weight: 800; color: #e8e4de;">
                                {row['ticker'].replace('.NS', '')}</span>
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
                            {f'<div style="font-size: 11px; color: rgba(255,255,255,0.3); margin-top: 2px;">charges ₹{_charges_val:.2f}</div>' if _charges_val else ''}
                        </div>
                    </div>
                    <div style="margin-top: 10px; display: flex; flex-wrap: wrap; gap: 2px;">
                        {_h_badges}
                    </div>
                    <div style="display: flex; gap: 20px; margin-top: 8px; font-size: 13px; color: rgba(255,255,255,0.22); flex-wrap: wrap;">
                        <span>Qty {row['quantity']}</span>
                        <span>\u20b9{row.get('capital_used', 0) or 0:,.0f} capital</span>
                        <span>Exit: {_exit_tgt}</span>
                        <span>Peak: \u20b9{_peak:,.2f}</span>
                        <span>{_opened} \u2192 {_closed}</span>
                    </div>
                </div>""", unsafe_allow_html=True)

            # Swing vs Scalp comparison
            if _pt_swing_stats["total_trades"] > 0 and _pt_scalp_stats["total_trades"] > 0:
                st.markdown(
                    "<div class='section-label'>Swing vs Scalp Comparison</div>", unsafe_allow_html=True)
                _cmp1, _cmp2 = st.columns(2)
                with _cmp1:
                    _sw_pnl_c = "#5eb88a" if _pt_swing_stats["total_pnl"] >= 0 else "#d45d5d"
                    _sw_roi_line = ""
                    if _fund_swing["initial_capital"] > 0:
                        _sw_roi = _fund_swing["realized_pnl"] / \
                            _fund_swing["initial_capital"] * 100
                        _sw_roi_c = "#5eb88a" if _sw_roi >= 0 else "#d45d5d"
                        _sw_roi_line = f'<div style="font-size: 13px; color: {_sw_roi_c}; margin-top: 4px;">ROI: {"+" if _sw_roi >= 0 else ""}{_sw_roi:.1f}%</div>'
                    st.markdown(f"""<div style="background: #16181e;
                        border: 1px solid rgba(212,160,84,0.15); border-radius: 10px; padding: 20px; text-align: center;">
                        <div style="font-size: 13px; font-weight: 700; color: #d4a054; text-transform: uppercase;
                            letter-spacing: 1px;">Swing Trading</div>
                        <div style="font-size: 28px; font-weight: 800; color: {_sw_pnl_c}; margin: 8px 0;">
                            ₹{_pt_swing_stats['total_pnl']:,.2f}</div>
                        <div style="font-size: 13px; color: rgba(255,255,255,0.45);">
                            {_pt_swing_stats['total_trades']} trades | {_pt_swing_stats['win_rate']}% win rate</div>
                        {_sw_roi_line}
                    </div>""", unsafe_allow_html=True)
                with _cmp2:
                    _sc_pnl_c = "#5eb88a" if _pt_scalp_stats["total_pnl"] >= 0 else "#d45d5d"
                    _sc_roi_line = ""
                    if _fund_scalp["initial_capital"] > 0:
                        _sc_roi = _fund_scalp["realized_pnl"] / \
                            _fund_scalp["initial_capital"] * 100
                        _sc_roi_c = "#5eb88a" if _sc_roi >= 0 else "#d45d5d"
                        _sc_roi_line = f'<div style="font-size: 13px; color: {_sc_roi_c}; margin-top: 4px;">ROI: {"+" if _sc_roi >= 0 else ""}{_sc_roi:.1f}%</div>'
                    st.markdown(f"""<div style="background: #16181e;
                        border: 1px solid rgba(155,142,196,0.15); border-radius: 10px; padding: 20px; text-align: center;">
                        <div style="font-size: 13px; font-weight: 700; color: #9b8ec4; text-transform: uppercase;
                            letter-spacing: 1px;">Scalp Trading</div>
                        <div style="font-size: 28px; font-weight: 800; color: {_sc_pnl_c}; margin: 8px 0;">
                            ₹{_pt_scalp_stats['total_pnl']:,.2f}</div>
                        <div style="font-size: 13px; color: rgba(255,255,255,0.45);">
                            {_pt_scalp_stats['total_trades']} trades | {_pt_scalp_stats['win_rate']}% win rate</div>
                        {_sc_roi_line}
                    </div>""", unsafe_allow_html=True)


# ========================================================================
# TAB 5: SENTIMENT
# ========================================================================
if st.session_state["page"] == "analysis":
    with tab_sentiment:
        _render_stock_header()
        _render_indicators_row()
        help_box("Get AI-powered sentiment analysis for this stock. <b>Step 1:</b> Fetch news & generate a prompt. "
                 "<b>Step 2:</b> Copy the prompt and paste it into ChatGPT or Claude. "
                 "<b>Step 3:</b> Paste the AI's response back here to see the analysis.")

        sent_col1, sent_col2 = st.columns([3, 1])
        with sent_col1:
            fetch_news_btn = st.button(
                "📰 Step 1: Fetch News & Generate Prompt", type="primary", use_container_width=True)
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
                st.warning(
                    "No recent news found for this stock. Try a different stock.")
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
            st.markdown(
                f"<div class='prompt-box'>{st.session_state['sentiment_prompt']}</div>", unsafe_allow_html=True)
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
                    sentiment = _parse_sentiment_response(
                        ai_response.strip(), news)
                    st.session_state["sentiment_result"] = sentiment
                else:
                    st.warning("Please paste the AI response first.")

        if "sentiment_result" in st.session_state:
            sentiment = st.session_state["sentiment_result"]
            st.markdown("---")
            st.markdown("#### Sentiment Analysis Result")
            sc = {"Bullish": "action-up", "Bearish": "action-down",
                  "Neutral": "action-sideways"}
            action_card(f"Sentiment: {sentiment['overall']}", f"Score: {sentiment['score']:+.2f}",
                        sc.get(sentiment["overall"], "action-sideways"))
            if sentiment.get("summary"):
                verdict_box(f"<b>Summary:</b> {sentiment['summary']}", "good" if sentiment["overall"]
                            == "Bullish" else "bad" if sentiment["overall"] == "Bearish" else "neutral")
            for d in sentiment.get("details", []):
                ds = "good" if d["sentiment"] == "Bullish" else "bad" if d["sentiment"] == "Bearish" else "neutral"
                de = {"Bullish": "🟢", "Bearish": "🔴",
                      "Neutral": "🟡"}.get(d["sentiment"], "⚪")
                verdict_box(
                    f"{de} <b>{d['sentiment']}</b> — {d['headline']}", ds)


# ========================================================================
# TAB 7: SCREENER
# ========================================================================
if st.session_state["page"] == "home":
    with tab_screener:
        st.markdown("""<div style="margin-bottom: 12px;">
            <span style="font-size: 20px; font-weight: 700; color: #e8e4de;">Screener</span>
            <div style="font-size: 12px; color: rgba(255,255,255,0.35); margin-top: 2px;">Scan &amp; find the best trades across top 25 stocks</div>
        </div>""", unsafe_allow_html=True)
        help_box("<b>Stock Screener</b> scans multiple stocks using a <b>multi-factor approach</b> inspired by BlackRock's SAE methodology. "
                 "It combines <b>Technical Signals</b> (RSI, MACD, ADX, EMAs) with <b>Fundamental Momentum</b> (earnings growth, volume trends) "
                 "to find the best stocks for swing trading and scalping.")

        scr_c1, scr_c2 = st.columns([2, 1])
        with scr_c1:
            scan_scope = st.selectbox("Scan scope", ["All Popular Stocks"] + list(SECTOR_STOCKS.keys()),
                                      help="Scan all 25 popular stocks or pick a sector")
        with scr_c2:
            st.markdown('<div style="height: 24px;"></div>', unsafe_allow_html=True)
            scan_btn = st.button("🔍 Scan Now", type="primary",
                                 use_container_width=True)

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
                    progress.progress((idx * 2 + 1) / total_steps,
                                      text=f"Swing scan: {t.replace('.NS', '')}...")
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
                            bullish_patterns = sum(
                                1 for p in t_patterns if "bullish" in p["implication"].lower())
                            sentiment_score += min(0.5,
                                                   bullish_patterns * 0.25)
                        # Price in lower half of BB = more upside potential
                        if "BB_Upper" in t_df.columns and "BB_Lower" in t_df.columns:
                            bb_range = t_latest["BB_Upper"] - \
                                t_latest["BB_Lower"]
                            if bb_range > 0:
                                bb_position = (
                                    t_latest["Close"] - t_latest["BB_Lower"]) / bb_range
                                if bb_position < 0.4:
                                    sentiment_score += 0.5

                        composite = (0.4 * tech_score + 0.2 * momentum_score +
                                     0.2 * quality_score + 0.2 * sentiment_score)

                        # Display-only scores
                        _d_tech = compute_tech_score(t_latest, t_signal.confidence)
                        try:
                            _d_fund_data = get_stock_fundamentals(t)
                        except Exception:
                            _d_fund_data = {}
                        _d_fund = compute_fundamental_score(_d_fund_data)

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
                            "tech_score": _d_tech,
                            "fund_score": _d_fund,
                        })
                    except Exception:
                        pass

                    # --- Scalp scan ---
                    progress.progress((idx * 2 + 2) / total_steps,
                                      text=f"Scalp scan: {t.replace('.NS', '')}...")
                    try:
                        intra_df = fetch_intraday_data(
                            t, interval="5m", period="5d")
                        intra_df = add_scalping_indicators(intra_df)
                        s_signal = generate_scalp_signal(intra_df)
                        s_micro = get_market_microstructure(intra_df)
                        s_levels = get_scalping_levels(intra_df)
                        # Reuse fund score from swing scan if available
                        _sc_fund = 0
                        for _sr in scan_results:
                            if _sr["ticker"] == t:
                                _sc_fund = _sr.get("fund_score", 0)
                                break
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
                            "scalp_score": int(min(100, s_micro["scalpability_score"] * 0.6 + s_signal.confidence * 40)),
                            "tech_score": compute_tech_score(intra_df.iloc[-1], s_signal.confidence),
                            "fund_score": _sc_fund,
                        })
                    except Exception:
                        pass

                scan_results.sort(
                    key=lambda x: x["composite_score"], reverse=True)
                # Sort scalp results by combined score: scalpability (weight 0.6) + confidence (weight 0.4)
                # This ensures high-scalpability stocks rank higher than just high-confidence ones
                scalp_results.sort(
                    key=lambda x: 0.6 *
                    (x["scalpability"] / 100) + 0.4 * x["confidence"],
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
                    <div class="qs-delta" style="color: rgba(255,255,255,0.45);">stocks</div>
                </div>
                <div class="quick-stat">
                    <div class="qs-label">Buy Signals</div>
                    <div class="qs-value" style="color: #5eb88a;">{len(buys)}</div>
                    <div class="qs-delta" style="color: #5eb88a;">{'🟢' * min(len(buys), 5)}</div>
                </div>
                <div class="quick-stat">
                    <div class="qs-label">Sell Signals</div>
                    <div class="qs-value" style="color: #d45d5d;">{len(sells)}</div>
                    <div class="qs-delta" style="color: #d45d5d;">{'🔴' * min(len(sells), 5)}</div>
                </div>
                <div class="quick-stat">
                    <div class="qs-label">Hold</div>
                    <div class="qs-value" style="color: #d4a054;">{len(holds)}</div>
                    <div class="qs-delta" style="color: #d4a054;">{'🟡' * min(len(holds), 5)}</div>
                </div>
            </div>""", unsafe_allow_html=True)

            # ── TOP SWING PICKS (Row-based) ──
            st.markdown("#### Top Swing Picks")
            if buys:
                for i, pick in enumerate(buys[:5]):
                    p_setup = pick.get("setup")
                    _badge = '<span class="badge-best">Best</span>' if i == 0 else f'<span class="badge-rank">#{i+1}</span>'
                    _entry_str = f"₹{p_setup.entry_price:,.2f}" if p_setup else "—"
                    _sl_str = f"₹{p_setup.stop_loss:,.2f}" if p_setup else "—"
                    _t1_str = f"₹{p_setup.target_1:,.2f}" if p_setup else "—"
                    _rr_str = f"1:{p_setup.risk_reward_1}" if p_setup else "—"
                    _score_pct = pick.get('composite_score', 0) * 100
                    _score_color = "#5eb88a" if _score_pct >= 60 else "#d4a054" if _score_pct >= 40 else "rgba(255,255,255,0.45)"
                    _ts = pick.get("tech_score", 0)
                    _fs = pick.get("fund_score", 0)
                    _ts_color = "#5eb88a" if _ts >= 60 else "#d4a054" if _ts >= 40 else "#d45d5d"
                    _fs_color = "#5eb88a" if _fs >= 60 else "#d4a054" if _fs >= 40 else "#d45d5d"

                    # Signal badge
                    _sig_bg = "rgba(94,184,138,0.12)" if pick["signal"] == "BUY" else "rgba(212,93,93,0.12)"
                    _sig_border = "rgba(94,184,138,0.25)" if pick["signal"] == "BUY" else "rgba(212,93,93,0.25)"
                    _sig_color = "#5eb88a" if pick["signal"] == "BUY" else "#d45d5d"

                    # Profit & charges
                    _sw_profit_html = ""
                    if p_setup:
                        _sw_ch = calc_angel_one_charges(p_setup.entry_price, p_setup.target_1, 1, "delivery")
                        _sw_np = _sw_ch["net_profit"]
                        _sw_np_color = "#5eb88a" if _sw_np > 0 else "#d45d5d"
                        _sw_np_sign = "+" if _sw_np > 0 else ""
                        _sw_profit_html = f"""<div style="text-align: center;">
                                <div style="font-size: 13px; font-weight: 700; color: {_sw_np_color}; font-family: var(--mono);">{_sw_np_sign}₹{_sw_np:,.2f}</div>
                                <div style="font-size: 12px; color: rgba(255,255,255,0.22);">charges ₹{_sw_ch['total_charges']:.2f}</div>
                            </div>"""

                    _sw_card_col, _sw_act_col = st.columns([8, 1], vertical_alignment="center")
                    with _sw_card_col:
                        st.markdown(f"""<div style="background: #16181e; border: 1px solid rgba(255,255,255,0.05); border-radius: 10px;
                            padding: 26px 18px; margin-bottom: 4px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
                            <div style="display: flex; align-items: center; gap: 12px; min-width: 200px;">
                                {_badge}
                                <div>
                                    <div style="display: flex; align-items: center; gap: 8px;">
                                        <span style="font-size: 16px; font-weight: 700; color: #e8e4de;">{pick['name']}</span>
                                        <span style="background: {_sig_bg}; border: 1px solid {_sig_border}; color: {_sig_color};
                                               font-size: 12px; font-weight: 700; padding: 2px 8px; border-radius: 5px; letter-spacing: 0.04em;">{pick['signal']}</span>
                                    </div>
                                    <div style="font-size: 14px; color: rgba(255,255,255,0.45); font-family: 'JetBrains Mono', monospace;">₹{pick['price']:,.2f}</div>
                                </div>
                            </div>
                            <div style="display: flex; gap: 16px; align-items: center; flex-wrap: wrap;">
                                <div style="text-align: center;">
                                    <div style="font-size: 12px; color: rgba(255,255,255,0.30); text-transform: uppercase; letter-spacing: 0.5px;">Entry</div>
                                    <div style="font-size: 14px; font-weight: 600; color: #e8e4de; font-family: 'JetBrains Mono', monospace;">{_entry_str}</div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="font-size: 12px; color: rgba(255,255,255,0.30); text-transform: uppercase; letter-spacing: 0.5px;">Target</div>
                                    <div style="font-size: 14px; font-weight: 600; color: #5eb88a; font-family: 'JetBrains Mono', monospace;">{_t1_str}</div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="font-size: 12px; color: rgba(255,255,255,0.30); text-transform: uppercase; letter-spacing: 0.5px;">Stop Loss</div>
                                    <div style="font-size: 14px; font-weight: 600; color: #d45d5d; font-family: 'JetBrains Mono', monospace;">{_sl_str}</div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="font-size: 12px; color: rgba(255,255,255,0.30); text-transform: uppercase; letter-spacing: 0.5px;">R:R</div>
                                    <div style="font-size: 14px; font-weight: 600; color: #e8e4de;">{_rr_str}</div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="font-size: 12px; color: rgba(255,255,255,0.30); text-transform: uppercase; letter-spacing: 0.5px;">Tech</div>
                                    <div style="font-size: 14px; font-weight: 700; color: {_ts_color};">{_ts}/100</div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="font-size: 12px; color: rgba(255,255,255,0.30); text-transform: uppercase; letter-spacing: 0.5px;">Fund</div>
                                    <div style="font-size: 14px; font-weight: 700; color: {_fs_color};">{_fs}/100</div>
                                </div>
                                {_sw_profit_html}
                            </div>
                        </div>""", unsafe_allow_html=True)
                    with _sw_act_col:
                        st.markdown('<div class="scr-btn-col">', unsafe_allow_html=True)
                        if st.button("Paper Trade", key=f"scr_sw_pt_{i}", type="primary", use_container_width=True):
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
                                        reasons=", ".join(
                                            pick["reasons"]) if pick["reasons"] else "",
                                    )
                                    st.success(f"Trade #{_tid} placed!")
                                except ValueError as e:
                                    st.error(str(e))
                            else:
                                st.warning("No trade setup available")
                        st.markdown(
                            f'<a class="scr-btn scr-btn-analyse" href="?stock={pick["ticker"]}&tab=swing" target="_blank">'
                            f'Analyze</a>',
                            unsafe_allow_html=True,
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No BUY signals found.")

            st.markdown("---")
            # ── TOP SCALPING PICKS (Row-based) ──
            scalp_longs_all = [r for r in scalp_res if r["signal"]
                               == "LONG"] if scalp_res else []
            if scalp_longs_all:
                st.markdown("#### Top Scalping Picks")
                for i, pick in enumerate(scalp_longs_all[:5]):
                    _badge = '<span class="badge-best">Best</span>' if i == 0 else f'<span class="badge-rank">#{i+1}</span>'
                    _scalp_color = "#5eb88a" if pick['scalpability'] >= 65 else "#d4a054" if pick[
                        'scalpability'] >= 40 else "#d45d5d"

                    # Signal badge
                    _sc_sig_bg = "rgba(94,184,138,0.12)" if pick["signal"] == "LONG" else "rgba(212,93,93,0.12)"
                    _sc_sig_border = "rgba(94,184,138,0.25)" if pick["signal"] == "LONG" else "rgba(212,93,93,0.25)"
                    _sc_sig_color = "#5eb88a" if pick["signal"] == "LONG" else "#d45d5d"
                    _sc_sig_label = "LONG" if pick["signal"] == "LONG" else "SHORT"

                    # Profit & charges
                    _sc_ch = calc_angel_one_charges(pick["entry"], pick["target_1"], 1)
                    _sc_np = _sc_ch["net_profit"]
                    _sc_np_color = "#5eb88a" if _sc_np > 0 else "#d45d5d"
                    _sc_np_sign = "+" if _sc_np > 0 else ""
                    _sc_ss = pick.get("scalp_score", 0)
                    _sc_ts = pick.get("tech_score", 0)
                    _sc_fs = pick.get("fund_score", 0)
                    _sc_ss_color = "#5eb88a" if _sc_ss >= 60 else "#d4a054" if _sc_ss >= 40 else "#d45d5d"
                    _sc_ts_color = "#5eb88a" if _sc_ts >= 60 else "#d4a054" if _sc_ts >= 40 else "#d45d5d"
                    _sc_fs_color = "#5eb88a" if _sc_fs >= 60 else "#d4a054" if _sc_fs >= 40 else "#d45d5d"

                    _sc_card_col, _sc_act_col = st.columns([8, 1], vertical_alignment="center")
                    with _sc_card_col:
                        st.markdown(f"""<div style="background: #16181e; border: 1px solid rgba(255,255,255,0.05); border-radius: 10px;
                            padding: 26px 18px; margin-bottom: 4px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
                            <div style="display: flex; align-items: center; gap: 12px; min-width: 200px;">
                                {_badge}
                                <div>
                                    <div style="display: flex; align-items: center; gap: 8px;">
                                        <span style="font-size: 16px; font-weight: 700; color: #e8e4de;">{pick['name']}</span>
                                        <span style="background: {_sc_sig_bg}; border: 1px solid {_sc_sig_border}; color: {_sc_sig_color};
                                               font-size: 12px; font-weight: 700; padding: 2px 8px; border-radius: 5px; letter-spacing: 0.04em;">{_sc_sig_label}</span>
                                    </div>
                                    <div style="font-size: 14px; color: rgba(255,255,255,0.45); font-family: 'JetBrains Mono', monospace;">₹{pick['price']:,.2f}</div>
                                </div>
                            </div>
                            <div style="display: flex; gap: 16px; align-items: center; flex-wrap: wrap;">
                                <div style="text-align: center;">
                                    <div style="font-size: 12px; color: rgba(255,255,255,0.30); text-transform: uppercase; letter-spacing: 0.5px;">Entry</div>
                                    <div style="font-size: 14px; font-weight: 600; color: #e8e4de; font-family: 'JetBrains Mono', monospace;">₹{pick['entry']:,.2f}</div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="font-size: 12px; color: rgba(255,255,255,0.30); text-transform: uppercase; letter-spacing: 0.5px;">Target</div>
                                    <div style="font-size: 14px; font-weight: 600; color: #5eb88a; font-family: 'JetBrains Mono', monospace;">₹{pick['target_1']:,.2f}</div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="font-size: 12px; color: rgba(255,255,255,0.30); text-transform: uppercase; letter-spacing: 0.5px;">Stop Loss</div>
                                    <div style="font-size: 14px; font-weight: 600; color: #d45d5d; font-family: 'JetBrains Mono', monospace;">₹{pick['stop_loss']:,.2f}</div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="font-size: 12px; color: rgba(255,255,255,0.30); text-transform: uppercase; letter-spacing: 0.5px;">R:R</div>
                                    <div style="font-size: 14px; font-weight: 600; color: #e8e4de;">{pick['risk_reward']}</div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="font-size: 12px; color: rgba(255,255,255,0.30); text-transform: uppercase; letter-spacing: 0.5px;">Scalp</div>
                                    <div style="font-size: 14px; font-weight: 700; color: {_sc_ss_color};">{_sc_ss}/100</div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="font-size: 12px; color: rgba(255,255,255,0.30); text-transform: uppercase; letter-spacing: 0.5px;">Tech</div>
                                    <div style="font-size: 14px; font-weight: 700; color: {_sc_ts_color};">{_sc_ts}/100</div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="font-size: 12px; color: rgba(255,255,255,0.30); text-transform: uppercase; letter-spacing: 0.5px;">Fund</div>
                                    <div style="font-size: 14px; font-weight: 700; color: {_sc_fs_color};">{_sc_fs}/100</div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="font-size: 13px; font-weight: 700; color: {_sc_np_color}; font-family: var(--mono);">{_sc_np_sign}₹{_sc_np:,.2f}</div>
                                    <div style="font-size: 12px; color: rgba(255,255,255,0.30);">charges ₹{_sc_ch['total_charges']:.2f}</div>
                                </div>
                            </div>
                        </div>""", unsafe_allow_html=True)
                    with _sc_act_col:
                        st.markdown('<div class="scr-btn-col">', unsafe_allow_html=True)
                        if st.button("Paper Trade", key=f"scr_sc_pt_{i}", type="primary", use_container_width=True):
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
                                    reasons=", ".join(
                                        pick["reasons"]) if pick["reasons"] else "",
                                )
                                st.success(f"Trade #{_tid} placed!")
                            except ValueError as e:
                                st.error(str(e))
                        st.markdown(
                            f'<a class="scr-btn scr-btn-analyse" href="?stock={pick["ticker"]}&tab=scalp" target="_blank">'
                            f'Analyze</a>',
                            unsafe_allow_html=True,
                        )
                        st.markdown('</div>', unsafe_allow_html=True)

            if sells:
                st.markdown("---")
                st.markdown("#### Stocks to Avoid")
                for pick in sells[:3]:
                    st.markdown(f"""<div style="background: rgba(212,93,93,0.05); border: 1px solid rgba(212,93,93,0.1); border-radius: 10px;
                        padding: 12px 18px; margin-bottom: 6px; display: flex; align-items: center; justify-content: space-between; gap: 12px;">
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <div style="font-size: 14px; font-weight: 700; color: #d45d5d;">{pick['name']}</div>
                            <div style="font-size: 14px; color: rgba(255,255,255,0.45); font-family: 'JetBrains Mono', monospace;">₹{pick['price']:,.2f}</div>
                        </div>
                        <div style="font-size: 11px; font-weight: 700; color: #d45d5d; text-transform: uppercase;">SELL — {pick['strength']}</div>
                    </div>""", unsafe_allow_html=True)

            # ── FULL RESULTS TABLES ──
            st.markdown("---")
            st.markdown("#### All Swing Signals")
            help_box("<b>Score</b> = Multi-factor composite (Technical 40% + Momentum 20% + Quality 20% + Sentiment 20%). "
                     "Inspired by BlackRock's SAE methodology.")
            table_rows = []
            for r in results:
                sig_emoji = {"BUY": "🟢", "SELL": "🔴",
                             "HOLD": "🟡"}.get(r["signal"], "⚪")
                r_setup = r.get("setup")
                if r_setup and r["signal"] == "BUY":
                    r_ch = calc_angel_one_charges(
                        r_setup.entry_price, r_setup.target_1, 1, "delivery")
                    entry_str = f"{r_setup.entry_price:,.2f}"
                    t1_str = f"{r_setup.target_1:,.2f}"
                    sl_str = f"{r_setup.stop_loss:,.2f}"
                    profit_str = f"{r_ch['net_profit']:.2f}"
                elif r_setup and r["signal"] == "SELL":
                    r_ch = calc_angel_one_charges(
                        r_setup.target_1, r_setup.entry_price, 1, "delivery")
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
            st.dataframe(pd.DataFrame(table_rows),
                         use_container_width=True, hide_index=True)

            if scalp_res:
                st.markdown("#### All Scalping Signals")
                scalp_table = []
                for r in scalp_res:
                    sig_emoji = {"LONG": "🟢", "SHORT": "🔴",
                                 "NO_TRADE": "🟡"}.get(r["signal"], "⚪")
                    if r["signal"] == "LONG":
                        sc_ch = calc_angel_one_charges(
                            r['entry'], r['target_1'], 1)
                        profit_str = f"{sc_ch['net_profit']:.2f}"
                    elif r["signal"] == "SHORT":
                        sc_ch = calc_angel_one_charges(
                            r['target_1'], r['entry'], 1)
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
                st.dataframe(pd.DataFrame(scalp_table),
                             use_container_width=True, hide_index=True)

            swing_summary = f"{len(buys)} BUY | {len(sells)} SELL | {len(holds)} HOLD"
            scalp_longs_count = len(
                [r for r in scalp_res if r["signal"] == "LONG"])
            scalp_shorts_count = len(
                [r for r in scalp_res if r["signal"] == "SHORT"])
            scalp_no_count = len(
                [r for r in scalp_res if r["signal"] == "NO_TRADE"])
            scalp_summary = f"{scalp_longs_count} LONG | {scalp_shorts_count} SHORT | {scalp_no_count} NO_TRADE"
            st.markdown(
                f"**Swing:** {swing_summary} out of {len(results)} stocks | **Scalp:** {scalp_summary} out of {len(scalp_res)} stocks")


# ========================================================================
# TAB 8: FUNDAMENTALS
# ========================================================================
if st.session_state["page"] == "analysis":
    with tab_fund:
        _render_stock_header()
        _render_indicators_row()
        help_box("<b>Fundamentals</b> show the financial health of a company — how profitable it is, "
                 "how much debt it has, and whether the stock price is cheap or expensive relative to earnings.")

        with st.spinner("Loading fundamentals..."):
            try:
                fund = get_stock_fundamentals(ticker)
            except Exception as e:
                st.error(f"Could not load fundamentals: {e}")
                fund = None

        if fund:
            pe = fund.get("trailing_pe")
            fpe = fund.get("forward_pe")
            pb = fund.get("price_to_book")
            ev_ebitda = fund.get("ev_to_ebitda")
            peg = fund.get("peg_ratio")
            ps = fund.get("price_to_sales")
            mcap = fund.get("market_cap")
            ev = fund.get("enterprise_value")
            pm = fund.get("profit_margin")
            om = fund.get("operating_margin")
            gm = fund.get("gross_margin")
            roe = fund.get("roe")
            roa = fund.get("roa")
            de = fund.get("debt_to_equity")
            cr = fund.get("current_ratio")
            td = fund.get("total_debt")
            tc = fund.get("total_cash")
            dy = fund.get("dividend_yield")
            pr = fund.get("payout_ratio")
            rg = fund.get("revenue_growth")
            eg = fund.get("earnings_growth")
            eps = fund.get("eps")
            feps = fund.get("forward_eps")
            rev = fund.get("revenue")
            earn = fund.get("earnings")
            avg50 = fund.get("50d_avg")
            avg200 = fund.get("200d_avg")
            beta_val = fund.get("beta")
            _f_cur = fund.get("current_price", 0)

            # Helper to build a metric row inside a card
            def _fund_row(label, value, color="#e8e4de"):
                return f"""<div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.04);">
                    <span style="font-size: 12px; color: rgba(255,255,255,0.5);">{label}</span>
                    <span style="font-size: 13px; font-weight: 700; color: {color}; font-family: var(--mono);">{value}</span>
                </div>"""

            def _fund_card_header(title):
                return f'<div style="font-size: 10px; font-weight: 700; color: rgba(255,255,255,0.3); text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 10px;">{title}</div>'

            # --- 2-Column Layout: Left (Valuation + Profitability + Growth) | Right (Health + Dividends + Averages) ---
            fund_left, fund_right = st.columns([1, 1])

            with fund_left:
                # --- VALUATION CARD ---
                _val_rows = _fund_card_header("Valuation")
                _val_rows += _fund_row("P/E Ratio", f"{pe:.1f}" if pe else "N/A", "#d4a054" if pe and pe > 30 else "#5eb88a" if pe and pe < 15 else "#e8e4de")
                _val_rows += _fund_row("Forward P/E", f"{fpe:.1f}" if fpe else "N/A")
                _val_rows += _fund_row("Price/Book", f"{pb:.2f}" if pb else "N/A")
                _val_rows += _fund_row("EV/EBITDA", f"{ev_ebitda:.1f}" if ev_ebitda else "N/A")
                _val_rows += _fund_row("PEG Ratio", f"{peg:.2f}" if peg else "N/A")
                _val_rows += _fund_row("Price/Sales", f"{ps:.2f}" if ps else "N/A")
                _val_rows += _fund_row("Market Cap", f"₹{mcap/1e7:,.0f} Cr" if mcap else "N/A")
                _val_rows += _fund_row("Enterprise Value", f"₹{ev/1e7:,.0f} Cr" if ev else "N/A")
                st.markdown(f"<div class='setup-card' style='border-left: none; padding: 14px 16px;'>{_val_rows}</div>", unsafe_allow_html=True)

                if pe:
                    if pe < 15:
                        verdict_box(f"P/E of <b>{pe:.1f}</b> — relatively <b>cheap</b>. Could be undervalued.", "good")
                    elif pe < 30:
                        verdict_box(f"P/E of <b>{pe:.1f}</b> — <b>fairly valued</b>. Reasonable for a quality company.", "neutral")
                    else:
                        verdict_box(f"P/E of <b>{pe:.1f}</b> — <b>expensive</b>. Market expects high growth.", "bad")

                # --- PROFITABILITY CARD ---
                _prof_rows = _fund_card_header("Profitability")
                _prof_rows += _fund_row("Profit Margin", f"{pm*100:.1f}%" if pm else "N/A", "#5eb88a" if pm and pm > 0.15 else "#d45d5d" if pm and pm < 0 else "#e8e4de")
                _prof_rows += _fund_row("Operating Margin", f"{om*100:.1f}%" if om else "N/A", "#5eb88a" if om and om > 0.15 else "#d45d5d" if om and om < 0 else "#e8e4de")
                _prof_rows += _fund_row("Gross Margin", f"{gm*100:.1f}%" if gm else "N/A")
                _prof_rows += _fund_row("ROE", f"{roe*100:.1f}%" if roe else "N/A", "#5eb88a" if roe and roe > 0.15 else "#d45d5d" if roe and roe < 0.10 else "#d4a054")
                _prof_rows += _fund_row("ROA", f"{roa*100:.1f}%" if roa else "N/A")
                st.markdown(f"<div class='setup-card' style='border-left: none; padding: 14px 16px; margin-top: 8px;'>{_prof_rows}</div>", unsafe_allow_html=True)

                if roe:
                    if roe > 0.20:
                        verdict_box(f"ROE of <b>{roe*100:.1f}%</b> — <b>excellent</b>. Company generates strong returns.", "good")
                    elif roe > 0.10:
                        verdict_box(f"ROE of <b>{roe*100:.1f}%</b> — <b>decent</b>. Adequate returns for shareholders.", "neutral")
                    else:
                        verdict_box(f"ROE of <b>{roe*100:.1f}%</b> — <b>weak</b>. Low returns on invested capital.", "bad")

                # --- GROWTH CARD ---
                _grow_rows = _fund_card_header("Growth")
                _grow_rows += _fund_row("Revenue Growth", f"{rg*100:.1f}%" if rg else "N/A", "#5eb88a" if rg and rg > 0 else "#d45d5d" if rg and rg < 0 else "#e8e4de")
                _grow_rows += _fund_row("Earnings Growth", f"{eg*100:.1f}%" if eg else "N/A", "#5eb88a" if eg and eg > 0 else "#d45d5d" if eg and eg < 0 else "#e8e4de")
                _grow_rows += _fund_row("Revenue", f"₹{rev/1e7:,.0f} Cr" if rev else "N/A")
                _grow_rows += _fund_row("Net Income", f"₹{earn/1e7:,.0f} Cr" if earn else "N/A")
                st.markdown(f"<div class='setup-card' style='border-left: none; padding: 14px 16px; margin-top: 8px;'>{_grow_rows}</div>", unsafe_allow_html=True)

                if rg and eg:
                    if rg > 0.15 and eg > 0.15:
                        verdict_box(f"Revenue growing <b>{rg*100:.1f}%</b> and earnings <b>{eg*100:.1f}%</b> — <b>strong growth</b>.", "good")
                    elif rg > 0 and eg > 0:
                        verdict_box(f"Revenue growing <b>{rg*100:.1f}%</b> and earnings <b>{eg*100:.1f}%</b> — <b>steady growth</b>.", "neutral")
                    else:
                        verdict_box("Growth is <b>declining</b> — company may be struggling.", "bad")

            with fund_right:
                # --- FINANCIAL HEALTH CARD ---
                _health_rows = _fund_card_header("Financial Health")
                _health_rows += _fund_row("Debt/Equity", f"{de:.1f}" if de else "N/A", "#5eb88a" if de is not None and de < 50 else "#d45d5d" if de is not None and de > 150 else "#d4a054")
                _health_rows += _fund_row("Current Ratio", f"{cr:.2f}" if cr else "N/A", "#5eb88a" if cr and cr > 1.5 else "#d45d5d" if cr and cr < 1 else "#d4a054")
                _health_rows += _fund_row("Total Debt", f"₹{td/1e7:,.0f} Cr" if td else "N/A", "#d45d5d")
                _health_rows += _fund_row("Total Cash", f"₹{tc/1e7:,.0f} Cr" if tc else "N/A", "#5eb88a")
                st.markdown(f"<div class='setup-card' style='border-left: none; padding: 14px 16px;'>{_health_rows}</div>", unsafe_allow_html=True)

                if de is not None:
                    if de < 50:
                        verdict_box(f"Debt/Equity of <b>{de:.1f}</b> — <b>low debt</b>. Financially strong.", "good")
                    elif de < 150:
                        verdict_box(f"Debt/Equity of <b>{de:.1f}</b> — <b>moderate debt</b>. Manageable but watch it.", "neutral")
                    else:
                        verdict_box(f"Debt/Equity of <b>{de:.1f}</b> — <b>high debt</b>. Risk if earnings drop.", "bad")

                # --- DIVIDENDS CARD ---
                _div_rows = _fund_card_header("Dividends & Earnings")
                _div_rows += _fund_row("Dividend Yield", f"{dy*100:.2f}%" if dy else "N/A", "#5eb88a" if dy and dy > 0.02 else "#e8e4de")
                _div_rows += _fund_row("Payout Ratio", f"{pr*100:.1f}%" if pr else "N/A")
                _div_rows += _fund_row("EPS", f"₹{eps:.2f}" if eps else "N/A")
                _div_rows += _fund_row("Forward EPS", f"₹{feps:.2f}" if feps else "N/A")
                st.markdown(f"<div class='setup-card' style='border-left: none; padding: 14px 16px; margin-top: 8px;'>{_div_rows}</div>", unsafe_allow_html=True)

                # --- PRICE VS AVERAGES CARD ---
                _avg_rows = _fund_card_header("Price vs Averages")
                if avg50:
                    _a50_color = "#5eb88a" if _f_cur > avg50 else "#d45d5d"
                    _a50_label = "Above" if _f_cur > avg50 else "Below"
                    _avg_rows += _fund_row(f"50-Day Avg <span style='font-size:10px; color:{_a50_color};'>({_a50_label})</span>", f"₹{avg50:,.2f}", _a50_color)
                else:
                    _avg_rows += _fund_row("50-Day Avg", "N/A")
                if avg200:
                    _a200_color = "#5eb88a" if _f_cur > avg200 else "#d45d5d"
                    _a200_label = "Above" if _f_cur > avg200 else "Below"
                    _avg_rows += _fund_row(f"200-Day Avg <span style='font-size:10px; color:{_a200_color};'>({_a200_label})</span>", f"₹{avg200:,.2f}", _a200_color)
                else:
                    _avg_rows += _fund_row("200-Day Avg", "N/A")
                _avg_rows += _fund_row("Beta", f"{beta_val:.2f}" if beta_val else "N/A", "#5eb88a" if beta_val and beta_val < 1 else "#d45d5d" if beta_val and beta_val > 1.2 else "#d4a054")
                st.markdown(f"<div class='setup-card' style='border-left: none; padding: 14px 16px; margin-top: 8px;'>{_avg_rows}</div>", unsafe_allow_html=True)

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
if st.session_state["page"] == "analysis":
    with tab_sector:
        _render_stock_header()
        _render_indicators_row()
        help_box("<b>Sector Analysis</b> compares your selected stock against peers in the same sector. "
                 "See which stocks outperform, how correlated they are, and key metrics side-by-side.")

        sector = get_sector_for_stock(ticker)
        if not sector:
            st.warning(
                f"{ticker} is not in any predefined sector. Try a stock from the popular list.")
        else:
            peers = SECTOR_STOCKS[sector]

            # Sector header card
            _peer_badges = " ".join(
                f'<span style="background: {"rgba(201,184,156,0.12); border: 1px solid rgba(201,184,156,0.2); color: #c9b89c" if p == ticker else "rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.06); color: rgba(255,255,255,0.5)"}; '
                f'font-size: 11px; font-weight: 600; padding: 3px 10px; border-radius: 6px; font-family: var(--mono);">'
                f'{p.replace(".NS", "")}</span>'
                for p in peers
            )
            st.markdown(f"""<div class='setup-card' style='border-left: none; padding: 14px 16px;'>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <div style="font-size: 10px; font-weight: 700; color: rgba(255,255,255,0.3); text-transform: uppercase; letter-spacing: 1.2px;">Sector</div>
                    <span style="font-size: 14px; font-weight: 700; color: #d4a054;">{sector}</span>
                </div>
                <div style="display: flex; gap: 6px; flex-wrap: wrap;">{_peer_badges}</div>
            </div>""", unsafe_allow_html=True)

            with st.spinner("Loading sector data..."):
                try:
                    prices_df = fetch_multiple_stocks(peers, period_years=1)
                except Exception as e:
                    st.error(f"Error loading sector data: {e}")
                    prices_df = pd.DataFrame()

            if not prices_df.empty:
                returns_1y = ((prices_df.iloc[-1] / prices_df.iloc[0]) - 1) * 100

                # --- 2-Column: Left (Perf Chart) | Right (Returns + Peer Cards) ---
                sec_left, sec_right = st.columns([3, 2])

                with sec_left:
                    # --- Normalized Performance Chart ---
                    st.markdown(f"""<div style="font-size: 10px; font-weight: 700; color: rgba(255,255,255,0.3); text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 8px; margin-top: 12px;">Price Performance (1Y, Indexed to 100)</div>""", unsafe_allow_html=True)

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

                with sec_right:
                    # --- Returns Card ---
                    returns_sorted = returns_1y.sort_values(ascending=False)
                    _ret_rows = '<div style="font-size: 10px; font-weight: 700; color: rgba(255,255,255,0.3); text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 10px;">1-Year Returns</div>'
                    for idx, val in returns_sorted.items():
                        _r_color = "#5eb88a" if val >= 0 else "#d45d5d"
                        _r_highlight = "background: rgba(201,184,156,0.06);" if idx == ticker else ""
                        _r_sign = "+" if val >= 0 else ""
                        _ret_rows += f"""<div style="display: flex; justify-content: space-between; align-items: center; padding: 7px 8px; border-bottom: 1px solid rgba(255,255,255,0.04); border-radius: 4px; {_r_highlight}">
                            <span style="font-size: 12px; color: {'#c9b89c' if idx == ticker else 'rgba(255,255,255,0.5)'}; font-weight: {'700' if idx == ticker else '400'}; font-family: var(--mono);">{idx.replace('.NS', '')}</span>
                            <span style="font-size: 13px; font-weight: 700; color: {_r_color}; font-family: var(--mono);">{_r_sign}{val:.1f}%</span>
                        </div>"""
                    st.markdown(f"<div class='setup-card' style='border-left: none; padding: 14px 16px; margin-top: 12px;'>{_ret_rows}</div>", unsafe_allow_html=True)

                # --- Key Metrics Comparison (Full Width) ---
                st.markdown(f"""<div style="font-size: 10px; font-weight: 700; color: rgba(255,255,255,0.3); text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 8px; margin-top: 16px;">Key Metrics Comparison</div>""", unsafe_allow_html=True)

                # Build HTML table
                metrics_data = []
                for p in peers:
                    if p in prices_df.columns:
                        try:
                            p_fund = get_stock_fundamentals(p)
                            p_ret = ((prices_df[p].iloc[-1] / prices_df[p].iloc[0]) - 1) * 100
                            metrics_data.append({"ticker": p, "fund": p_fund, "ret": p_ret})
                        except Exception:
                            continue

                if metrics_data:
                    _th_style = 'style="font-size: 10px; color: rgba(255,255,255,0.35); text-transform: uppercase; letter-spacing: 0.5px; padding: 8px 10px; text-align: right; border-bottom: 1px solid rgba(255,255,255,0.06);"'
                    _th_style_l = 'style="font-size: 10px; color: rgba(255,255,255,0.35); text-transform: uppercase; letter-spacing: 0.5px; padding: 8px 10px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.06);"'
                    _tbl = f'<table style="width: 100%; border-collapse: collapse;">'
                    _tbl += f'<tr><th {_th_style_l}>Stock</th><th {_th_style}>Price</th><th {_th_style}>Mkt Cap</th><th {_th_style}>P/E</th><th {_th_style}>ROE</th><th {_th_style}>Div Yield</th><th {_th_style}>1Y Return</th><th {_th_style}>Beta</th></tr>'

                    for md in metrics_data:
                        _p = md["ticker"]
                        _pf = md["fund"]
                        _pr = md["ret"]
                        _is_cur = _p == ticker
                        _row_bg = "background: rgba(201,184,156,0.04);" if _is_cur else ""
                        _name_color = "#c9b89c" if _is_cur else "rgba(255,255,255,0.6)"
                        _name_weight = "700" if _is_cur else "500"
                        _td_style = f'style="font-size: 12px; font-family: var(--mono); font-weight: 600; color: #e8e4de; padding: 8px 10px; text-align: right; border-bottom: 1px solid rgba(255,255,255,0.03);"'
                        _td_style_l = f'style="font-size: 12px; font-family: var(--mono); font-weight: {_name_weight}; color: {_name_color}; padding: 8px 10px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.03);"'
                        _ret_color = "#5eb88a" if _pr >= 0 else "#d45d5d"

                        _pe_val = _pf.get("trailing_pe")
                        _roe_val = _pf.get("roe")
                        _mcap_val = _pf.get("market_cap")
                        _dy_val = _pf.get("dividend_yield")
                        _beta_val = _pf.get("beta")
                        _mcap_str = f"₹{_mcap_val/1e7:,.0f} Cr" if _mcap_val else "N/A"
                        _pe_str = f"{_pe_val:.1f}" if _pe_val else "N/A"
                        _roe_str = f"{_roe_val*100:.1f}%" if _roe_val else "N/A"
                        _dy_str = f"{_dy_val*100:.2f}%" if _dy_val else "N/A"
                        _beta_str = f"{_beta_val:.2f}" if _beta_val else "N/A"
                        _pe_color = "#5eb88a" if _pe_val and _pe_val < 20 else "#d45d5d" if _pe_val and _pe_val > 40 else "#e8e4de"
                        _roe_color = "#5eb88a" if _roe_val and _roe_val > 0.15 else "#d45d5d" if _roe_val and _roe_val < 0.10 else "#e8e4de"
                        _tbl += f'<tr style="{_row_bg}">'
                        _tbl += f'<td {_td_style_l}>{_p.replace(".NS", "")}</td>'
                        _tbl += f'<td {_td_style}>₹{_pf.get("current_price", 0):,.2f}</td>'
                        _tbl += f'<td {_td_style}>{_mcap_str}</td>'
                        _tbl += f'<td style="font-size: 12px; font-family: var(--mono); font-weight: 600; color: {_pe_color}; padding: 8px 10px; text-align: right; border-bottom: 1px solid rgba(255,255,255,0.03);">{_pe_str}</td>'
                        _tbl += f'<td style="font-size: 12px; font-family: var(--mono); font-weight: 600; color: {_roe_color}; padding: 8px 10px; text-align: right; border-bottom: 1px solid rgba(255,255,255,0.03);">{_roe_str}</td>'
                        _tbl += f'<td {_td_style}>{_dy_str}</td>'
                        _tbl += f'<td style="font-size: 12px; font-family: var(--mono); font-weight: 600; color: {_ret_color}; padding: 8px 10px; text-align: right; border-bottom: 1px solid rgba(255,255,255,0.03);">{"+" if _pr >= 0 else ""}{_pr:.1f}%</td>'
                        _tbl += f'<td {_td_style}>{_beta_str}</td>'
                        _tbl += '</tr>'
                    _tbl += '</table>'
                    st.markdown(f"<div class='setup-card' style='border-left: none; padding: 10px 8px; overflow-x: auto;'>{_tbl}</div>", unsafe_allow_html=True)

                # --- Correlation + Risk in 2 columns ---
                sec_bot_left, sec_bot_right = st.columns([1, 1])

                with sec_bot_left:
                    st.markdown(f"""<div style="font-size: 10px; font-weight: 700; color: rgba(255,255,255,0.3); text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 8px; margin-top: 16px;">Correlation Matrix</div>""", unsafe_allow_html=True)
                    corr_matrix = calculate_correlation_matrix(prices_df)
                    corr_labels = [t.replace(".NS", "") for t in corr_matrix.columns]
                    fig_corr = go.Figure(go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_labels, y=corr_labels,
                        colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
                        text=np.round(corr_matrix.values, 2),
                        texttemplate="%{text}",
                        textfont=dict(size=11),
                    ))
                    fig_corr.update_layout(height=380, **CHART_LAYOUT)
                    st.plotly_chart(fig_corr, use_container_width=True)

                with sec_bot_right:
                    # --- Risk & Return Card ---
                    st.markdown(f"""<div style="font-size: 10px; font-weight: 700; color: rgba(255,255,255,0.3); text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 8px; margin-top: 16px;">Risk & Return Metrics</div>""", unsafe_allow_html=True)
                    _risk_th = 'style="font-size: 10px; color: rgba(255,255,255,0.35); text-transform: uppercase; letter-spacing: 0.5px; padding: 7px 8px; text-align: right; border-bottom: 1px solid rgba(255,255,255,0.06);"'
                    _risk_th_l = 'style="font-size: 10px; color: rgba(255,255,255,0.35); text-transform: uppercase; letter-spacing: 0.5px; padding: 7px 8px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.06);"'
                    _risk_tbl = f'<table style="width: 100%; border-collapse: collapse;">'
                    _risk_tbl += f'<tr><th {_risk_th_l}>Stock</th><th {_risk_th}>Sharpe</th><th {_risk_th}>Sortino</th><th {_risk_th}>Ann. Ret</th><th {_risk_th}>Ann. Vol</th><th {_risk_th}>Max DD</th></tr>'
                    for p in peers:
                        if p in prices_df.columns:
                            try:
                                p_returns = prices_df[p].pct_change().dropna()
                                ratios = calculate_sharpe_ratio(p_returns)
                                _is_cur = p == ticker
                                _row_bg = "background: rgba(201,184,156,0.04);" if _is_cur else ""
                                _n_color = "#c9b89c" if _is_cur else "rgba(255,255,255,0.6)"
                                _n_weight = "700" if _is_cur else "500"
                                _rtd = f'style="font-size: 12px; font-family: var(--mono); font-weight: 600; color: #e8e4de; padding: 7px 8px; text-align: right; border-bottom: 1px solid rgba(255,255,255,0.03);"'
                                _rtd_l = f'style="font-size: 12px; font-family: var(--mono); font-weight: {_n_weight}; color: {_n_color}; padding: 7px 8px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.03);"'
                                _sharpe_color = "#5eb88a" if ratios['sharpe_ratio'] > 1 else "#d45d5d" if ratios['sharpe_ratio'] < 0 else "#e8e4de"
                                _dd_color = "#d45d5d" if ratios['max_drawdown'] < -20 else "#d4a054" if ratios['max_drawdown'] < -10 else "#5eb88a"
                                _risk_tbl += f'<tr style="{_row_bg}">'
                                _risk_tbl += f'<td {_rtd_l}>{p.replace(".NS", "")}</td>'
                                _risk_tbl += f'<td style="font-size: 12px; font-family: var(--mono); font-weight: 600; color: {_sharpe_color}; padding: 7px 8px; text-align: right; border-bottom: 1px solid rgba(255,255,255,0.03);">{ratios["sharpe_ratio"]:.2f}</td>'
                                _risk_tbl += f'<td {_rtd}>{ratios["sortino_ratio"]:.2f}</td>'
                                _risk_tbl += f'<td {_rtd}>{ratios["annualized_return"]:.1f}%</td>'
                                _risk_tbl += f'<td {_rtd}>{ratios["annualized_volatility"]:.1f}%</td>'
                                _risk_tbl += f'<td style="font-size: 12px; font-family: var(--mono); font-weight: 600; color: {_dd_color}; padding: 7px 8px; text-align: right; border-bottom: 1px solid rgba(255,255,255,0.03);">{ratios["max_drawdown"]:.1f}%</td>'
                                _risk_tbl += '</tr>'
                            except Exception:
                                continue
                    _risk_tbl += '</table>'
                    st.markdown(f"<div class='setup-card' style='border-left: none; padding: 10px 8px; overflow-x: auto;'>{_risk_tbl}</div>", unsafe_allow_html=True)


# ============================================================
# TAB 10: ACCURACY TRACKING
# ============================================================
if st.session_state["page"] == "home":
    with tab_accuracy:
        st.markdown("""<div style="margin-bottom: 12px;">
            <span style="font-size: 20px; font-weight: 700; color: #e8e4de;">Accuracy</span>
            <div style="font-size: 12px; color: rgba(255,255,255,0.35); margin-top: 2px;">Review prediction accuracy &amp; model performance</div>
        </div>""", unsafe_allow_html=True)
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
            acc_col1, acc_col2 = st.columns([1, 2])
            with acc_col1:
                acc_ticker_options = ["All Tickers"] + tracked
                acc_ticker = st.selectbox(
                    "Filter by ticker", acc_ticker_options, key="acc_ticker")
                filter_ticker = None if acc_ticker == "All Tickers" else acc_ticker
            with acc_col2:
                st.markdown("<div style='margin-top: 24px;'></div>",
                            unsafe_allow_html=True)
                if st.button("🔄 Validate Now", help="Fetch actual prices for pending predictions"):
                    with st.spinner("Validating predictions..."):
                        val_result = validate_pending_predictions(
                            filter_ticker)
                    st.success(
                        f"Validated {val_result['validated_count']} predictions")
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
                verdict_box(
                    f"Strong prediction accuracy ({metrics['direction_accuracy_pct']}% direction correct)", "bullish")
            elif metrics["direction_accuracy_pct"] >= 50:
                verdict_box(
                    f"Moderate accuracy ({metrics['direction_accuracy_pct']}% direction correct)", "neutral")
            elif metrics["validated_count"] > 0:
                verdict_box(
                    f"Low accuracy ({metrics['direction_accuracy_pct']}% direction correct) — consider relearning", "bearish")

            # Prediction history table
            history_df = get_prediction_history(filter_ticker, limit=100)

            if not history_df.empty:
                st.markdown("### Prediction History")

                # Charts for validated predictions
                validated_df = history_df[history_df["actual_price"].notna()].copy(
                )

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
                            conf_df = validated_df[validated_df["confidence_upper"].notna(
                            )]
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
                            validated_df = validated_df.sort_values(
                                "target_date")
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
                            fig_roll.update_layout(
                                height=350, yaxis_title="Accuracy %", **CHART_LAYOUT)
                            st.plotly_chart(fig_roll, use_container_width=True)
                        else:
                            st.info(
                                "Need at least 5 validated predictions for rolling accuracy chart.")

                    # Per-model comparison
                    if "lstm_price" in validated_df.columns and "xgb_price" in validated_df.columns:
                        model_df = validated_df[validated_df["lstm_price"].notna(
                        ) & validated_df["xgb_price"].notna()]
                        if not model_df.empty:
                            with st.expander("📊 LSTM vs XGBoost Comparison"):
                                lstm_err = (
                                    model_df["lstm_price"] - model_df["actual_price"]).abs() / model_df["actual_price"] * 100
                                xgb_err = (
                                    model_df["xgb_price"] - model_df["actual_price"]).abs() / model_df["actual_price"] * 100
                                ens_err = (
                                    model_df["predicted_price"] - model_df["actual_price"]).abs() / model_df["actual_price"] * 100

                                mc1, mc2, mc3 = st.columns(3)
                                mc1.metric("LSTM Avg Error",
                                           f"{lstm_err.mean():.2f}%")
                                mc2.metric("XGBoost Avg Error",
                                           f"{xgb_err.mean():.2f}%")
                                mc3.metric("Ensemble Avg Error",
                                           f"{ens_err.mean():.2f}%")

                                better_model = "XGBoost" if xgb_err.mean() < lstm_err.mean() else "LSTM"
                                verdict_box(
                                    f"{better_model} has been more accurate recently", "neutral")

                # Display history table
                display_df = history_df.copy()
                display_df["Status"] = display_df["actual_price"].apply(
                    lambda x: "✅ Validated" if pd.notna(x) else "⏳ Pending"
                )
                display_df["Direction"] = display_df["direction_correct"].apply(
                    lambda x: "✅ Correct" if x == 1 else (
                        "❌ Wrong" if x == 0 else "—")
                )
                show_cols = ["ticker", "prediction_date", "target_date", "predicted_price",
                             "actual_price", "error_pct", "Direction", "Status"]
                available_cols = [
                    c for c in show_cols if c in display_df.columns]
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

            relearn_ticker = st.selectbox(
                "Select ticker to relearn", tracked, key="relearn_ticker")

            if relearn_ticker:
                version = get_current_model_version(relearn_ticker)
                can_relearn, reason = should_relearn(relearn_ticker)

                rl1, rl2, rl3 = st.columns(3)
                rl1.metric("Model Version", version if version >
                           0 else "Initial")
                rl2.metric("Adaptive XGB Weight",
                           f"{compute_adaptive_weights(relearn_ticker):.2f}")

                relearn_metrics = get_accuracy_metrics(relearn_ticker)
                rl3.metric("Direction Accuracy",
                           f"{relearn_metrics['direction_accuracy_pct']}%")

                if can_relearn:
                    verdict_box(f"Relearning recommended: {reason}", "bearish")
                    if st.button(f"🔄 Relearn {relearn_ticker}", type="primary"):
                        progress = st.empty()
                        status = st.empty()

                        def relearn_progress(msg, pct):
                            progress.progress(pct, text=msg)
                        with st.spinner("Relearning..."):
                            result = relearn_models(
                                relearn_ticker, progress_callback=relearn_progress)
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
                            result = relearn_models(
                                relearn_ticker, progress_callback=force_progress)
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
    <div style="margin-top: 8px; font-size: 11px; color: rgba(255,255,255,0.2);">
        Built with Streamlit &bull; LSTM + XGBoost AI Models &bull; Data from Yahoo Finance
    </div>
</div>""", unsafe_allow_html=True)
