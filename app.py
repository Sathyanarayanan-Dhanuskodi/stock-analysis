import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from src.data_fetcher import (
    POPULAR_INDIAN_STOCKS, SECTOR_STOCKS, fetch_stock_data, get_stock_info,
    get_stock_news, fetch_multiple_stocks, get_sector_peers,
    fetch_intraday_data, get_stock_fundamentals, scan_swing_opportunities,
    get_sector_for_stock,
)
from src.hedge_trading import calculate_correlation_matrix, calculate_sharpe_ratio
from src.feature_engineering import add_technical_indicators
from src.predictor import train_and_predict, predict_with_saved_model
from src.sentiment import is_sentiment_available, analyze_sentiment
from src.model import load_saved_model
from src.prediction_tracker import (
    init_db, log_predictions, validate_pending_predictions,
    get_accuracy_metrics, get_prediction_history, get_tracked_tickers,
    should_relearn, relearn_models, compute_adaptive_weights,
    get_current_model_version,
)
from datetime import date
from src.swing_trading import (
    calculate_pivot_points, calculate_support_resistance,
    calculate_fibonacci_retracements, calculate_atr_stop_loss,
    generate_swing_signals, calculate_trade_setup, identify_swing_patterns,
)
from src.scalping import (
    add_scalping_indicators, generate_scalp_signal,
    get_scalping_levels, get_market_microstructure,
)

st.set_page_config(page_title="Indian Stock Predictor", page_icon="📈", layout="wide")

# ============================================================
# PREDICTION TRACKING — init DB & auto-validate on load
# ============================================================
init_db()

@st.cache_data(ttl=300)
def _auto_validate():
    return validate_pending_predictions()

_auto_validate()

# ============================================================
# CLEAN DARK THEME CSS
# ============================================================
st.markdown("""<style>
    /* === ACTION CARDS === */
    .action-card {
        padding: 22px; border-radius: 14px; text-align: center;
        font-size: 20px; font-weight: 700; color: #fff; margin: 12px 0;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .action-card span { display: block; font-size: 13px; font-weight: 400; opacity: 0.85; margin-top: 6px; }
    .action-buy, .action-long, .action-up   { background: linear-gradient(135deg, #059669, #10b981); }
    .action-sell, .action-short, .action-down { background: linear-gradient(135deg, #dc2626, #ef4444); }
    .action-hold, .action-sideways           { background: linear-gradient(135deg, #b45309, #d97706); }
    .action-notrade                          { background: linear-gradient(135deg, #374151, #4b5563); }

    /* === HELP BOX === */
    .help-box {
        background: rgba(0,212,255,0.05); border-left: 3px solid #00d4ff;
        padding: 14px 18px; border-radius: 0 10px 10px 0; margin: 10px 0;
        font-size: 13.5px; color: #c0c0d0; line-height: 1.65;
    }
    .help-box b { color: #e8e8f0; }

    /* === SETUP CARD === */
    .setup-card {
        background: #1a1a2e; padding: 18px; border-radius: 12px;
        border: 1px solid #2a2a40; margin: 8px 0;
        color: #c8c8d8; line-height: 1.85; font-size: 14.5px;
    }
    .setup-card b { color: #e8e8f0; }

    /* === VERDICT BOX === */
    .verdict-box {
        padding: 14px 18px; border-radius: 10px; margin: 10px 0;
        font-size: 14px; color: #c8c8d8; line-height: 1.6;
    }
    .verdict-box b { color: #e8e8f0; }
    .good    { background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.3); border-left: 3px solid #10b981; }
    .bad     { background: rgba(239,68,68,0.1);  border: 1px solid rgba(239,68,68,0.3);  border-left: 3px solid #ef4444; }
    .neutral { background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.3); border-left: 3px solid #f59e0b; }

    /* === CHECKLIST === */
    .checklist {
        background: #1a1a2e; padding: 16px 20px; border-radius: 12px; margin: 10px 0;
        color: #c8c8d8; line-height: 1.85; font-size: 14px; border: 1px solid #2a2a40;
    }
    .checklist b { color: #e8e8f0; }

    /* === METRIC CARDS === */
    [data-testid="stMetric"] {
        background: #1a1a2e; border: 1px solid #2a2a40; border-radius: 12px; padding: 16px;
    }
    [data-testid="stMetricLabel"] { color: #8888a0 !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.5px; }
    [data-testid="stMetricValue"] { color: #e8e8f0 !important; font-weight: 700 !important; }

    /* === TABS === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; background: #1a1a2e; border-radius: 10px; padding: 4px; border: 1px solid #2a2a40;
    }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 8px 16px; color: #8888a0; font-weight: 500; }
    .stTabs [aria-selected="true"] { background: #00d4ff !important; color: #0f0f1e !important; font-weight: 600; }

    /* === BUTTONS === */
    .stButton > button[kind="primary"] {
        background: #00d4ff !important; color: #0f0f1e !important;
        border: none !important; border-radius: 8px !important; font-weight: 600 !important;
    }
    .stButton > button[kind="primary"]:hover { background: #00bce0 !important; }

    /* === GRADIENT TITLE === */
    .app-title {
        text-align: center; padding: 8px 0 4px 0;
    }
    .app-title h1 {
        background: linear-gradient(135deg, #00d4ff, #a855f7);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 2rem; font-weight: 800; margin: 0;
    }
    .app-title p { color: #6b6b80; font-size: 13px; margin: 4px 0 0 0; }

    /* === FOOTER === */
    .footer-card {
        background: #1a1a2e; border: 1px solid #2a2a40; border-radius: 10px;
        padding: 14px 18px; margin-top: 20px; text-align: center;
        font-size: 12px; color: #6b6b80; line-height: 1.5;
    }
    .footer-card b { color: #d97706; }
</style>""", unsafe_allow_html=True)

# ============================================================
# TITLE
# ============================================================
st.markdown("""<div class="app-title">
    <h1>Indian Stock Market Predictor</h1>
    <p>AI-powered predictions &bull; swing &amp; scalping tools &bull; screener &bull; fundamentals &bull; sector analysis</p>
</div>""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Pick a Stock")
    stock_options = {f"{name} ({ticker})": ticker for ticker, name in POPULAR_INDIAN_STOCKS.items()}
    selected_label = st.selectbox("Select Stock", options=list(stock_options.keys()),
                                  help="Pick any popular Indian stock to analyze")
    ticker = stock_options[selected_label]

    custom_ticker = st.text_input("Or enter custom ticker", placeholder="e.g. ADANIENT.NS",
                                  help="Add .NS for NSE stocks, .BO for BSE stocks")
    if custom_ticker.strip():
        ticker = custom_ticker.strip().upper()

    st.divider()
    st.header("Settings")
    forecast_days = st.slider("How many days to predict?", min_value=1, max_value=30, value=7,
                              help="Shorter = more accurate. 7 days is a good balance.")
    epochs = st.slider("Training quality", min_value=10, max_value=150, value=80,
                       help="Higher = better model but slower training. 80 is recommended.")

    with st.expander("Advanced Model Settings"):
        st.caption("Leave these as default unless you know what you're doing")
        xgb_weight = st.slider("XGBoost Weight", 0.2, 0.8, 0.55, 0.05,
                               help="How much to trust XGBoost vs LSTM.")
        use_market_context = st.checkbox("Include market data (Nifty/VIX)", value=True,
                                         help="Uses overall market trends to improve predictions")

    with st.expander("Swing Trading Settings"):
        capital = st.number_input("Your capital (₹)", 10000, 10000000, 100000, step=10000,
                                  help="How much money you plan to invest")
        max_risk_pct = st.slider("Max risk per trade (%)", 0.5, 5.0, 2.0, 0.5,
                                 help="Maximum % of capital you're willing to lose. 2% is safe.")
        sr_lookback = st.slider("Analysis period (days)", 30, 180, 90,
                                help="How far back to look for support/resistance levels")
        pivot_method = st.selectbox("Pivot calculation", ["standard", "fibonacci", "camarilla", "woodie"])

    st.divider()
    if is_sentiment_available():
        st.success("Sentiment Analysis: Enabled")
    else:
        st.info("Add ANTHROPIC_API_KEY to .env for sentiment")

# ============================================================
# TABS
# ============================================================
tab_portfolio, tab_screener, tab_chart, tab_predict, tab_swing, tab_scalp, tab_fund, tab_sector, tab_sentiment, tab_accuracy = st.tabs([
    "💼 Portfolio", "🔍 Screener", "📊 Chart", "🤖 Predictions",
    "📈 Swing", "⚡ Scalping", "📋 Fundamentals", "🏭 Sector", "📰 Sentiment", "🎯 Accuracy",
])

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
# STOCK BANNER
# ============================================================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Price", f"₹{info['current_price']:,.2f}")
col2.metric("Sector", info["sector"])
price_change = raw_df["Close"].iloc[-1] - raw_df["Close"].iloc[-2]
pct_change = (price_change / raw_df["Close"].iloc[-2]) * 100
col3.metric("Day Change", f"₹{price_change:,.2f}", f"{pct_change:+.2f}%")
col4.metric("Market Cap", f"₹{info['market_cap'] / 1e7:,.0f} Cr" if info["market_cap"] else "N/A")

# ============================================================
# PLOTLY DEFAULTS
# ============================================================
CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0f0f1e",
    plot_bgcolor="#0f0f1e",
    font=dict(color="#c0c0d0"),
    xaxis=dict(gridcolor="#1e1e35", zerolinecolor="#1e1e35"),
    yaxis=dict(gridcolor="#1e1e35", zerolinecolor="#1e1e35"),
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
             "Lines show average prices over different time periods.")

    cc1, cc2, cc3 = st.columns(3)
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


# ========================================================================
# TAB 2: PREDICTIONS
# ========================================================================
with tab_predict:
    help_box(
        "Our AI uses <b>LSTM</b> (neural network) + <b>XGBoost</b> (decision trees), "
        "blended together for best accuracy. Click <b>Train & Predict</b> to start."
    )

    c_train, c_load = st.columns(2)
    with c_train:
        train_btn = st.button("Train & Predict", type="primary", use_container_width=True,
                              help="Trains on 5 years of data. Takes 2-5 minutes.")
    with c_load:
        saved = load_saved_model(ticker)
        load_btn = st.button("Use Saved Model" if saved else "No Saved Model",
                             disabled=saved is None, use_container_width=True)

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

    p1, p2 = st.columns(2)
    with p1:
        st.markdown(f"""<div class='setup-card'>
            <b>📍 Entry:</b> ₹{setup.entry_price:,.2f}<br>
            <b>🛑 Stop Loss:</b> ₹{setup.stop_loss:,.2f} <span style='color:#ef4444'>(−{abs(setup.entry_price - setup.stop_loss) / setup.entry_price * 100:.1f}%)</span><br>
            <b>💰 Position:</b> {setup.position_size_pct:.1f}% of ₹{capital:,}<br>
            <b>📊 Volatility:</b> {atr_data['volatility_regime']} (ATR: ₹{atr_data['current_atr']:.2f})
        </div>""", unsafe_allow_html=True)
    with p2:
        st.markdown(f"""<div class='setup-card'>
            <b>🎯 Target 1:</b> ₹{setup.target_1:,.2f} <span style='color:#10b981'>(R:R 1:{setup.risk_reward_1})</span><br>
            <b>🎯 Target 2:</b> ₹{setup.target_2:,.2f} <span style='color:#10b981'>(R:R 1:{setup.risk_reward_2})</span><br>
            <b>🎯 Target 3:</b> ₹{setup.target_3:,.2f} <span style='color:#10b981'>(R:R 1:{setup.risk_reward_3})</span>
        </div>""", unsafe_allow_html=True)

    risk_amount = setup.entry_price - setup.stop_loss if signal.signal == "BUY" else setup.stop_loss - setup.entry_price
    max_loss = abs(risk_amount) * (capital * setup.position_size_pct / 100) / setup.entry_price
    verdict_box(f"⚠️ <b>Max loss if stopped out:</b> ~₹{max_loss:,.0f} ({max_risk_pct:.1f}% of capital). Always use a stop loss!", "neutral")

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
        st.info("No significant chart patterns detected.")


# ========================================================================
# TAB 4: SCALPING
# ========================================================================
with tab_scalp:
    help_box("<b>Scalping</b> = ultra-short-term trading on 5-min candles. Buy and sell within minutes/hours. "
             "⚠️ Fast-paced and risky — always use stop losses!")

    sc1, sc2 = st.columns([2, 1])
    with sc1:
        scalp_period = st.selectbox("Data period", ["1d", "2d", "5d"], index=2)
    with sc2:
        scalp_btn = st.button("Load Intraday Data", type="primary", use_container_width=True)

    if scalp_btn:
        with st.spinner("Fetching 5-min candles..."):
            try:
                idf = fetch_intraday_data(ticker, interval="5m", period=scalp_period)
                idf = add_scalping_indicators(idf)
                st.session_state["scalp_data"] = idf
            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Intraday data only available during market hours (Mon-Fri, 9:15 AM - 3:30 PM IST)")

    if "scalp_data" in st.session_state:
        idf = st.session_state["scalp_data"]
        ss = generate_scalp_signal(idf)
        levels = get_scalping_levels(idf)
        micro = get_market_microstructure(idf)

        scalp_class = {"LONG": "action-long", "SHORT": "action-short", "NO_TRADE": "action-notrade"}
        scalp_label = {"LONG": "BUY (Long)", "SHORT": "SELL (Short)", "NO_TRADE": "NO TRADE — Wait"}
        scalp_advice = {"LONG": "Price likely going UP.", "SHORT": "Price likely going DOWN.", "NO_TRADE": "Signals weak — don't trade."}
        action_card(scalp_label.get(ss.signal, "NO TRADE"),
                    f"{ss.strength} | {ss.confidence*100:.0f}% — {scalp_advice.get(ss.signal, '')}", scalp_class.get(ss.signal, "action-notrade"))

        scalpability = micro["scalpability_score"]
        if scalpability >= 65:
            verdict_box(f"⚡ <b>Scalpability: {scalpability}/100 — Good!</b> {micro['volatility_regime']} volatility, {micro['volume_status'].lower()} volume, {micro['trend'].lower()}.", "good")
        elif scalpability >= 40:
            verdict_box(f"⚡ <b>Scalpability: {scalpability}/100 — Moderate.</b> Proceed with caution.", "neutral")
        else:
            verdict_box(f"⚡ <b>Scalpability: {scalpability}/100 — Poor.</b> Low volatility/volume. Consider waiting.", "bad")

        if ss.reasons:
            checklist([(r, any(w in r.lower() for w in ["bullish", "above", "bounce", "oversold", "positive", "up candle", "upward"])) for r in ss.reasons])

        st.markdown("#### Trade Setup")
        t1, t2 = st.columns(2)
        risk_ps = abs(ss.entry_price - ss.stop_loss)
        rew_ps = abs(ss.target_1 - ss.entry_price)
        with t1:
            st.markdown(f"""<div class='setup-card'>
                <b>📍 Entry:</b> ₹{ss.entry_price:,.2f}<br>
                <b>🛑 Stop Loss:</b> ₹{ss.stop_loss:,.2f} <span style='color:#ef4444'>(₹{risk_ps:.2f} risk)</span><br>
                <b>📊 R:R:</b> 1:{ss.risk_reward}
            </div>""", unsafe_allow_html=True)
        with t2:
            st.markdown(f"""<div class='setup-card'>
                <b>🎯 Target 1:</b> ₹{ss.target_1:,.2f} <span style='color:#10b981'>(+₹{rew_ps:.2f})</span><br>
                <b>🎯 Target 2:</b> ₹{ss.target_2:,.2f} <span style='color:#10b981'>(+₹{abs(ss.target_2 - ss.entry_price):.2f})</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("#### Key Levels")
        lv1, lv2, lv3, lv4 = st.columns(4)
        lv1.metric("VWAP", f"₹{levels['vwap']:,.2f}", "Above ↑" if idf["Close"].iloc[-1] > levels["vwap"] else "Below ↓")
        lv2.metric("Pivot", f"₹{levels['pivot']:,.2f}")
        lv3.metric("Today High", f"₹{levels['today_high']:,.2f}")
        lv4.metric("Today Low", f"₹{levels['today_low']:,.2f}")

        with st.expander("Camarilla Levels"):
            cam = st.columns(6)
            for i, (l, k) in enumerate(zip(["S3", "S2", "S1", "R1", "R2", "R3"],
                                            ["cam_s3", "cam_s2", "cam_s1", "cam_r1", "cam_r2", "cam_r3"])):
                cam[i].metric(l, f"₹{levels[k]:,.2f}")

        # 5-min chart
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
        fig_s.update_layout(height=550, xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02), **CHART_LAYOUT)
        fig_s.update_yaxes(gridcolor="#1e1e35", row=2, col=1)
        st.plotly_chart(fig_s, use_container_width=True)

        with st.expander("Market Conditions"):
            mi1, mi2, mi3, mi4 = st.columns(4)
            mi1.metric("ATR", f"₹{micro['atr']:.2f}")
            mi2.metric("Candle Size", f"{micro['avg_candle_range']:.3f}%")
            mi3.metric("Consecutive", f"{micro['consecutive_candles']} {micro['consecutive_direction']}")
            mi4.metric("Trend Slope", f"{micro['trend_slope']:.4f}")


# ========================================================================
# TAB 5: SENTIMENT
# ========================================================================
with tab_sentiment:
    if not is_sentiment_available():
        st.warning("Sentiment analysis requires an Anthropic API key.")
        help_box("Add ANTHROPIC_API_KEY to .env to enable AI news analysis.")
    else:
        help_box("AI reads recent news and determines if sentiment is <b>bullish</b>, <b>bearish</b>, or <b>neutral</b>.")
        if st.button("Analyze News", type="primary"):
            with st.spinner("Analyzing..."):
                news = get_stock_news(ticker)
                if not news:
                    st.info("No recent news found.")
                else:
                    sentiment = analyze_sentiment(news)
                    if sentiment:
                        sc = {"Bullish": "action-up", "Bearish": "action-down", "Neutral": "action-sideways"}
                        action_card(f"Sentiment: {sentiment['overall']}", f"Score: {sentiment['score']:+.2f}",
                                    sc.get(sentiment["overall"], "action-sideways"))
                        st.markdown(f"**Summary:** {sentiment['summary']}")
                        for d in sentiment["details"]:
                            ds = "good" if d["sentiment"] == "Bullish" else "bad" if d["sentiment"] == "Bearish" else "neutral"
                            de = {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "🟡"}.get(d["sentiment"], "⚪")
                            verdict_box(f"{de} <b>{d['sentiment']}</b> — {d['headline']}", ds)
                    else:
                        st.error("Could not analyze sentiment.")


# ========================================================================
# TAB 6: PORTFOLIO / WATCHLIST
# ========================================================================
with tab_portfolio:
    help_box("<b>Portfolio Watchlist</b> — Add stocks to track prices and calculate profit/loss. "
             "Data is stored in your browser session only (lost on refresh).")

    if "watchlist" not in st.session_state:
        st.session_state["watchlist"] = []

    st.markdown("#### Add Stock to Watchlist")
    wl_c1, wl_c2, wl_c3 = st.columns([2, 1, 1])
    with wl_c1:
        wl_options = {f"{name} ({t})": t for t, name in POPULAR_INDIAN_STOCKS.items()}
        wl_label = st.selectbox("Stock", list(wl_options.keys()), key="wl_stock")
        wl_ticker = wl_options[wl_label]
    with wl_c2:
        wl_buy_price = st.number_input("Buy price (₹)", min_value=0.0, value=0.0, step=1.0,
                                        help="Enter your buy price to track P&L. Leave 0 to skip.")
    with wl_c3:
        wl_qty = st.number_input("Quantity", min_value=0, value=0, step=1,
                                  help="Number of shares bought")

    if st.button("Add to Watchlist", type="primary"):
        already = any(w["ticker"] == wl_ticker for w in st.session_state["watchlist"])
        if already:
            st.warning("Stock already in watchlist.")
        else:
            st.session_state["watchlist"].append({
                "ticker": wl_ticker,
                "name": POPULAR_INDIAN_STOCKS.get(wl_ticker, wl_ticker),
                "buy_price": wl_buy_price if wl_buy_price > 0 else None,
                "qty": wl_qty if wl_qty > 0 else None,
            })
            st.success(f"Added {wl_ticker} to watchlist!")
            st.rerun()

    if st.session_state["watchlist"]:
        st.markdown("#### Your Watchlist")
        total_invested = 0.0
        total_current = 0.0

        wl_rows = []
        for item in st.session_state["watchlist"]:
            try:
                wl_info = get_stock_info(item["ticker"])
                cp = wl_info["current_price"]
                row = {
                    "Stock": f"{item['name']}",
                    "Ticker": item["ticker"],
                    "Price (₹)": f"{cp:,.2f}",
                }
                if item["buy_price"] and item["buy_price"] > 0:
                    pnl_pct = ((cp - item["buy_price"]) / item["buy_price"]) * 100
                    row["Buy (₹)"] = f"{item['buy_price']:,.2f}"
                    row["P&L %"] = f"{'🟢' if pnl_pct >= 0 else '🔴'} {pnl_pct:+.2f}%"
                    if item["qty"] and item["qty"] > 0:
                        invested = item["buy_price"] * item["qty"]
                        current = cp * item["qty"]
                        row["Qty"] = str(item["qty"])
                        row["Invested (₹)"] = f"{invested:,.0f}"
                        row["Current (₹)"] = f"{current:,.0f}"
                        row["P&L (₹)"] = f"{'🟢' if current >= invested else '🔴'} {current - invested:+,.0f}"
                        total_invested += invested
                        total_current += current
                else:
                    row["Buy (₹)"] = "—"
                    row["P&L %"] = "—"
                wl_rows.append(row)
            except Exception:
                wl_rows.append({"Stock": item["name"], "Ticker": item["ticker"], "Price (₹)": "Error", "Buy (₹)": "—", "P&L %": "—"})

        st.dataframe(pd.DataFrame(wl_rows), use_container_width=True, hide_index=True)

        if total_invested > 0:
            st.markdown("#### Portfolio Summary")
            ps1, ps2, ps3, ps4 = st.columns(4)
            ps1.metric("Total Invested", f"₹{total_invested:,.0f}")
            ps2.metric("Current Value", f"₹{total_current:,.0f}")
            net_pnl = total_current - total_invested
            net_pct = (net_pnl / total_invested) * 100
            ps3.metric("Net P&L", f"₹{net_pnl:+,.0f}", f"{net_pct:+.2f}%")
            ps4.metric("Stocks", str(len(st.session_state["watchlist"])))

            if net_pnl >= 0:
                verdict_box(f"📈 <b>Portfolio is UP {net_pct:.2f}%</b> — Total gain: ₹{net_pnl:,.0f}", "good")
            else:
                verdict_box(f"📉 <b>Portfolio is DOWN {net_pct:.2f}%</b> — Total loss: ₹{abs(net_pnl):,.0f}", "bad")

        # Remove buttons
        st.markdown("#### Remove Stocks")
        rm_cols = st.columns(min(len(st.session_state["watchlist"]), 5))
        for i, item in enumerate(st.session_state["watchlist"]):
            col_idx = i % 5
            with rm_cols[col_idx]:
                if st.button(f"❌ {item['ticker'].replace('.NS','')}", key=f"rm_{item['ticker']}"):
                    st.session_state["watchlist"] = [w for w in st.session_state["watchlist"] if w["ticker"] != item["ticker"]]
                    st.rerun()
    else:
        st.info("Your watchlist is empty. Add stocks above to start tracking!")


# ========================================================================
# TAB 7: SCREENER
# ========================================================================
with tab_screener:
    help_box("<b>Stock Screener</b> scans multiple stocks and ranks them by swing trading signal strength. "
             "It checks RSI, MACD, ADX, moving averages, and more for each stock to find the best opportunities.")

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

        progress = st.progress(0, text="Scanning stocks...")
        with st.spinner("Analyzing each stock..."):
            scan_results = []
            for idx, t in enumerate(scan_tickers):
                progress.progress((idx + 1) / len(scan_tickers), text=f"Scanning {t.replace('.NS', '')}...")
                try:
                    t_df = fetch_stock_data(t, period_years=1)
                    t_df = add_technical_indicators(t_df)
                    t_signal = generate_swing_signals(t_df)
                    t_latest = t_df.iloc[-1]
                    scan_results.append({
                        "ticker": t,
                        "name": POPULAR_INDIAN_STOCKS.get(t, t.replace(".NS", "")),
                        "price": t_latest["Close"],
                        "signal": t_signal.signal,
                        "strength": t_signal.strength,
                        "confidence": t_signal.confidence,
                        "rsi": t_latest.get("RSI", 0),
                        "macd": t_latest.get("MACD", 0),
                        "macd_signal": t_latest.get("MACD_Signal", 0),
                        "adx": t_latest.get("ADX", 0),
                        "volume_ratio": t_latest.get("Volume_Ratio", 1),
                    })
                except Exception:
                    continue
            scan_results.sort(key=lambda x: x["confidence"], reverse=True)
            progress.progress(1.0, text="Scan complete!")

        st.session_state["scan_results"] = scan_results

    if "scan_results" in st.session_state and st.session_state["scan_results"]:
        results = st.session_state["scan_results"]

        # Top picks
        buys = [r for r in results if r["signal"] == "BUY"]
        sells = [r for r in results if r["signal"] == "SELL"]
        holds = [r for r in results if r["signal"] == "HOLD"]

        st.markdown("#### Top Opportunities")
        if buys:
            top = buys[:3]
            tc = st.columns(len(top))
            for i, pick in enumerate(top):
                with tc[i]:
                    action_card(
                        f"BUY: {pick['name']}",
                        f"₹{pick['price']:,.2f} | {pick['strength']} | {pick['confidence']*100:.0f}% conf",
                        "action-buy",
                    )
        else:
            st.info("No BUY signals found in this scan.")

        if sells:
            st.markdown("#### Stocks to Avoid / Sell")
            sell_top = sells[:3]
            sc_cols = st.columns(len(sell_top))
            for i, pick in enumerate(sell_top):
                with sc_cols[i]:
                    action_card(
                        f"SELL: {pick['name']}",
                        f"₹{pick['price']:,.2f} | {pick['strength']} | {pick['confidence']*100:.0f}% conf",
                        "action-sell",
                    )

        # Full results table
        st.markdown("#### All Scanned Stocks")
        table_rows = []
        for r in results:
            sig_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(r["signal"], "⚪")
            table_rows.append({
                "Stock": r["name"],
                "Price (₹)": f"{r['price']:,.2f}",
                "Signal": f"{sig_emoji} {r['signal']}",
                "Strength": r["strength"],
                "Confidence": f"{r['confidence']*100:.0f}%",
                "RSI": f"{r['rsi']:.0f}",
                "ADX": f"{r['adx']:.0f}",
                "Vol Ratio": f"{r['volume_ratio']:.1f}x",
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        st.markdown(f"**Summary:** {len(buys)} BUY | {len(sells)} SELL | {len(holds)} HOLD out of {len(results)} stocks scanned")


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
        st.info("No predictions logged yet. Go to the Predictions tab and run a prediction to start tracking.")
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
    <b>Disclaimer:</b> Educational purposes only. Not financial advice. Markets are risky. Do your own research.
</div>""", unsafe_allow_html=True)
