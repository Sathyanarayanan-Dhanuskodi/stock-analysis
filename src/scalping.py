import pandas as pd
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ScalpSignal:
    signal: str  # "LONG" | "SHORT" | "NO_TRADE"
    strength: str  # "Strong" | "Moderate" | "Weak"
    confidence: float
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_reward: float
    reasons: list[str] = field(default_factory=list)


def add_scalping_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicators optimized for 5-minute scalping."""
    df = df.copy()

    # Fast EMAs for scalping
    df["EMA_5"] = df["Close"].ewm(span=5, adjust=False).mean()
    df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()

    # VWAP (Volume Weighted Average Price)
    # Group by date for intraday VWAP reset
    df["Typical_Price"] = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VP"] = df["Typical_Price"] * df["Volume"]

    # Approximate VWAP using cumulative within each day
    df["Date"] = df.index.date
    df["Cum_VP"] = df.groupby("Date")["VP"].cumsum()
    df["Cum_Vol"] = df.groupby("Date")["Volume"].cumsum()
    df["VWAP"] = df["Cum_VP"] / df["Cum_Vol"]
    df.drop(columns=["Typical_Price", "VP", "Date", "Cum_VP", "Cum_Vol"], inplace=True)

    # RSI (fast, 7-period for scalping)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=7).mean()
    avg_loss = loss.rolling(window=7).mean()
    rs = avg_gain / avg_loss
    df["RSI_7"] = 100 - (100 / (1 + rs))

    # Stochastic RSI (for overbought/oversold in fast timeframes)
    rsi = df["RSI_7"]
    rsi_min = rsi.rolling(window=14).min()
    rsi_max = rsi.rolling(window=14).max()
    df["StochRSI"] = (rsi - rsi_min) / (rsi_max - rsi_min) * 100

    # MACD (fast settings for scalping: 5, 13, 4)
    ema_fast = df["Close"].ewm(span=5, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=13, adjust=False).mean()
    df["MACD_Scalp"] = ema_fast - ema_slow
    df["MACD_Scalp_Signal"] = df["MACD_Scalp"].ewm(span=4, adjust=False).mean()
    df["MACD_Scalp_Hist"] = df["MACD_Scalp"] - df["MACD_Scalp_Signal"]

    # Bollinger Bands (tight, 10-period for scalping)
    bb_sma = df["Close"].rolling(window=10).mean()
    bb_std = df["Close"].rolling(window=10).std()
    df["BB_Upper_Scalp"] = bb_sma + 2 * bb_std
    df["BB_Lower_Scalp"] = bb_sma - 2 * bb_std
    df["BB_Mid_Scalp"] = bb_sma

    # ATR (fast, 7-period)
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR_7"] = tr.rolling(window=7).mean()

    # Volume analysis
    df["Vol_MA_10"] = df["Volume"].rolling(window=10).mean()
    df["Vol_Ratio"] = df["Volume"] / df["Vol_MA_10"]

    # Price momentum (rate of change over last 3 candles)
    df["Momentum_3"] = (df["Close"] - df["Close"].shift(3)) / df["Close"].shift(3) * 100

    # Spread / Range
    df["Candle_Range"] = (df["High"] - df["Low"]) / df["Close"] * 100
    df["Body_Pct"] = abs(df["Close"] - df["Open"]) / (df["High"] - df["Low"]) * 100

    df.dropna(inplace=True)
    return df


def _find_best_price(df: pd.DataFrame, lookback: int = 24) -> dict:
    """Find the best actionable entry/exit prices from recent price action.

    Adapts strategy based on market condition:
    - Trending: use pullback levels (EMA, VWAP) for entry, momentum for exit
    - Ranging: use swing support/resistance for entry/exit
    - Always considers multiple price sources and picks the best one

    lookback=24 candles on 5-min = ~2 hours of recent data.
    """
    recent = df.tail(lookback)
    highs = recent["High"].values
    lows = recent["Low"].values
    closes = recent["Close"].values
    current_price = closes[-1]

    # --- 1. Swing highs/lows (S/R levels) ---
    swing_lows = []
    for i in range(1, len(lows) - 1):
        if lows[i] <= lows[i - 1] and lows[i] <= lows[i + 1]:
            swing_lows.append(lows[i])
    swing_highs = []
    for i in range(1, len(highs) - 1):
        if highs[i] >= highs[i - 1] and highs[i] >= highs[i + 1]:
            swing_highs.append(highs[i])

    if not swing_lows:
        swing_lows = [recent["Low"].min()]
    if not swing_highs:
        swing_highs = [recent["High"].max()]

    supports_below = [s for s in swing_lows if s <= current_price]
    nearest_support = max(supports_below) if supports_below else min(swing_lows)
    resistances_above = [r for r in swing_highs if r >= current_price]
    nearest_resistance = min(resistances_above) if resistances_above else max(swing_highs)

    # --- 2. Detect market condition: trending vs ranging ---
    ema5 = recent["EMA_5"].values if "EMA_5" in recent.columns else None
    ema9 = recent["EMA_9"].values if "EMA_9" in recent.columns else None
    vwap = recent["VWAP"].values[-1] if "VWAP" in recent.columns else current_price
    bb_lower = recent["BB_Lower_Scalp"].values[-1] if "BB_Lower_Scalp" in recent.columns else None
    bb_upper = recent["BB_Upper_Scalp"].values[-1] if "BB_Upper_Scalp" in recent.columns else None

    is_trending_up = False
    is_trending_down = False
    if ema5 is not None and len(ema5) >= 5:
        slope = (ema5[-1] - ema5[-5]) / ema5[-5] * 100 if ema5[-5] > 0 else 0
        if slope > 0.1:
            is_trending_up = True
        elif slope < -0.1:
            is_trending_down = True

    # --- 3. Build candidate prices from multiple sources ---
    # Each candidate: (price, source_label)
    buy_candidates = []
    sell_candidates = []

    # S/R levels
    buy_candidates.append((nearest_support, "recent support"))
    sell_candidates.append((nearest_resistance, "recent resistance"))

    # VWAP — institutional magnet level
    if vwap < current_price:
        buy_candidates.append((vwap, "VWAP"))
    elif vwap > current_price:
        sell_candidates.append((vwap, "VWAP"))

    # Bollinger Bands — mean reversion zones
    if bb_lower is not None and bb_lower < current_price:
        buy_candidates.append((bb_lower, "lower Bollinger Band"))
    if bb_upper is not None and bb_upper > current_price:
        sell_candidates.append((bb_upper, "upper Bollinger Band"))

    # EMA pullback — in a trend, price often pulls back to the fast EMA
    if ema5 is not None:
        ema5_now = ema5[-1]
        if is_trending_up and ema5_now < current_price:
            buy_candidates.append((ema5_now, "EMA5 pullback"))
        elif is_trending_down and ema5_now > current_price:
            sell_candidates.append((ema5_now, "EMA5 pullback"))
    if ema9 is not None:
        ema9_now = ema9[-1]
        if is_trending_up and ema9_now < current_price:
            buy_candidates.append((ema9_now, "EMA9 pullback"))
        elif is_trending_down and ema9_now > current_price:
            sell_candidates.append((ema9_now, "EMA9 pullback"))

    # Recent candle low/high — immediate action levels
    last_3_low = recent["Low"].iloc[-3:].min()
    last_3_high = recent["High"].iloc[-3:].max()
    if last_3_low < current_price:
        buy_candidates.append((last_3_low, "last 3-candle low"))
    if last_3_high > current_price:
        sell_candidates.append((last_3_high, "last 3-candle high"))

    # --- 4. Pick the BEST price based on market condition ---
    atr = recent["ATR_7"].values[-1] if "ATR_7" in recent.columns else current_price * 0.003

    if is_trending_up:
        # In uptrend: best buy = highest candidate (shallowest pullback, most realistic)
        # best sell = highest candidate (ride the trend)
        buy_candidates.sort(key=lambda x: x[0], reverse=True)
        sell_candidates.sort(key=lambda x: x[0], reverse=True)
    elif is_trending_down:
        # In downtrend: best sell = lowest candidate (shallowest pullback)
        # best buy = lowest candidate (deepest support)
        buy_candidates.sort(key=lambda x: x[0])
        sell_candidates.sort(key=lambda x: x[0])
    else:
        # Ranging: best buy = nearest support (highest below price)
        # best sell = nearest resistance (lowest above price)
        buy_candidates.sort(key=lambda x: x[0], reverse=True)
        sell_candidates.sort(key=lambda x: x[0])

    # Filter out candidates that are too far (> 2x ATR from current price)
    buy_candidates = [(p, s) for p, s in buy_candidates if abs(current_price - p) <= 2 * atr]
    sell_candidates = [(p, s) for p, s in sell_candidates if abs(p - current_price) <= 2 * atr]

    best_buy = buy_candidates[0] if buy_candidates else (current_price, "current price")
    best_sell = sell_candidates[0] if sell_candidates else (current_price + atr, "ATR projection")

    condition = "uptrend" if is_trending_up else "downtrend" if is_trending_down else "ranging"

    return {
        "best_buy_price": best_buy[0],
        "best_buy_source": best_buy[1],
        "best_sell_price": best_sell[0],
        "best_sell_source": best_sell[1],
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "condition": condition,
        "atr": atr,
    }


def generate_scalp_signal(df: pd.DataFrame) -> ScalpSignal:
    """Generate scalping signal from 5-min candle data with indicators.

    Uses recent price action (last ~2 hours) for entry/exit levels,
    combined with multi-factor confirmation to reduce false signals.
    """
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    recent_5 = df.tail(5)

    # Find best actionable prices from last 2 hours of price action
    best_prices = _find_best_price(df, lookback=24)

    score = 0
    reasons = []
    confirmations = 0  # Track how many independent confirmations we have

    # --- TREND FILTERS (higher weight) ---

    # EMA alignment (trend direction) — must have all 3 aligned for full score
    ema_bullish = latest["EMA_5"] > latest["EMA_9"] > latest["EMA_21"]
    ema_bearish = latest["EMA_5"] < latest["EMA_9"] < latest["EMA_21"]
    if ema_bullish:
        score += 2
        confirmations += 1
        reasons.append("EMA 5 > 9 > 21 (bullish alignment)")
    elif ema_bearish:
        score -= 2
        confirmations += 1
        reasons.append("EMA 5 < 9 < 21 (bearish alignment)")

    # EMA crossover (entry trigger) — only count if aligned with trend
    if prev["EMA_5"] <= prev["EMA_9"] and latest["EMA_5"] > latest["EMA_9"]:
        score += 2
        confirmations += 1
        reasons.append("EMA 5/9 bullish crossover (fresh)")
    elif prev["EMA_5"] >= prev["EMA_9"] and latest["EMA_5"] < latest["EMA_9"]:
        score -= 2
        confirmations += 1
        reasons.append("EMA 5/9 bearish crossover (fresh)")

    # EMA 21 slope direction (higher timeframe trend)
    if len(df) >= 10:
        ema21_now = latest["EMA_21"]
        ema21_prev5 = df.iloc[-6]["EMA_21"] if len(df) > 5 else ema21_now
        ema21_slope = (ema21_now - ema21_prev5) / ema21_prev5 * 100 if ema21_prev5 > 0 else 0
        if ema21_slope > 0.05:
            score += 1
            reasons.append(f"EMA 21 trending up ({ema21_slope:+.2f}%)")
        elif ema21_slope < -0.05:
            score -= 1
            reasons.append(f"EMA 21 trending down ({ema21_slope:+.2f}%)")

    # --- VWAP (key institutional level) ---
    vwap_dist_pct = (latest["Close"] - latest["VWAP"]) / latest["VWAP"] * 100 if latest["VWAP"] > 0 else 0
    if latest["Close"] > latest["VWAP"]:
        score += 1
        confirmations += 1
        reasons.append(f"Price above VWAP by {vwap_dist_pct:.2f}%")
    else:
        score -= 1
        confirmations += 1
        reasons.append(f"Price below VWAP by {vwap_dist_pct:.2f}%")

    # --- MOMENTUM INDICATORS ---

    # RSI conditions (fast RSI) — use stricter thresholds
    rsi = latest["RSI_7"]
    if rsi < 20:
        score += 2
        confirmations += 1
        reasons.append(f"RSI deeply oversold ({rsi:.0f}) - strong bounce likely")
    elif rsi < 30:
        score += 1
        reasons.append(f"RSI oversold ({rsi:.0f}) - bounce possible")
    elif rsi > 80:
        score -= 2
        confirmations += 1
        reasons.append(f"RSI deeply overbought ({rsi:.0f}) - strong pullback likely")
    elif rsi > 70:
        score -= 1
        reasons.append(f"RSI overbought ({rsi:.0f}) - pullback possible")

    # Stochastic RSI — only count at extremes
    stoch_rsi = latest.get("StochRSI", 50)
    if stoch_rsi < 15:
        score += 1
        reasons.append(f"StochRSI deeply oversold ({stoch_rsi:.0f})")
    elif stoch_rsi > 85:
        score -= 1
        reasons.append(f"StochRSI deeply overbought ({stoch_rsi:.0f})")

    # MACD scalping — require histogram building (not just crossing)
    macd_hist = latest["MACD_Scalp_Hist"]
    prev_hist = prev["MACD_Scalp_Hist"]
    if macd_hist > 0 and prev_hist <= 0:
        score += 2
        confirmations += 1
        reasons.append("MACD histogram turned positive")
    elif macd_hist < 0 and prev_hist >= 0:
        score -= 2
        confirmations += 1
        reasons.append("MACD histogram turned negative")
    elif macd_hist > 0 and macd_hist > prev_hist:
        score += 1
        reasons.append("MACD histogram growing positive")
    elif macd_hist < 0 and macd_hist < prev_hist:
        score -= 1
        reasons.append("MACD histogram growing negative")

    # --- PRICE ACTION ---

    # Bollinger Band — only at clear extremes
    bb_width_pct = (latest["BB_Upper_Scalp"] - latest["BB_Lower_Scalp"]) / latest["BB_Mid_Scalp"] * 100 if latest["BB_Mid_Scalp"] > 0 else 0
    if latest["Close"] <= latest["BB_Lower_Scalp"]:
        score += 1
        reasons.append(f"Price at/below lower BB (bounce zone, BB width: {bb_width_pct:.2f}%)")
    elif latest["Close"] >= latest["BB_Upper_Scalp"]:
        score -= 1
        reasons.append(f"Price at/above upper BB (rejection zone, BB width: {bb_width_pct:.2f}%)")

    # Consecutive candle direction (trend persistence)
    up_candles = sum(1 for i in range(len(recent_5)) if recent_5["Close"].iloc[i] > recent_5["Open"].iloc[i])
    if up_candles >= 4:
        score += 1
        reasons.append(f"{up_candles}/5 recent candles bullish")
    elif up_candles <= 1:
        score -= 1
        reasons.append(f"{5 - up_candles}/5 recent candles bearish")

    # --- VOLUME CONFIRMATION (critical for scalping) ---
    vol_ratio = latest.get("Vol_Ratio", 1)
    if vol_ratio > 2.0:
        if latest["Close"] > prev["Close"]:
            score += 2
            confirmations += 1
            reasons.append(f"Strong volume surge on up candle ({vol_ratio:.1f}x)")
        else:
            score -= 2
            confirmations += 1
            reasons.append(f"Strong volume surge on down candle ({vol_ratio:.1f}x)")
    elif vol_ratio > 1.3:
        if latest["Close"] > prev["Close"]:
            score += 1
            reasons.append(f"Volume above average on up candle ({vol_ratio:.1f}x)")
        else:
            score -= 1
            reasons.append(f"Volume above average on down candle ({vol_ratio:.1f}x)")
    elif vol_ratio < 0.5:
        # Low volume = unreliable signals, penalize
        if score > 0:
            score -= 1
        elif score < 0:
            score += 1
        reasons.append(f"Low volume ({vol_ratio:.1f}x) — signal less reliable")

    # Momentum (last 3 candles)
    momentum = latest.get("Momentum_3", 0)
    if momentum > 0.2:
        score += 1
        reasons.append(f"Strong upward momentum ({momentum:.2f}%)")
    elif momentum < -0.2:
        score -= 1
        reasons.append(f"Strong downward momentum ({momentum:.2f}%)")

    # --- SIGNAL DETERMINATION ---
    # Stricter thresholds: require score >= 4 for Strong, >= 2 for Moderate
    # Also require minimum confirmations (independent factor agreement)
    current_price = latest["Close"]
    atr = latest.get("ATR_7", current_price * 0.003)

    # Check if ATR is too small relative to price (stock not moving enough)
    atr_pct = atr / current_price * 100 if current_price > 0 else 0
    if atr_pct < 0.15:
        reasons.append(f"Low ATR ({atr_pct:.2f}%) — tight range, scalping difficult")
        # Dampen signals in tight ranges
        if abs(score) <= 3:
            score = 0

    if score >= 5 and confirmations >= 3:
        signal, strength = "LONG", "Strong"
    elif score >= 3 and confirmations >= 2:
        signal, strength = "LONG", "Moderate"
    elif score <= -5 and confirmations >= 3:
        signal, strength = "SHORT", "Strong"
    elif score <= -3 and confirmations >= 2:
        signal, strength = "SHORT", "Moderate"
    else:
        signal, strength = "NO_TRADE", "Weak"

    # Use detected S/R levels for stop-loss and targets; entry is always at current market price.
    support = best_prices["nearest_support"]
    resistance = best_prices["nearest_resistance"]
    condition = best_prices["condition"]

    if signal == "LONG":
        entry = current_price  # Enter at market price NOW
        # Stop at nearest support, capped at 1.5×ATR, but at least 0.5×ATR below entry
        stop_loss = max(support, entry - 1.5 * atr)
        stop_loss = min(stop_loss, entry - 0.5 * atr)
        risk = entry - stop_loss
        if risk < 0.2 * atr:  # Too tight, not tradeable
            signal, strength = "NO_TRADE", "Weak"
            risk = atr
        # Target capped at 1×ATR (realistic 5-min scalp move); use resistance if closer
        target_1 = min(resistance, entry + atr)
        if target_1 <= entry:
            target_1 = entry + atr
        target_2 = target_1 + 0.5 * atr
        reasons.append(f"Entry at market ({current_price:.2f}), stop at {stop_loss:.2f} (support), target {target_1:.2f} (resistance) [{condition}]")
    elif signal == "SHORT":
        entry = current_price  # Enter at market price NOW (short)
        # Stop at nearest resistance, capped at 1.5×ATR, but at least 0.5×ATR above entry
        stop_loss = min(resistance, entry + 1.5 * atr)
        stop_loss = max(stop_loss, entry + 0.5 * atr)
        risk = stop_loss - entry
        if risk < 0.2 * atr:  # Too tight, not tradeable
            signal, strength = "NO_TRADE", "Weak"
            risk = atr
        # Target capped at 1×ATR (realistic 5-min scalp move); use support if closer
        target_1 = max(support, entry - atr)
        if target_1 >= entry:
            target_1 = entry - atr
        target_2 = target_1 - 0.5 * atr
        reasons.append(f"Entry at market ({current_price:.2f}), stop at {stop_loss:.2f} (resistance), target {target_1:.2f} (support) [{condition}]")
    else:
        entry = current_price
        stop_loss = current_price - 1.5 * atr
        target_1 = current_price + 1.5 * atr
        target_2 = target_1 + 0.5 * atr

    risk = abs(entry - stop_loss)
    reward = abs(target_1 - entry)
    rr = reward / risk if risk > 0 else 0

    confidence = min(1.0, abs(score) / 10)

    return ScalpSignal(
        signal=signal, strength=strength, confidence=confidence,
        entry_price=round(entry, 2), stop_loss=round(stop_loss, 2),
        target_1=round(target_1, 2), target_2=round(target_2, 2),
        risk_reward=round(rr, 2), reasons=reasons,
    )


def get_scalping_levels(df: pd.DataFrame) -> dict:
    """Calculate key intraday levels for scalping."""
    latest = df.iloc[-1]

    # Today's data
    today = df.index[-1].date() if hasattr(df.index[-1], 'date') else df.index[-1]
    today_data = df[df.index.date == today] if hasattr(df.index[0], 'date') else df.tail(78)  # ~78 5-min candles per day

    # Previous day data (for pivots)
    dates = sorted(set(df.index.date)) if hasattr(df.index[0], 'date') else []
    if len(dates) >= 2:
        prev_date = dates[-2]
        prev_data = df[df.index.date == prev_date]
        prev_high = prev_data["High"].max()
        prev_low = prev_data["Low"].min()
        prev_close = prev_data["Close"].iloc[-1]
    else:
        prev_data = df.iloc[:-78] if len(df) > 78 else df.head(len(df)//2)
        prev_high = prev_data["High"].max()
        prev_low = prev_data["Low"].min()
        prev_close = prev_data["Close"].iloc[-1]

    # Central Pivot Range (CPR)
    pp = (prev_high + prev_low + prev_close) / 3
    bc = (prev_high + prev_low) / 2  # Bottom Central
    tc = 2 * pp - bc  # Top Central

    # Camarilla pivots (tight for scalping)
    cam_range = prev_high - prev_low
    cam_r1 = prev_close + cam_range * 1.1 / 12
    cam_r2 = prev_close + cam_range * 1.1 / 6
    cam_r3 = prev_close + cam_range * 1.1 / 4
    cam_s1 = prev_close - cam_range * 1.1 / 12
    cam_s2 = prev_close - cam_range * 1.1 / 6
    cam_s3 = prev_close - cam_range * 1.1 / 4

    # Today's range
    today_high = today_data["High"].max() if len(today_data) > 0 else latest["High"]
    today_low = today_data["Low"].min() if len(today_data) > 0 else latest["Low"]
    today_open = today_data["Open"].iloc[0] if len(today_data) > 0 else latest["Open"]

    vwap = latest.get("VWAP", pp)

    return {
        "pivot": round(pp, 2),
        "top_central": round(tc, 2),
        "bottom_central": round(bc, 2),
        "cam_r1": round(cam_r1, 2), "cam_r2": round(cam_r2, 2), "cam_r3": round(cam_r3, 2),
        "cam_s1": round(cam_s1, 2), "cam_s2": round(cam_s2, 2), "cam_s3": round(cam_s3, 2),
        "vwap": round(vwap, 2),
        "today_high": round(today_high, 2),
        "today_low": round(today_low, 2),
        "today_open": round(today_open, 2),
        "prev_high": round(prev_high, 2),
        "prev_low": round(prev_low, 2),
        "prev_close": round(prev_close, 2),
    }


def get_market_microstructure(df: pd.DataFrame) -> dict:
    """Analyze market microstructure for scalping edge."""
    latest = df.iloc[-1]
    recent = df.tail(20)  # Last 20 candles (~100 mins)

    # Trend detection (last 20 candles)
    ema5 = recent["EMA_5"].values if "EMA_5" in recent.columns else recent["Close"].ewm(span=5).mean().values
    trend_slope = (ema5[-1] - ema5[0]) / len(ema5) if len(ema5) > 1 else 0
    trend = "Uptrend" if trend_slope > 0.05 else "Downtrend" if trend_slope < -0.05 else "Sideways"

    # Volatility regime
    atr = latest.get("ATR_7", 0)
    avg_atr = recent["ATR_7"].mean() if "ATR_7" in recent.columns else atr
    vol_regime = "High" if atr > avg_atr * 1.3 else "Low" if atr < avg_atr * 0.7 else "Normal"

    # Volume profile
    avg_vol = recent["Volume"].mean()
    current_vol = latest["Volume"]
    vol_status = "Above Average" if current_vol > avg_vol * 1.3 else "Below Average" if current_vol < avg_vol * 0.7 else "Average"

    # Spread analysis
    avg_range = recent["Candle_Range"].mean() if "Candle_Range" in recent.columns else 0
    current_range = latest.get("Candle_Range", 0)

    # Consecutive candle direction
    closes = recent["Close"].values
    consecutive = 0
    direction = "up" if closes[-1] > closes[-2] else "down"
    for i in range(len(closes) - 1, 0, -1):
        if direction == "up" and closes[i] > closes[i-1]:
            consecutive += 1
        elif direction == "down" and closes[i] < closes[i-1]:
            consecutive += 1
        else:
            break

    # Scalp-ability score (0-100) — stricter criteria
    scalpability = 40  # Start lower, earn points
    # Volatility is key for scalping
    if vol_regime == "High":
        scalpability += 20
    elif vol_regime == "Normal":
        scalpability += 5
    elif vol_regime == "Low":
        scalpability -= 25
    # Volume is essential — can't scalp without liquidity
    if vol_status == "Above Average":
        scalpability += 20
    elif vol_status == "Average":
        scalpability += 5
    elif vol_status == "Below Average":
        scalpability -= 20
    # Clear trend is better than sideways for scalping
    if trend != "Sideways":
        scalpability += 10
    # ATR must be large enough to cover spread + slippage
    if atr > 0 and latest["Close"] > 0:
        spread_pct = atr / latest["Close"] * 100
        if spread_pct > 0.4:
            scalpability += 15
        elif spread_pct > 0.25:
            scalpability += 5
        elif spread_pct < 0.15:
            scalpability -= 15  # Too tight, can't profit
    # Consecutive candles in one direction = trending = better for scalping
    if consecutive >= 3:
        scalpability += 5
    scalpability = max(0, min(100, scalpability))

    return {
        "trend": trend,
        "trend_slope": round(trend_slope, 4),
        "volatility_regime": vol_regime,
        "volume_status": vol_status,
        "avg_candle_range": round(avg_range, 3),
        "current_candle_range": round(current_range, 3),
        "consecutive_candles": consecutive,
        "consecutive_direction": direction,
        "scalpability_score": scalpability,
        "atr": round(atr, 2),
    }
