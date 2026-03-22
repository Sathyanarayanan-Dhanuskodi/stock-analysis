import pandas as pd
import numpy as np
from dataclasses import dataclass, field


@dataclass
class DayTradeSignal:
    signal: str  # "LONG" | "SHORT" | "NO_TRADE"
    strength: str  # "Strong" | "Moderate" | "Weak"
    confidence: float
    strategy: str  # "ORB" | "Trend Following" | "Breakout" | "Mixed"
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    target_3: float
    risk_reward: float
    reasons: list[str] = field(default_factory=list)


def add_day_trading_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicators optimized for 15-minute day trading."""
    df = df.copy()

    # EMAs for day trading (medium speed)
    df["EMA_9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA_21"] = df["Close"].ewm(span=21, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # VWAP (daily reset)
    df["Typical_Price"] = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VP"] = df["Typical_Price"] * df["Volume"]
    df["Date"] = df.index.date
    df["Cum_VP"] = df.groupby("Date")["VP"].cumsum()
    df["Cum_Vol"] = df.groupby("Date")["Volume"].cumsum()
    df["VWAP"] = df["Cum_VP"] / df["Cum_Vol"]
    df.drop(columns=["Typical_Price", "VP", "Date", "Cum_VP", "Cum_Vol"], inplace=True)

    # RSI (standard 14-period)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD (standard 12/26/9)
    ema_fast = df["Close"].ewm(span=12, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # Bollinger Bands (standard 20-period)
    bb_sma = df["Close"].rolling(window=20).mean()
    bb_std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = bb_sma + 2 * bb_std
    df["BB_Lower"] = bb_sma - 2 * bb_std
    df["BB_Mid"] = bb_sma

    # ATR (standard 14-period)
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(window=14).mean()

    # ADX (14-period, inline)
    plus_dm = df["High"].diff()
    minus_dm = -df["Low"].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr_adx = tr.rolling(window=14).mean()
    df["Plus_DI"] = 100 * (plus_dm.rolling(window=14).mean() / atr_adx)
    df["Minus_DI"] = 100 * (minus_dm.rolling(window=14).mean() / atr_adx)
    dx = (abs(df["Plus_DI"] - df["Minus_DI"]) / (df["Plus_DI"] + df["Minus_DI"])) * 100
    df["ADX"] = dx.rolling(window=14).mean()

    # Volume analysis
    df["Vol_MA_20"] = df["Volume"].rolling(window=20).mean()
    df["Vol_Ratio"] = df["Volume"] / df["Vol_MA_20"]

    # Price momentum (rate of change over last 5 candles = ~75 min)
    df["Momentum_5"] = (df["Close"] - df["Close"].shift(5)) / df["Close"].shift(5) * 100

    # Candle analysis
    df["Candle_Range"] = (df["High"] - df["Low"]) / df["Close"] * 100
    df["Body_Pct"] = abs(df["Close"] - df["Open"]) / (df["High"] - df["Low"]) * 100

    df.dropna(inplace=True)
    return df


def get_opening_range(df: pd.DataFrame, minutes: int = 30) -> dict:
    """Calculate Opening Range Breakout levels (first 30 min: 9:15-9:45 IST).

    minutes: 30 for standard ORB (2 candles on 15-min), 15 for narrow ORB (1 candle).
    """
    today = df.index[-1].date() if hasattr(df.index[-1], "date") else None
    if today is None:
        return _empty_orb()

    today_data = df[df.index.date == today]
    if len(today_data) == 0:
        return _empty_orb()

    # First N minutes of trading (9:15 IST start)
    market_open = today_data.index[0]
    orb_end = market_open + pd.Timedelta(minutes=minutes)
    orb_candles = today_data[today_data.index <= orb_end]

    if len(orb_candles) == 0:
        return _empty_orb()

    orb_high = orb_candles["High"].max()
    orb_low = orb_candles["Low"].min()
    orb_mid = (orb_high + orb_low) / 2
    orb_width = orb_high - orb_low

    # Compare ORB width to ATR to classify narrow vs wide
    atr = today_data["ATR_14"].iloc[-1] if "ATR_14" in today_data.columns else orb_width
    orb_type = "Narrow" if orb_width < 0.5 * atr else "Wide" if orb_width > 1.5 * atr else "Normal"

    current_price = today_data["Close"].iloc[-1]
    breakout_status = "None"
    if current_price > orb_high:
        breakout_status = "Bullish Breakout"
    elif current_price < orb_low:
        breakout_status = "Bearish Breakdown"
    elif current_price > orb_mid:
        breakout_status = "Above Mid"
    else:
        breakout_status = "Below Mid"

    return {
        "orb_high": round(orb_high, 2),
        "orb_low": round(orb_low, 2),
        "orb_mid": round(orb_mid, 2),
        "orb_width": round(orb_width, 2),
        "orb_type": orb_type,
        "breakout_status": breakout_status,
        "candles_in_orb": len(orb_candles),
    }


def _empty_orb() -> dict:
    return {
        "orb_high": 0, "orb_low": 0, "orb_mid": 0, "orb_width": 0,
        "orb_type": "N/A", "breakout_status": "N/A", "candles_in_orb": 0,
    }


def get_day_trading_levels(df: pd.DataFrame) -> dict:
    """Calculate key intraday levels for day trading."""
    latest = df.iloc[-1]

    today = df.index[-1].date() if hasattr(df.index[-1], "date") else df.index[-1]
    today_data = df[df.index.date == today] if hasattr(df.index[0], "date") else df.tail(26)

    # Previous day data
    dates = sorted(set(df.index.date)) if hasattr(df.index[0], "date") else []
    if len(dates) >= 2:
        prev_date = dates[-2]
        prev_data = df[df.index.date == prev_date]
        prev_high = prev_data["High"].max()
        prev_low = prev_data["Low"].min()
        prev_close = prev_data["Close"].iloc[-1]
    else:
        prev_data = df.iloc[:-26] if len(df) > 26 else df.head(len(df) // 2)
        prev_high = prev_data["High"].max()
        prev_low = prev_data["Low"].min()
        prev_close = prev_data["Close"].iloc[-1]

    # Standard Pivot Points
    pp = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pp - prev_low
    s1 = 2 * pp - prev_high
    r2 = pp + (prev_high - prev_low)
    s2 = pp - (prev_high - prev_low)
    r3 = prev_high + 2 * (pp - prev_low)
    s3 = prev_low - 2 * (prev_high - pp)

    # Camarilla pivots
    cam_range = prev_high - prev_low
    cam_r1 = prev_close + cam_range * 1.1 / 12
    cam_r2 = prev_close + cam_range * 1.1 / 6
    cam_r3 = prev_close + cam_range * 1.1 / 4
    cam_s1 = prev_close - cam_range * 1.1 / 12
    cam_s2 = prev_close - cam_range * 1.1 / 6
    cam_s3 = prev_close - cam_range * 1.1 / 4

    # ORB levels
    orb = get_opening_range(df)

    # Today's range
    today_high = today_data["High"].max() if len(today_data) > 0 else latest["High"]
    today_low = today_data["Low"].min() if len(today_data) > 0 else latest["Low"]
    today_open = today_data["Open"].iloc[0] if len(today_data) > 0 else latest["Open"]

    vwap = latest.get("VWAP", pp)

    return {
        # Standard pivots
        "pivot": round(pp, 2),
        "r1": round(r1, 2), "r2": round(r2, 2), "r3": round(r3, 2),
        "s1": round(s1, 2), "s2": round(s2, 2), "s3": round(s3, 2),
        # Camarilla
        "cam_r1": round(cam_r1, 2), "cam_r2": round(cam_r2, 2), "cam_r3": round(cam_r3, 2),
        "cam_s1": round(cam_s1, 2), "cam_s2": round(cam_s2, 2), "cam_s3": round(cam_s3, 2),
        # ORB
        "orb_high": orb["orb_high"], "orb_low": orb["orb_low"],
        # Day range
        "vwap": round(vwap, 2),
        "today_high": round(today_high, 2), "today_low": round(today_low, 2),
        "today_open": round(today_open, 2),
        "prev_high": round(prev_high, 2), "prev_low": round(prev_low, 2),
        "prev_close": round(prev_close, 2),
    }


def _find_best_day_trade_price(df: pd.DataFrame, lookback: int = 16) -> dict:
    """Find best entry/exit from recent price action (~4 hours on 15-min).

    Same approach as scalping but with wider lookback and ATR filter.
    """
    recent = df.tail(lookback)
    highs = recent["High"].values
    lows = recent["Low"].values
    closes = recent["Close"].values
    current_price = closes[-1]

    # Swing highs/lows
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

    # Detect market condition
    ema9 = recent["EMA_9"].values if "EMA_9" in recent.columns else None
    ema21 = recent["EMA_21"].values if "EMA_21" in recent.columns else None
    vwap = recent["VWAP"].values[-1] if "VWAP" in recent.columns else current_price

    is_trending_up = False
    is_trending_down = False
    if ema9 is not None and len(ema9) >= 5:
        slope = (ema9[-1] - ema9[-5]) / ema9[-5] * 100 if ema9[-5] > 0 else 0
        if slope > 0.1:
            is_trending_up = True
        elif slope < -0.1:
            is_trending_down = True

    # Build candidate prices
    buy_candidates = [(nearest_support, "recent support")]
    sell_candidates = [(nearest_resistance, "recent resistance")]

    if vwap < current_price:
        buy_candidates.append((vwap, "VWAP"))
    elif vwap > current_price:
        sell_candidates.append((vwap, "VWAP"))

    bb_lower = recent["BB_Lower"].values[-1] if "BB_Lower" in recent.columns else None
    bb_upper = recent["BB_Upper"].values[-1] if "BB_Upper" in recent.columns else None
    if bb_lower is not None and bb_lower < current_price:
        buy_candidates.append((bb_lower, "lower Bollinger Band"))
    if bb_upper is not None and bb_upper > current_price:
        sell_candidates.append((bb_upper, "upper Bollinger Band"))

    if ema9 is not None:
        ema9_now = ema9[-1]
        if is_trending_up and ema9_now < current_price:
            buy_candidates.append((ema9_now, "EMA9 pullback"))
        elif is_trending_down and ema9_now > current_price:
            sell_candidates.append((ema9_now, "EMA9 pullback"))
    if ema21 is not None:
        ema21_now = ema21[-1]
        if is_trending_up and ema21_now < current_price:
            buy_candidates.append((ema21_now, "EMA21 pullback"))
        elif is_trending_down and ema21_now > current_price:
            sell_candidates.append((ema21_now, "EMA21 pullback"))

    last_4_low = recent["Low"].iloc[-4:].min()
    last_4_high = recent["High"].iloc[-4:].max()
    if last_4_low < current_price:
        buy_candidates.append((last_4_low, "last 4-candle low"))
    if last_4_high > current_price:
        sell_candidates.append((last_4_high, "last 4-candle high"))

    # Sort by market condition
    atr = recent["ATR_14"].values[-1] if "ATR_14" in recent.columns else current_price * 0.005

    if is_trending_up:
        buy_candidates.sort(key=lambda x: x[0], reverse=True)
        sell_candidates.sort(key=lambda x: x[0], reverse=True)
    elif is_trending_down:
        buy_candidates.sort(key=lambda x: x[0])
        sell_candidates.sort(key=lambda x: x[0])
    else:
        buy_candidates.sort(key=lambda x: x[0], reverse=True)
        sell_candidates.sort(key=lambda x: x[0])

    # Wider filter: 3x ATR (day trade has more room than scalp)
    buy_candidates = [(p, s) for p, s in buy_candidates if abs(current_price - p) <= 3 * atr]
    sell_candidates = [(p, s) for p, s in sell_candidates if abs(p - current_price) <= 3 * atr]

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


def generate_day_trade_signal(df: pd.DataFrame) -> DayTradeSignal:
    """Generate day trading signal from 15-min candle data.

    Combines three strategies: ORB, Trend Following, and Breakout/Breakdown.
    """
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    recent_5 = df.tail(5)

    best_prices = _find_best_day_trade_price(df, lookback=16)
    orb = get_opening_range(df)
    levels = get_day_trading_levels(df)

    score = 0
    reasons = []
    confirmations = 0
    strategy_scores = {"orb": 0, "trend": 0, "breakout": 0}

    current_price = latest["Close"]
    atr = latest.get("ATR_14", current_price * 0.005)

    # ========== ORB STRATEGY (max ±5) ==========
    orb_high = orb["orb_high"]
    orb_low = orb["orb_low"]
    orb_width = orb["orb_width"]

    if orb_high > 0 and orb_low > 0:
        vol_ratio = latest.get("Vol_Ratio", 1)
        has_volume = vol_ratio > 1.3

        # Breakout above ORB high
        if current_price > orb_high:
            pts = 3 if has_volume else 2
            strategy_scores["orb"] += pts
            score += pts
            confirmations += 1
            vol_tag = " with volume" if has_volume else ""
            reasons.append(f"ORB bullish breakout{vol_tag} (price {current_price:.2f} > ORB high {orb_high:.2f})")

            # Narrow ORB breakouts are stronger
            if orb["orb_type"] == "Narrow":
                strategy_scores["orb"] += 1
                score += 1
                reasons.append("Narrow ORB — breakout has more momentum potential")

            # Retest: price came back near ORB high and bounced
            if len(df) >= 4:
                recent_low = df.tail(4)["Low"].min()
                if abs(recent_low - orb_high) / atr < 0.3:
                    strategy_scores["orb"] += 1
                    score += 1
                    reasons.append("ORB high retest confirmed as support")

        # Breakdown below ORB low
        elif current_price < orb_low:
            pts = -3 if has_volume else -2
            strategy_scores["orb"] += pts
            score += pts
            confirmations += 1
            vol_tag = " with volume" if has_volume else ""
            reasons.append(f"ORB bearish breakdown{vol_tag} (price {current_price:.2f} < ORB low {orb_low:.2f})")

            if orb["orb_type"] == "Narrow":
                strategy_scores["orb"] -= 1
                score -= 1
                reasons.append("Narrow ORB — breakdown has more momentum potential")

            if len(df) >= 4:
                recent_high = df.tail(4)["High"].max()
                if abs(recent_high - orb_low) / atr < 0.3:
                    strategy_scores["orb"] -= 1
                    score -= 1
                    reasons.append("ORB low retest confirmed as resistance")

    # ========== TREND FOLLOWING (max ±5) ==========

    # EMA alignment
    ema_bullish = latest["EMA_9"] > latest["EMA_21"] > latest["EMA_50"]
    ema_bearish = latest["EMA_9"] < latest["EMA_21"] < latest["EMA_50"]
    if ema_bullish:
        strategy_scores["trend"] += 2
        score += 2
        confirmations += 1
        reasons.append("EMA 9 > 21 > 50 (bullish alignment)")
    elif ema_bearish:
        strategy_scores["trend"] -= 2
        score -= 2
        confirmations += 1
        reasons.append("EMA 9 < 21 < 50 (bearish alignment)")

    # VWAP position
    vwap_dist_pct = (current_price - latest["VWAP"]) / latest["VWAP"] * 100 if latest["VWAP"] > 0 else 0
    if current_price > latest["VWAP"]:
        strategy_scores["trend"] += 1
        score += 1
        confirmations += 1
        reasons.append(f"Price above VWAP by {vwap_dist_pct:.2f}%")
    else:
        strategy_scores["trend"] -= 1
        score -= 1
        confirmations += 1
        reasons.append(f"Price below VWAP by {vwap_dist_pct:.2f}%")

    # RSI momentum (avoid overbought/oversold extremes)
    rsi = latest["RSI_14"]
    if 40 <= rsi <= 60:
        reasons.append(f"RSI neutral ({rsi:.0f})")
    elif rsi < 30:
        strategy_scores["trend"] += 1
        score += 1
        reasons.append(f"RSI oversold ({rsi:.0f}) — bounce potential")
    elif rsi > 70:
        strategy_scores["trend"] -= 1
        score -= 1
        reasons.append(f"RSI overbought ({rsi:.0f}) — pullback potential")
    elif 50 < rsi <= 70:
        strategy_scores["trend"] += 1
        score += 1
        reasons.append(f"RSI bullish momentum ({rsi:.0f})")
    elif 30 <= rsi < 50:
        strategy_scores["trend"] -= 1
        score -= 1
        reasons.append(f"RSI bearish momentum ({rsi:.0f})")

    # MACD histogram
    macd_hist = latest["MACD_Hist"]
    prev_hist = prev["MACD_Hist"]
    if macd_hist > 0 and prev_hist <= 0:
        strategy_scores["trend"] += 1
        score += 1
        confirmations += 1
        reasons.append("MACD histogram turned positive")
    elif macd_hist < 0 and prev_hist >= 0:
        strategy_scores["trend"] -= 1
        score -= 1
        confirmations += 1
        reasons.append("MACD histogram turned negative")
    elif macd_hist > 0 and macd_hist > prev_hist:
        reasons.append("MACD histogram growing positive")
    elif macd_hist < 0 and macd_hist < prev_hist:
        reasons.append("MACD histogram growing negative")

    # ========== BREAKOUT/BREAKDOWN (max ±4) ==========

    # Pivot level breakout
    r1 = levels["r1"]
    s1 = levels["s1"]
    if current_price > r1:
        strategy_scores["breakout"] += 2
        score += 2
        confirmations += 1
        reasons.append(f"Price broke above R1 ({r1:.2f})")
    elif current_price < s1:
        strategy_scores["breakout"] -= 2
        score -= 2
        confirmations += 1
        reasons.append(f"Price broke below S1 ({s1:.2f})")

    # ADX trend strength
    adx = latest.get("ADX", 0)
    if adx > 25:
        if score > 0:
            strategy_scores["breakout"] += 1
            score += 1
        elif score < 0:
            strategy_scores["breakout"] -= 1
            score -= 1
        confirmations += 1
        reasons.append(f"ADX strong trend ({adx:.0f})")
    elif adx < 20:
        reasons.append(f"ADX weak trend ({adx:.0f}) — range-bound")

    # Volume surge on breakout
    vol_ratio = latest.get("Vol_Ratio", 1)
    if vol_ratio > 1.5:
        if current_price > prev["Close"]:
            strategy_scores["breakout"] += 1
            score += 1
            confirmations += 1
            reasons.append(f"Volume surge on up move ({vol_ratio:.1f}x)")
        else:
            strategy_scores["breakout"] -= 1
            score -= 1
            confirmations += 1
            reasons.append(f"Volume surge on down move ({vol_ratio:.1f}x)")
    elif vol_ratio < 0.5:
        if score > 0:
            score -= 1
        elif score < 0:
            score += 1
        reasons.append(f"Low volume ({vol_ratio:.1f}x) — signal less reliable")

    # ========== SIGNAL DETERMINATION ==========

    # Check ATR viability
    atr_pct = atr / current_price * 100 if current_price > 0 else 0
    if atr_pct < 0.2:
        reasons.append(f"Low ATR ({atr_pct:.2f}%) — tight range, day trading difficult")
        if abs(score) <= 4:
            score = 0

    if score >= 7 and confirmations >= 3:
        signal, strength = "LONG", "Strong"
    elif score >= 4 and confirmations >= 2:
        signal, strength = "LONG", "Moderate"
    elif score <= -7 and confirmations >= 3:
        signal, strength = "SHORT", "Strong"
    elif score <= -4 and confirmations >= 2:
        signal, strength = "SHORT", "Moderate"
    else:
        signal, strength = "NO_TRADE", "Weak"

    # Determine dominant strategy
    abs_scores = {k: abs(v) for k, v in strategy_scores.items()}
    max_strat = max(abs_scores, key=abs_scores.get)
    strategy_map = {"orb": "ORB", "trend": "Trend Following", "breakout": "Breakout"}
    if abs_scores[max_strat] == 0:
        strategy = "Mixed"
    else:
        non_zero = sum(1 for v in abs_scores.values() if v > 0)
        strategy = strategy_map[max_strat] if non_zero <= 2 else "Mixed"

    # Stop-loss and targets
    support = best_prices["nearest_support"]
    resistance = best_prices["nearest_resistance"]
    condition = best_prices["condition"]

    if signal == "LONG":
        entry = current_price
        stop_loss = max(support, entry - 2 * atr)
        stop_loss = min(stop_loss, entry - 0.75 * atr)
        risk = entry - stop_loss
        if risk < 0.3 * atr:
            signal, strength = "NO_TRADE", "Weak"
            risk = atr
        target_1 = min(resistance, entry + 1.5 * atr)
        if target_1 <= entry:
            target_1 = entry + 1.5 * atr
        target_2 = entry + 2.5 * atr
        target_3 = entry + 3.5 * atr
        reasons.append(f"Entry at market ({current_price:.2f}), stop {stop_loss:.2f}, targets {target_1:.2f}/{target_2:.2f}/{target_3:.2f} [{condition}]")
    elif signal == "SHORT":
        entry = current_price
        stop_loss = min(resistance, entry + 2 * atr)
        stop_loss = max(stop_loss, entry + 0.75 * atr)
        risk = stop_loss - entry
        if risk < 0.3 * atr:
            signal, strength = "NO_TRADE", "Weak"
            risk = atr
        target_1 = max(support, entry - 1.5 * atr)
        if target_1 >= entry:
            target_1 = entry - 1.5 * atr
        target_2 = entry - 2.5 * atr
        target_3 = entry - 3.5 * atr
        reasons.append(f"Entry at market ({current_price:.2f}), stop {stop_loss:.2f}, targets {target_1:.2f}/{target_2:.2f}/{target_3:.2f} [{condition}]")
    else:
        entry = current_price
        stop_loss = current_price - 2 * atr
        target_1 = current_price + 2 * atr
        target_2 = current_price + 3 * atr
        target_3 = current_price + 4 * atr

    risk = abs(entry - stop_loss)
    reward = abs(target_1 - entry)
    rr = reward / risk if risk > 0 else 0

    confidence = min(1.0, abs(score) / 14)

    return DayTradeSignal(
        signal=signal, strength=strength, confidence=confidence,
        strategy=strategy, entry_price=round(entry, 2),
        stop_loss=round(stop_loss, 2), target_1=round(target_1, 2),
        target_2=round(target_2, 2), target_3=round(target_3, 2),
        risk_reward=round(rr, 2), reasons=reasons,
    )


def get_day_trade_microstructure(df: pd.DataFrame) -> dict:
    """Analyze market microstructure for day trading viability."""
    latest = df.iloc[-1]
    recent = df.tail(16)  # Last 16 candles (~4 hours)

    # Trend detection
    ema9 = recent["EMA_9"].values if "EMA_9" in recent.columns else recent["Close"].ewm(span=9).mean().values
    trend_slope = (ema9[-1] - ema9[0]) / len(ema9) if len(ema9) > 1 else 0
    trend = "Uptrend" if trend_slope > 0.05 else "Downtrend" if trend_slope < -0.05 else "Sideways"

    # Volatility regime
    atr = latest.get("ATR_14", 0)
    avg_atr = recent["ATR_14"].mean() if "ATR_14" in recent.columns else atr
    vol_regime = "High" if atr > avg_atr * 1.3 else "Low" if atr < avg_atr * 0.7 else "Normal"

    # Volume profile
    avg_vol = recent["Volume"].mean()
    current_vol = latest["Volume"]
    vol_status = "Above Average" if current_vol > avg_vol * 1.3 else "Below Average" if current_vol < avg_vol * 0.7 else "Average"

    # ADX trend strength
    adx = latest.get("ADX", 0)
    adx_status = "Strong Trend" if adx > 25 else "Weak Trend" if adx < 20 else "Moderate Trend"

    # Consecutive candle direction
    closes = recent["Close"].values
    consecutive = 0
    direction = "up" if closes[-1] > closes[-2] else "down"
    for i in range(len(closes) - 1, 0, -1):
        if direction == "up" and closes[i] > closes[i - 1]:
            consecutive += 1
        elif direction == "down" and closes[i] < closes[i - 1]:
            consecutive += 1
        else:
            break

    # Day-tradability score (0-100)
    tradability = 40
    if vol_regime == "High":
        tradability += 15
    elif vol_regime == "Normal":
        tradability += 5
    elif vol_regime == "Low":
        tradability -= 20

    if vol_status == "Above Average":
        tradability += 15
    elif vol_status == "Average":
        tradability += 5
    elif vol_status == "Below Average":
        tradability -= 15

    if trend != "Sideways":
        tradability += 10

    if adx > 25:
        tradability += 15
    elif adx > 20:
        tradability += 5

    if atr > 0 and latest["Close"] > 0:
        spread_pct = atr / latest["Close"] * 100
        if spread_pct > 0.5:
            tradability += 10
        elif spread_pct > 0.3:
            tradability += 5
        elif spread_pct < 0.2:
            tradability -= 10

    if consecutive >= 3:
        tradability += 5

    tradability = max(0, min(100, tradability))

    avg_range = recent["Candle_Range"].mean() if "Candle_Range" in recent.columns else 0
    current_range = latest.get("Candle_Range", 0)

    return {
        "trend": trend,
        "trend_slope": round(trend_slope, 4),
        "volatility_regime": vol_regime,
        "volume_status": vol_status,
        "adx": round(adx, 1),
        "adx_status": adx_status,
        "avg_candle_range": round(avg_range, 3),
        "current_candle_range": round(current_range, 3),
        "consecutive_candles": consecutive,
        "consecutive_direction": direction,
        "tradability_score": tradability,
        "atr": round(atr, 2),
    }
