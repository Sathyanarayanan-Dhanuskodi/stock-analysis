import pandas as pd
import numpy as np
from dataclasses import dataclass, field


@dataclass
class SwingSignal:
    signal: str  # "BUY" | "SELL" | "HOLD"
    strength: str  # "Strong" | "Moderate" | "Weak"
    confidence: float  # 0.0 to 1.0
    reasons: list[str] = field(default_factory=list)


@dataclass
class TradeSetup:
    signal: str
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    target_3: float
    risk_reward_1: float
    risk_reward_2: float
    risk_reward_3: float
    position_size_pct: float
    risk_amount: float


def calculate_pivot_points(df: pd.DataFrame, method: str = "standard") -> dict:
    """Calculate pivot points from yesterday's OHLC."""
    last = df.iloc[-1]
    h, l, c = last["High"], last["Low"], last["Close"]

    if method == "fibonacci":
        pp = (h + l + c) / 3
        r = h - l
        return {
            "PP": pp, "R1": pp + 0.382 * r, "R2": pp + 0.618 * r, "R3": pp + r,
            "S1": pp - 0.382 * r, "S2": pp - 0.618 * r, "S3": pp - r,
            "method": method,
        }
    elif method == "camarilla":
        pp = (h + l + c) / 3
        r = h - l
        return {
            "PP": pp, "R1": c + r * 1.1 / 12, "R2": c + r * 1.1 / 6, "R3": c + r * 1.1 / 4,
            "S1": c - r * 1.1 / 12, "S2": c - r * 1.1 / 6, "S3": c - r * 1.1 / 4,
            "method": method,
        }
    elif method == "woodie":
        pp = (h + l + 2 * c) / 4
        return {
            "PP": pp, "R1": 2 * pp - l, "R2": pp + (h - l), "R3": h + 2 * (pp - l),
            "S1": 2 * pp - h, "S2": pp - (h - l), "S3": l - 2 * (h - pp),
            "method": method,
        }
    else:  # standard
        pp = (h + l + c) / 3
        return {
            "PP": pp, "R1": 2 * pp - l, "R2": pp + (h - l), "R3": h + 2 * (pp - l),
            "S1": 2 * pp - h, "S2": pp - (h - l), "S3": l - 2 * (h - pp),
            "method": "standard",
        }


def calculate_support_resistance(
    df: pd.DataFrame, lookback: int = 90, sensitivity: float = 0.02, min_touches: int = 2,
) -> dict:
    """Identify S/R levels using swing highs/lows with clustering."""
    data = df.tail(lookback)
    window = 5

    # Find swing highs and lows
    swing_highs = []
    swing_lows = []

    for i in range(window, len(data) - window):
        high_val = data["High"].iloc[i]
        low_val = data["Low"].iloc[i]

        if all(high_val >= data["High"].iloc[i - j] for j in range(1, window + 1)) and \
           all(high_val >= data["High"].iloc[i + j] for j in range(1, window + 1)):
            swing_highs.append(high_val)

        if all(low_val <= data["Low"].iloc[i - j] for j in range(1, window + 1)) and \
           all(low_val <= data["Low"].iloc[i + j] for j in range(1, window + 1)):
            swing_lows.append(low_val)

    # Cluster nearby levels
    def cluster_levels(levels, threshold_pct):
        if not levels:
            return []
        levels = sorted(levels)
        clusters = []
        current_cluster = [levels[0]]
        for level in levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] < threshold_pct:
                current_cluster.append(level)
            else:
                clusters.append(current_cluster)
                current_cluster = [level]
        clusters.append(current_cluster)
        return [{"price": np.mean(c), "touches": len(c),
                 "strength": "Strong" if len(c) >= 3 else "Moderate" if len(c) >= 2 else "Weak"}
                for c in clusters if len(c) >= min_touches]

    resistance_levels = cluster_levels(swing_highs, sensitivity)
    support_levels = cluster_levels(swing_lows, sensitivity)

    current_price = data["Close"].iloc[-1]
    nearest_r = min((r["price"] for r in resistance_levels if r["price"] > current_price), default=current_price * 1.05)
    nearest_s = max((s["price"] for s in support_levels if s["price"] < current_price), default=current_price * 0.95)

    return {
        "resistance_levels": sorted(resistance_levels, key=lambda x: x["price"]),
        "support_levels": sorted(support_levels, key=lambda x: x["price"]),
        "nearest_resistance": nearest_r,
        "nearest_support": nearest_s,
        "current_price": current_price,
    }


def calculate_fibonacci_retracements(df: pd.DataFrame, lookback: int = 90) -> dict:
    """Compute Fibonacci retracement and extension levels."""
    data = df.tail(lookback)
    swing_high = data["High"].max()
    swing_low = data["Low"].min()

    high_idx = data["High"].idxmax()
    low_idx = data["Low"].idxmin()
    trend = "uptrend" if low_idx < high_idx else "downtrend"

    diff = swing_high - swing_low

    if trend == "uptrend":
        retracements = {
            "23.6%": swing_high - 0.236 * diff,
            "38.2%": swing_high - 0.382 * diff,
            "50.0%": swing_high - 0.500 * diff,
            "61.8%": swing_high - 0.618 * diff,
            "78.6%": swing_high - 0.786 * diff,
        }
        extensions = {
            "127.2%": swing_high + 0.272 * diff,
            "141.4%": swing_high + 0.414 * diff,
            "161.8%": swing_high + 0.618 * diff,
            "200.0%": swing_high + 1.000 * diff,
        }
    else:
        retracements = {
            "23.6%": swing_low + 0.236 * diff,
            "38.2%": swing_low + 0.382 * diff,
            "50.0%": swing_low + 0.500 * diff,
            "61.8%": swing_low + 0.618 * diff,
            "78.6%": swing_low + 0.786 * diff,
        }
        extensions = {
            "127.2%": swing_low - 0.272 * diff,
            "141.4%": swing_low - 0.414 * diff,
            "161.8%": swing_low - 0.618 * diff,
            "200.0%": swing_low - 1.000 * diff,
        }

    current_price = data["Close"].iloc[-1]
    fib_levels = sorted(retracements.values())
    current_zone = "Below all levels"
    for i, level in enumerate(fib_levels):
        if current_price < level:
            current_zone = f"Near {list(retracements.keys())[i]} level"
            break
    else:
        current_zone = "Above all levels"

    return {
        "swing_high": swing_high, "swing_low": swing_low, "trend": trend,
        "retracements": retracements, "extensions": extensions,
        "current_fib_zone": current_zone,
    }


def calculate_atr_stop_loss(df: pd.DataFrame, atr_period: int = 14, atr_multiplier: float = 2.0) -> dict:
    """ATR-based dynamic stop loss calculation."""
    if "ATR" not in df.columns:
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift(1)).abs()
        low_close = (df["Low"] - df["Close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=atr_period).mean().iloc[-1]
    else:
        atr = df["ATR"].iloc[-1]

    current_price = df["Close"].iloc[-1]
    atr_pct = atr / current_price * 100

    # Volatility regime
    atr_50d = df["ATR"].rolling(50).mean().iloc[-1] if "ATR" in df.columns else atr
    regime = "high" if atr > atr_50d * 1.3 else "low" if atr < atr_50d * 0.7 else "normal"

    return {
        "current_atr": atr,
        "atr_pct": atr_pct,
        "stop_loss_conservative": current_price - 1.5 * atr,
        "stop_loss_standard": current_price - 2.0 * atr,
        "stop_loss_wide": current_price - 3.0 * atr,
        "volatility_regime": regime,
    }


def generate_swing_signals(df: pd.DataFrame) -> SwingSignal:
    """Multi-condition swing trade signal generator with stricter thresholds.

    Requires confluence of trend + momentum + volume for reliable signals.
    Uses multi-day confirmation and divergence detection.
    """
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    prev2 = df.iloc[-3] if len(df) > 2 else prev
    recent_5 = df.tail(5)
    recent_10 = df.tail(10)

    score = 0
    reasons = []
    confirmations = 0  # Independent factor count

    # --- TREND STRUCTURE (most important for swing) ---

    # SMA trend alignment — full stack alignment gets higher weight
    if "SMA_20" in df.columns and "SMA_50" in df.columns:
        sma200 = latest.get("SMA_200", latest["SMA_50"])
        if latest["Close"] > latest["SMA_20"] > latest["SMA_50"] > sma200:
            score += 2
            confirmations += 1
            reasons.append("Full trend alignment: Price > SMA20 > SMA50 > SMA200 (strong uptrend)")
        elif latest["Close"] > latest["SMA_20"] > latest["SMA_50"]:
            score += 1
            confirmations += 1
            reasons.append("Price above SMA20 > SMA50 (uptrend)")
        elif latest["Close"] < latest["SMA_20"] < latest["SMA_50"]:
            score -= 1
            confirmations += 1
            reasons.append("Price below SMA20 < SMA50 (downtrend)")
        elif latest["Close"] < latest["SMA_20"] < latest["SMA_50"] < sma200:
            score -= 2
            confirmations += 1
            reasons.append("Full trend alignment: Price < SMA20 < SMA50 < SMA200 (strong downtrend)")

    # SMA 20 slope over last 5 days (trend direction persistence)
    if "SMA_20" in df.columns and len(df) >= 5:
        sma20_slope = (latest["SMA_20"] - df.iloc[-6]["SMA_20"]) / df.iloc[-6]["SMA_20"] * 100
        if sma20_slope > 0.5:
            score += 1
            reasons.append(f"SMA20 rising strongly ({sma20_slope:+.2f}%/5d)")
        elif sma20_slope < -0.5:
            score -= 1
            reasons.append(f"SMA20 falling strongly ({sma20_slope:+.2f}%/5d)")

    # ADX trend strength + DI directional
    if "ADX" in df.columns and "Plus_DI" in df.columns and "Minus_DI" in df.columns:
        adx = latest["ADX"]
        if adx > 25:
            if latest["Plus_DI"] > latest["Minus_DI"]:
                score += 2
                confirmations += 1
                reasons.append(f"Strong uptrend confirmed (ADX={adx:.0f}, +DI={latest['Plus_DI']:.0f} > -DI={latest['Minus_DI']:.0f})")
            else:
                score -= 2
                confirmations += 1
                reasons.append(f"Strong downtrend confirmed (ADX={adx:.0f}, -DI={latest['Minus_DI']:.0f} > +DI={latest['Plus_DI']:.0f})")
        elif adx < 15:
            reasons.append(f"No trend (ADX={adx:.0f}) — ranging market, avoid swings")

    # --- MOMENTUM INDICATORS ---

    # RSI conditions — with trend context
    if "RSI" in df.columns:
        rsi = latest["RSI"]
        rsi_prev = prev["RSI"]
        # RSI divergence detection (price makes new low but RSI doesn't)
        if len(df) >= 20:
            price_min_10 = recent_10["Close"].min()
            rsi_min_10 = recent_10["RSI"].min()
            price_min_prev = df.tail(20).head(10)["Close"].min()
            rsi_min_prev = df.tail(20).head(10)["RSI"].min()
            if price_min_10 < price_min_prev and rsi_min_10 > rsi_min_prev:
                score += 2
                confirmations += 1
                reasons.append(f"Bullish RSI divergence (price lower low, RSI higher low)")
            elif price_min_10 > price_min_prev and rsi_min_10 < rsi_min_prev:
                score -= 1
                reasons.append(f"Bearish RSI divergence detected")

        if rsi < 30 and rsi > rsi_prev:
            score += 2
            confirmations += 1
            reasons.append(f"RSI oversold and turning up ({rsi:.1f}, was {rsi_prev:.1f})")
        elif rsi < 30:
            score += 1
            reasons.append(f"RSI oversold ({rsi:.1f}, potential reversal)")
        elif rsi > 70 and rsi < rsi_prev:
            score -= 2
            confirmations += 1
            reasons.append(f"RSI overbought and turning down ({rsi:.1f}, was {rsi_prev:.1f})")
        elif rsi > 70:
            score -= 1
            reasons.append(f"RSI overbought ({rsi:.1f})")
        elif 40 < rsi < 60 and rsi > rsi_prev:
            score += 1
            reasons.append(f"RSI in healthy zone and rising ({rsi:.1f})")

    # MACD crossover — fresh crossovers get more weight
    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        macd_diff = latest["MACD"] - latest["MACD_Signal"]
        prev_diff = prev["MACD"] - prev["MACD_Signal"]
        if macd_diff > 0 and prev_diff <= 0:
            score += 2
            confirmations += 1
            reasons.append("MACD bullish crossover (fresh)")
        elif macd_diff > 0 and macd_diff > prev_diff:
            score += 1
            reasons.append("MACD momentum building (histogram expanding)")
        elif macd_diff < 0 and prev_diff >= 0:
            score -= 2
            confirmations += 1
            reasons.append("MACD bearish crossover (fresh)")
        elif macd_diff < 0 and macd_diff < prev_diff:
            score -= 1
            reasons.append("MACD bearish momentum building")

    # Stochastic crossover — only in extreme zones
    if "Stoch_K" in df.columns and "Stoch_D" in df.columns:
        if latest["Stoch_K"] > latest["Stoch_D"] and prev["Stoch_K"] <= prev["Stoch_D"] and latest["Stoch_K"] < 25:
            score += 2
            confirmations += 1
            reasons.append(f"Stochastic bullish crossover in oversold zone (K={latest['Stoch_K']:.0f})")
        elif latest["Stoch_K"] < latest["Stoch_D"] and prev["Stoch_K"] >= prev["Stoch_D"] and latest["Stoch_K"] > 75:
            score -= 2
            confirmations += 1
            reasons.append(f"Stochastic bearish crossover in overbought zone (K={latest['Stoch_K']:.0f})")

    # --- PRICE ACTION ---

    # Bollinger Band bounces — require volume confirmation
    if "BB_Lower" in df.columns and "BB_Upper" in df.columns:
        if latest["Close"] <= latest["BB_Lower"] * 1.005 and latest["Close"] > prev["Close"]:
            score += 1
            reasons.append("Price bouncing off lower Bollinger Band")
        elif latest["Close"] >= latest["BB_Upper"] * 0.995 and latest["Close"] < prev["Close"]:
            score -= 1
            reasons.append("Price rejecting at upper Bollinger Band")

    # Multi-day price action (last 5 days trend)
    if len(recent_5) >= 5:
        up_days = sum(1 for i in range(1, len(recent_5)) if recent_5["Close"].iloc[i] > recent_5["Close"].iloc[i-1])
        if up_days >= 4:
            score += 1
            reasons.append(f"{up_days}/4 recent days green — bullish momentum")
        elif up_days <= 1:
            score -= 1
            reasons.append(f"{4 - up_days}/4 recent days red — bearish momentum")

    # --- VOLUME CONFIRMATION ---
    if "Volume_Ratio" in df.columns:
        vol_r = latest["Volume_Ratio"]
        if vol_r > 2.0 and latest["Close"] > prev["Close"]:
            score += 2
            confirmations += 1
            reasons.append(f"Strong volume surge on up day ({vol_r:.1f}x avg) — institutional buying")
        elif vol_r > 1.5 and latest["Close"] > prev["Close"]:
            score += 1
            reasons.append(f"Above-average volume on up day ({vol_r:.1f}x)")
        elif vol_r > 2.0 and latest["Close"] < prev["Close"]:
            score -= 2
            confirmations += 1
            reasons.append(f"Strong volume surge on down day ({vol_r:.1f}x avg) — institutional selling")
        elif vol_r > 1.5 and latest["Close"] < prev["Close"]:
            score -= 1
            reasons.append(f"Above-average volume on down day ({vol_r:.1f}x)")
        elif vol_r < 0.5:
            reasons.append(f"Low volume ({vol_r:.1f}x) — moves may not sustain")

    # MFI (money flow)
    if "MFI" in df.columns:
        mfi = latest["MFI"]
        if mfi < 20:
            score += 1
            reasons.append(f"MFI oversold ({mfi:.0f}) — money flowing out, reversal possible")
        elif mfi > 80:
            score -= 1
            reasons.append(f"MFI overbought ({mfi:.0f}) — excess buying, correction possible")

    # OBV trend (on-balance volume confirms price trend)
    if "OBV" in df.columns and "OBV_MA" in df.columns:
        if latest["OBV"] > latest["OBV_MA"] and latest["Close"] > prev["Close"]:
            score += 1
            reasons.append("OBV confirming uptrend (volume supports price rise)")
        elif latest["OBV"] < latest["OBV_MA"] and latest["Close"] < prev["Close"]:
            score -= 1
            reasons.append("OBV confirming downtrend (volume supports price fall)")

    # --- SIGNAL DETERMINATION ---
    # Stricter: require score >= 4 for Strong BUY, >= 2 for Moderate
    # Also require minimum confirmations for Strong signals
    if score >= 5 and confirmations >= 3:
        signal, strength = "BUY", "Strong"
    elif score >= 3 and confirmations >= 2:
        signal, strength = "BUY", "Moderate"
    elif score <= -5 and confirmations >= 3:
        signal, strength = "SELL", "Strong"
    elif score <= -3 and confirmations >= 2:
        signal, strength = "SELL", "Moderate"
    elif score >= 1:
        signal, strength = "BUY", "Weak"
    elif score <= -1:
        signal, strength = "SELL", "Weak"
    else:
        signal, strength = "HOLD", "Weak"

    max_score = 14  # approximate max possible score
    confidence = min(1.0, abs(score) / max_score)

    return SwingSignal(signal=signal, strength=strength, confidence=confidence, reasons=reasons)


def calculate_trade_setup(
    df: pd.DataFrame,
    signal: SwingSignal,
    capital: float = 100000,
    max_risk_pct: float = 0.02,
    ai_predictions: list[float] | None = None,
    seasonality_avg: float | None = None,
) -> TradeSetup:
    """Calculate full trade setup with entry, stops, targets, position sizing.

    Uses S/R levels, Fibonacci, AI predictions, and seasonality to set
    realistic targets instead of pure R:R multiples.
    """
    entry_price = df["Close"].iloc[-1]
    atr_data = calculate_atr_stop_loss(df)
    fib_data = calculate_fibonacci_retracements(df)
    sr_data = calculate_support_resistance(df)
    atr = atr_data["current_atr"]

    ai_target = None
    if ai_predictions and len(ai_predictions) >= 5:
        ai_5d = ai_predictions[4]
        ai_target = ai_5d

    season_bias = 0.0
    if seasonality_avg is not None:
        season_bias = seasonality_avg / 100.0

    if signal.signal == "BUY":
        stop_loss = atr_data["stop_loss_standard"]
        risk = entry_price - stop_loss

        nearest_r = sr_data["nearest_resistance"]
        extensions = sorted(fib_data["extensions"].values())
        fib_ext_1 = next((e for e in extensions if e > entry_price), None)

        candidates_t1 = []
        if nearest_r > entry_price + risk:
            candidates_t1.append(nearest_r)
        if fib_ext_1 and fib_ext_1 > entry_price + risk:
            candidates_t1.append(fib_ext_1)
        if ai_target and ai_target > entry_price + risk:
            candidates_t1.append(ai_target)

        if candidates_t1:
            raw_t1 = min(candidates_t1)
            if season_bias < -0.005:
                raw_t1 = entry_price + max((raw_t1 - entry_price) * 0.85, risk)
            target_1 = raw_t1
        else:
            target_1 = entry_price + 1.5 * risk

        target_1 = max(target_1, entry_price + risk)

        candidates_t2 = [e for e in extensions if e > target_1] if extensions else []
        res_above_t1 = [r["price"] for r in sr_data["resistance_levels"] if r["price"] > target_1]
        candidates_t2.extend(res_above_t1)
        if ai_target and ai_target > target_1:
            candidates_t2.append(ai_target)

        if candidates_t2:
            target_2 = min(candidates_t2)
        else:
            target_2 = entry_price + 2 * risk

        if ai_predictions and len(ai_predictions) >= 7:
            ai_7d = max(ai_predictions[:7])
            if ai_7d > target_2:
                target_3 = ai_7d
            else:
                target_3 = entry_price + 3 * risk
        else:
            target_3 = entry_price + 3 * risk

        target_2 = max(target_2, target_1 + 0.5 * risk)
        target_3 = max(target_3, target_2 + 0.5 * risk)

    elif signal.signal == "SELL":
        stop_loss = entry_price + atr * 2
        risk = stop_loss - entry_price

        nearest_s = sr_data["nearest_support"]
        extensions = sorted(fib_data["extensions"].values(), reverse=True)
        fib_ext_1 = next((e for e in extensions if e < entry_price), None)

        candidates_t1 = []
        if nearest_s < entry_price - risk:
            candidates_t1.append(nearest_s)
        if fib_ext_1 and fib_ext_1 < entry_price - risk:
            candidates_t1.append(fib_ext_1)
        if ai_target and ai_target < entry_price - risk:
            candidates_t1.append(ai_target)

        if candidates_t1:
            raw_t1 = max(candidates_t1)
            if season_bias > 0.005:
                raw_t1 = entry_price - max((entry_price - raw_t1) * 0.85, risk)
            target_1 = raw_t1
        else:
            target_1 = entry_price - 1.5 * risk

        target_1 = min(target_1, entry_price - risk)

        sup_below_t1 = [s["price"] for s in sr_data["support_levels"] if s["price"] < target_1]
        candidates_t2 = [e for e in extensions if e < target_1] if extensions else []
        candidates_t2.extend(sup_below_t1)
        if ai_target and ai_target < target_1:
            candidates_t2.append(ai_target)

        if candidates_t2:
            target_2 = max(candidates_t2)
        else:
            target_2 = entry_price - 2 * risk

        if ai_predictions and len(ai_predictions) >= 7:
            ai_7d = min(ai_predictions[:7])
            if ai_7d < target_2:
                target_3 = ai_7d
            else:
                target_3 = entry_price - 3 * risk
        else:
            target_3 = entry_price - 3 * risk

        target_2 = min(target_2, target_1 - 0.5 * risk)
        target_3 = min(target_3, target_2 - 0.5 * risk)

    else:  # HOLD
        stop_loss = atr_data["stop_loss_standard"]
        risk = abs(entry_price - stop_loss)
        target_1 = entry_price + atr
        target_2 = entry_price + 2 * atr
        target_3 = entry_price + 3 * atr

    risk = max(abs(entry_price - stop_loss), 0.01)

    def rr(target):
        return round(abs(target - entry_price) / risk, 2)

    risk_amount = capital * max_risk_pct
    shares = int(risk_amount / risk) if risk > 0 else 0
    position_value = shares * entry_price
    position_pct = (position_value / capital * 100) if capital > 0 else 0

    return TradeSetup(
        signal=signal.signal,
        entry_price=round(entry_price, 2),
        stop_loss=round(stop_loss, 2),
        target_1=round(target_1, 2),
        target_2=round(target_2, 2),
        target_3=round(target_3, 2),
        risk_reward_1=rr(target_1),
        risk_reward_2=rr(target_2),
        risk_reward_3=rr(target_3),
        position_size_pct=round(position_pct, 1),
        risk_amount=round(risk_amount, 2),
    )


def identify_swing_patterns(df: pd.DataFrame, lookback: int = 30) -> list[dict]:
    """Detect classic swing trading chart patterns."""
    data = df.tail(lookback)
    patterns = []

    closes = data["Close"].values
    opens = data["Open"].values
    highs = data["High"].values
    lows = data["Low"].values

    # Bullish Engulfing
    if len(closes) >= 2:
        if closes[-2] < opens[-2] and closes[-1] > opens[-1]:
            if opens[-1] <= closes[-2] and closes[-1] >= opens[-2]:
                patterns.append({
                    "pattern": "Bullish Engulfing",
                    "confidence": 0.7,
                    "implication": "Bullish reversal - previous bearish candle fully engulfed",
                })

    # Bearish Engulfing
    if len(closes) >= 2:
        if closes[-2] > opens[-2] and closes[-1] < opens[-1]:
            if opens[-1] >= closes[-2] and closes[-1] <= opens[-2]:
                patterns.append({
                    "pattern": "Bearish Engulfing",
                    "confidence": 0.7,
                    "implication": "Bearish reversal - previous bullish candle fully engulfed",
                })

    # Inside Bar (consolidation)
    if len(highs) >= 2:
        if highs[-1] < highs[-2] and lows[-1] > lows[-2]:
            patterns.append({
                "pattern": "Inside Bar",
                "confidence": 0.6,
                "implication": "Consolidation - breakout imminent in either direction",
            })

    # Higher Highs + Higher Lows (uptrend)
    if len(highs) >= 10:
        recent_highs = [highs[i] for i in range(len(highs) - 10, len(highs), 2)]
        recent_lows = [lows[i] for i in range(len(lows) - 10, len(lows), 2)]
        if all(recent_highs[i] > recent_highs[i-1] for i in range(1, len(recent_highs))):
            if all(recent_lows[i] > recent_lows[i-1] for i in range(1, len(recent_lows))):
                patterns.append({
                    "pattern": "Higher Highs & Higher Lows",
                    "confidence": 0.8,
                    "implication": "Strong uptrend confirmation",
                })

    # Lower Highs + Lower Lows (downtrend)
    if len(highs) >= 10:
        recent_highs = [highs[i] for i in range(len(highs) - 10, len(highs), 2)]
        recent_lows = [lows[i] for i in range(len(lows) - 10, len(lows), 2)]
        if all(recent_highs[i] < recent_highs[i-1] for i in range(1, len(recent_highs))):
            if all(recent_lows[i] < recent_lows[i-1] for i in range(1, len(recent_lows))):
                patterns.append({
                    "pattern": "Lower Highs & Lower Lows",
                    "confidence": 0.8,
                    "implication": "Strong downtrend confirmation",
                })

    # Double Bottom (W pattern) - simplified
    if len(lows) >= 20:
        mid = len(lows) // 2
        first_low = min(lows[:mid])
        second_low = min(lows[mid:])
        if abs(first_low - second_low) / first_low < 0.02:  # Within 2%
            middle_high = max(highs[mid-5:mid+5]) if mid > 5 else max(highs[:mid])
            if middle_high > first_low * 1.03:  # At least 3% bounce between lows
                patterns.append({
                    "pattern": "Double Bottom (W)",
                    "confidence": 0.65,
                    "implication": "Bullish reversal - two similar lows with recovery",
                })

    # Double Top (M pattern) - simplified
    if len(highs) >= 20:
        mid = len(highs) // 2
        first_high = max(highs[:mid])
        second_high = max(highs[mid:])
        if abs(first_high - second_high) / first_high < 0.02:
            middle_low = min(lows[mid-5:mid+5]) if mid > 5 else min(lows[:mid])
            if middle_low < first_high * 0.97:
                patterns.append({
                    "pattern": "Double Top (M)",
                    "confidence": 0.65,
                    "implication": "Bearish reversal - two similar highs with pullback",
                })

    return patterns
