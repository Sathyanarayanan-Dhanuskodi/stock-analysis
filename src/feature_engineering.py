import pandas as pd
import numpy as np


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive technical indicators to OHLCV DataFrame."""
    df = df.copy()

    # --- Moving Averages ---
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    # --- RSI (14-period) ---
    df["RSI"] = _compute_rsi(df["Close"], period=14)

    # --- MACD (12, 26, 9) ---
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # --- Bollinger Bands (20, 2) ---
    bb_sma = df["Close"].rolling(window=20).mean()
    bb_std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = bb_sma + 2 * bb_std
    df["BB_Lower"] = bb_sma - 2 * bb_std
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / bb_sma

    # --- Volume ---
    df["Volume_MA"] = df["Volume"].rolling(window=20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA"]

    # --- Rate of Change ---
    df["ROC"] = df["Close"].pct_change(periods=10) * 100

    # --- ATR (Average True Range, 14-period) ---
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(window=14).mean()
    df["ATR_Pct"] = df["ATR"] / df["Close"] * 100

    # --- ADX (Average Directional Index, 14-period) ---
    df["ADX"], df["Plus_DI"], df["Minus_DI"] = _compute_adx(df, period=14)

    # --- Stochastic Oscillator (14, 3, 3) ---
    low_14 = df["Low"].rolling(window=14).min()
    high_14 = df["High"].rolling(window=14).max()
    df["Stoch_K"] = ((df["Close"] - low_14) / (high_14 - low_14)) * 100
    df["Stoch_D"] = df["Stoch_K"].rolling(window=3).mean()

    # --- Williams %R (14-period) ---
    df["Williams_R"] = ((high_14 - df["Close"]) / (high_14 - low_14)) * -100

    # --- CCI (Commodity Channel Index, 20-period) ---
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    tp_sma = typical_price.rolling(window=20).mean()
    tp_mad = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df["CCI"] = (typical_price - tp_sma) / (0.015 * tp_mad)

    # --- OBV (On-Balance Volume) ---
    obv = [0]
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
            obv.append(obv[-1] + df["Volume"].iloc[i])
        elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
            obv.append(obv[-1] - df["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv
    df["OBV_MA"] = df["OBV"].rolling(window=20).mean()

    # --- MFI (Money Flow Index, 14-period) ---
    df["MFI"] = _compute_mfi(df, period=14)

    # --- High-Low Range % ---
    df["HL_Range_Pct"] = (df["High"] - df["Low"]) / df["Close"] * 100

    # --- Daily Returns ---
    df["Returns"] = df["Close"].pct_change()

    # Drop rows with NaN from rolling calculations
    df.dropna(inplace=True)

    return df


def add_market_context(df: pd.DataFrame, context_df: pd.DataFrame) -> pd.DataFrame:
    """Merge market context (Nifty, VIX) into stock DataFrame."""
    if context_df.empty:
        return df

    df = df.copy()
    # Align on date index
    for col in context_df.columns:
        df[col] = context_df[col].reindex(df.index, method="ffill")

    # Normalize Nifty relative to its 20-day mean
    if "Nifty_Close" in df.columns:
        nifty_sma = df["Nifty_Close"].rolling(window=20).mean()
        df["Nifty_Normalized"] = df["Nifty_Close"] / nifty_sma

    df.dropna(inplace=True)
    return df


def add_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged features for XGBoost (which has no sequential memory)."""
    df = df.copy()

    for lag in [1, 3, 5, 10]:
        df[f"Close_Lag_{lag}"] = df["Close"].shift(lag)
        df[f"Returns_Lag_{lag}"] = df["Returns"].shift(lag)

    for lag in [1, 3]:
        df[f"RSI_Lag_{lag}"] = df["RSI"].shift(lag)
        df[f"Volume_Ratio_Lag_{lag}"] = df["Volume_Ratio"].shift(lag)

    df.dropna(inplace=True)
    return df


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _compute_adx(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Average Directional Index with +DI and -DI."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=period).mean()

    return adx, plus_di, minus_di


def _compute_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Money Flow Index."""
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    money_flow = typical_price * df["Volume"]

    positive_flow = pd.Series(0.0, index=df.index)
    negative_flow = pd.Series(0.0, index=df.index)

    tp_diff = typical_price.diff()
    positive_flow = money_flow.where(tp_diff > 0, 0.0)
    negative_flow = money_flow.where(tp_diff < 0, 0.0)

    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()

    mfi = 100 - (100 / (1 + positive_mf / negative_mf))
    return mfi


def get_feature_columns() -> list[str]:
    """Return base feature columns used by LSTM model."""
    return [
        "Open", "High", "Low", "Close", "Volume",
        "SMA_20", "SMA_50", "SMA_200",
        "RSI", "MACD", "MACD_Signal", "MACD_Hist",
        "BB_Upper", "BB_Lower", "BB_Width",
        "Volume_MA", "Volume_Ratio", "ROC",
        "ATR", "ATR_Pct",
        "ADX", "Plus_DI", "Minus_DI",
        "Stoch_K", "Stoch_D",
        "Williams_R", "CCI",
        "OBV", "OBV_MA",
        "MFI", "HL_Range_Pct", "Returns",
    ]


def get_feature_columns_with_context() -> list[str]:
    """Feature columns including market context (for models with context)."""
    return get_feature_columns() + [
        "Nifty_Close", "Nifty_Returns", "Nifty_Normalized",
        "VIX_Close", "VIX_Change",
    ]


def get_xgboost_feature_columns() -> list[str]:
    """Extended feature columns for XGBoost including lags."""
    base = get_feature_columns()
    lags = []
    for lag in [1, 3, 5, 10]:
        lags.append(f"Close_Lag_{lag}")
        lags.append(f"Returns_Lag_{lag}")
    for lag in [1, 3]:
        lags.append(f"RSI_Lag_{lag}")
        lags.append(f"Volume_Ratio_Lag_{lag}")
    return base + lags
