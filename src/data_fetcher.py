import yfinance as yf
import pandas as pd
import streamlit as st
import time
from datetime import datetime, timedelta


def _yf_retry(func, max_retries=3):
    """Retry a yfinance call with exponential backoff on rate limits."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            msg = str(e).lower()
            if "too many requests" in msg or "rate limit" in msg or "429" in msg:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
            raise
    raise RuntimeError("Max retries exceeded")


POPULAR_INDIAN_STOCKS = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "HDFCBANK.NS": "HDFC Bank",
    "INFY.NS": "Infosys",
    "ICICIBANK.NS": "ICICI Bank",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ITC.NS": "ITC Limited",
    "SBIN.NS": "State Bank of India",
    "BHARTIARTL.NS": "Bharti Airtel",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "LT.NS": "Larsen & Toubro",
    "AXISBANK.NS": "Axis Bank",
    "WIPRO.NS": "Wipro",
    "ASIANPAINT.NS": "Asian Paints",
    "MARUTI.NS": "Maruti Suzuki",
    "TATAMOTORS.NS": "Tata Motors",
    "SUNPHARMA.NS": "Sun Pharma",
    "TITAN.NS": "Titan Company",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "POWERGRID.NS": "Power Grid Corp",
    "NTPC.NS": "NTPC Limited",
    "TATASTEEL.NS": "Tata Steel",
    "ONGC.NS": "ONGC",
    "HCLTECH.NS": "HCL Technologies",
    "BAJFINANCE.NS": "Bajaj Finance",
}

MARKET_INDICES = {
    "^NSEI": "Nifty 50",
    "^INDIAVIX": "India VIX",
    "^CNXBANK": "Nifty Bank",
    "^CNXIT": "Nifty IT",
    "^CNXPHARMA": "Nifty Pharma",
    "^CNXFMCG": "Nifty FMCG",
    "^CNXAUTO": "Nifty Auto",
    "^CNXMETAL": "Nifty Metal",
    "^CNXENERGY": "Nifty Energy",
}

SECTOR_STOCKS = {
    "IT": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
    "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "KOTAKBANK.NS"],
    "FMCG": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS"],
    "Auto": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS"],
    "Pharma": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS"],
    "Energy": ["RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS"],
    "Metals": ["TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "VEDL.NS"],
    "Infrastructure": ["LT.NS", "ULTRACEMCO.NS", "POWERGRID.NS", "NTPC.NS"],
}


@st.cache_data(ttl=60, show_spinner=False)
def fetch_stock_data(ticker: str, period_years: int = 5) -> pd.DataFrame:
    """Fetch historical OHLCV data for a given stock ticker."""
    end_date = datetime.now() + timedelta(days=1)
    start_date = end_date - timedelta(days=period_years * 365 + 1)

    stock = yf.Ticker(ticker)
    df = _yf_retry(lambda: stock.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d")))

    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'. Check the ticker symbol.")

    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)

    return df


@st.cache_data(ttl=60, show_spinner=False)
def fetch_market_context(period_years: int = 5) -> pd.DataFrame:
    """Fetch Nifty 50 and India VIX data for market context features."""
    end_date = datetime.now() + timedelta(days=1)
    start_date = end_date - timedelta(days=period_years * 365 + 1)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    result = pd.DataFrame()

    # Fetch Nifty 50
    try:
        nifty = _yf_retry(lambda: yf.Ticker("^NSEI").history(start=start_str, end=end_str))
        if not nifty.empty:
            nifty.index = pd.to_datetime(nifty.index).tz_localize(None)
            result["Nifty_Close"] = nifty["Close"]
            result["Nifty_Returns"] = nifty["Close"].pct_change()
    except Exception:
        pass

    # Fetch India VIX
    try:
        vix = _yf_retry(lambda: yf.Ticker("^INDIAVIX").history(start=start_str, end=end_str))
        if not vix.empty:
            vix.index = pd.to_datetime(vix.index).tz_localize(None)
            result["VIX_Close"] = vix["Close"]
            result["VIX_Change"] = vix["Close"].pct_change()
    except Exception:
        pass

    if not result.empty:
        result.ffill(inplace=True)
        result.dropna(inplace=True)

    return result


@st.cache_data(ttl=60, show_spinner=False)
def fetch_multiple_stocks(tickers: tuple | list, period_years: int = 2) -> pd.DataFrame:
    """Fetch Close prices for multiple tickers, aligned on common dates."""
    if isinstance(tickers, list):
        tickers = tuple(tickers)
    end_date = datetime.now() + timedelta(days=1)
    start_date = end_date - timedelta(days=period_years * 365 + 1)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    prices = pd.DataFrame()
    for i, ticker in enumerate(tickers):
        try:
            if i > 0:
                time.sleep(0.5)
            df = _yf_retry(lambda t=ticker: yf.Ticker(t).history(start=start_str, end=end_str))
            if not df.empty:
                df.index = pd.to_datetime(df.index).tz_localize(None)
                prices[ticker] = df["Close"]
        except Exception:
            continue

    if not prices.empty:
        prices.ffill(limit=3, inplace=True)
        prices.dropna(inplace=True)

    return prices


@st.cache_data(ttl=30, show_spinner=False)
def get_stock_info(ticker: str) -> dict:
    """Get basic stock information."""
    stock = yf.Ticker(ticker)
    try:
        info = _yf_retry(lambda: stock.info) or {}
    except Exception:
        info = {}
    price = (info.get("currentPrice")
             or info.get("regularMarketPrice")
             or info.get("previousClose")
             or 0)
    return {
        "name": info.get("longName", ticker.replace(".NS", "")),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "market_cap": info.get("marketCap", 0),
        "current_price": price,
        "currency": info.get("currency", "INR"),
        # Real-time intraday fields
        "prev_close": info.get("regularMarketPreviousClose", 0),
        "open_price": info.get("regularMarketOpen", 0),
        "day_high": info.get("regularMarketDayHigh", 0),
        "day_low": info.get("regularMarketDayLow", 0),
        "week_52_high": info.get("fiftyTwoWeekHigh", 0),
        "week_52_low": info.get("fiftyTwoWeekLow", 0),
    }


@st.cache_data(ttl=300, show_spinner=False)
def get_stock_news(ticker: str) -> list[dict]:
    """Get recent news for a stock ticker."""
    stock = yf.Ticker(ticker)
    news = _yf_retry(lambda: stock.news) or []
    results = []
    for item in news[:10]:
        content = item.get("content", {})
        if content:
            title = content.get("title", "")
            publisher = (content.get("provider") or {}).get("displayName", "")
            link = (content.get("canonicalUrl") or {}).get("url", "")
            published = content.get("pubDate", "")
        else:
            title = item.get("title", "")
            publisher = item.get("publisher", "")
            link = item.get("link", "")
            published = item.get("providerPublishTime", "")
        results.append({"title": title, "publisher": publisher, "link": link, "published": published})
    return results


@st.cache_data(ttl=60, show_spinner=False)
def fetch_intraday_data(ticker: str, interval: str = "5m", period: str = "5d") -> pd.DataFrame:
    """Fetch intraday OHLCV data for scalping analysis.

    Args:
        ticker: Stock ticker symbol
        interval: Candle interval - "1m", "2m", "5m", "15m", "30m", "1h"
        period: How far back - "1d", "5d", "1mo" (yfinance limits: 1m=7d, 5m=60d, 15m=60d)
    """
    stock = yf.Ticker(ticker)
    df = _yf_retry(lambda: stock.history(interval=interval, period=period))

    if df.empty:
        raise ValueError(f"No intraday data for '{ticker}'. Market may be closed.")

    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("Asia/Kolkata").tz_localize(None)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)

    return df


def get_sector_for_stock(ticker: str) -> str | None:
    """Find which sector a stock belongs to."""
    for sector, stocks in SECTOR_STOCKS.items():
        if ticker in stocks:
            return sector
    return None


def get_sector_peers(ticker: str) -> list[str]:
    """Get sector peer stocks for a given ticker."""
    sector = get_sector_for_stock(ticker)
    if sector:
        return [s for s in SECTOR_STOCKS[sector] if s != ticker]
    return []


@st.cache_data(ttl=1800, show_spinner=False)
def get_stock_fundamentals(ticker: str) -> dict:
    """Fetch comprehensive fundamental data from yfinance."""
    stock = yf.Ticker(ticker)
    info = _yf_retry(lambda: stock.info)
    return {
        # Valuation
        "market_cap": info.get("marketCap"),
        "enterprise_value": info.get("enterpriseValue"),
        "trailing_pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "peg_ratio": info.get("pegRatio"),
        "price_to_book": info.get("priceToBook"),
        "price_to_sales": info.get("priceToSalesTrailing12Months"),
        "ev_to_ebitda": info.get("enterpriseToEbitda"),
        # Profitability
        "profit_margin": info.get("profitMargins"),
        "operating_margin": info.get("operatingMargins"),
        "roe": info.get("returnOnEquity"),
        "roa": info.get("returnOnAssets"),
        "gross_margin": info.get("grossMargins"),
        # Financial Health
        "debt_to_equity": info.get("debtToEquity"),
        "current_ratio": info.get("currentRatio"),
        "quick_ratio": info.get("quickRatio"),
        "total_debt": info.get("totalDebt"),
        "total_cash": info.get("totalCash"),
        # Dividends
        "dividend_yield": info.get("dividendYield"),
        "dividend_rate": info.get("dividendRate"),
        "payout_ratio": info.get("payoutRatio"),
        # Growth
        "revenue_growth": info.get("revenueGrowth"),
        "earnings_growth": info.get("earningsGrowth"),
        "revenue": info.get("totalRevenue"),
        "earnings": info.get("netIncomeToCommon"),
        "eps": info.get("trailingEps"),
        "forward_eps": info.get("forwardEps"),
        # Trading Info
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow"),
        "50d_avg": info.get("fiftyDayAverage"),
        "200d_avg": info.get("twoHundredDayAverage"),
        "avg_volume": info.get("averageVolume"),
        "beta": info.get("beta"),
        "current_price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
        "name": info.get("longName", ticker),
    }


def scan_swing_opportunities(tickers: list[str]) -> list[dict]:
    """Scan multiple stocks for swing trading opportunities.

    Returns a list of dicts sorted by confidence (best first).
    """
    from src.feature_engineering import add_technical_indicators
    from src.swing_trading import generate_swing_signals

    results = []
    for i, t in enumerate(tickers):
        try:
            if i > 0:
                time.sleep(0.5)
            df = fetch_stock_data(t, period_years=1)
            df = add_technical_indicators(df)
            signal = generate_swing_signals(df)
            latest = df.iloc[-1]
            results.append({
                "ticker": t,
                "name": POPULAR_INDIAN_STOCKS.get(t, t.replace(".NS", "")),
                "price": latest["Close"],
                "signal": signal.signal,
                "strength": signal.strength,
                "confidence": signal.confidence,
                "rsi": latest.get("RSI", 0),
                "macd": latest.get("MACD", 0),
                "macd_signal": latest.get("MACD_Signal", 0),
                "adx": latest.get("ADX", 0),
                "volume_ratio": latest.get("Volume_Ratio", 1),
            })
        except Exception:
            continue
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results


@st.cache_data(ttl=3600, show_spinner=False)
def get_ohlol_stats(ticker: str, lookback_days: int = 252) -> dict:
    df = fetch_stock_data(ticker, period_years=2)
    if df is None or df.empty or len(df) < 10:
        return {"oh_days": 0, "oh_bearish_rate": 0.0, "ol_days": 0, "ol_bullish_rate": 0.0}
    df = df.tail(lookback_days).copy()
    tolerance = 0.001
    oh_mask = (df["High"] - df["Open"]).abs() / df["Open"] <= tolerance
    ol_mask = (df["Low"] - df["Open"]).abs() / df["Open"] <= tolerance
    oh_days = oh_mask.sum()
    ol_days = ol_mask.sum()
    oh_bearish = ((df["Close"] < df["Open"]) & oh_mask).sum()
    ol_bullish = ((df["Close"] > df["Open"]) & ol_mask).sum()
    return {
        "oh_days": int(oh_days),
        "oh_bearish_rate": round(oh_bearish / oh_days * 100, 1) if oh_days > 0 else 0.0,
        "ol_days": int(ol_days),
        "ol_bullish_rate": round(ol_bullish / ol_days * 100, 1) if ol_days > 0 else 0.0,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def get_gap_fill_stats(ticker: str, lookback_days: int = 252) -> dict:
    df = fetch_stock_data(ticker, period_years=2)
    if df is None or df.empty or len(df) < 10:
        return {"gap_up_count": 0, "gap_up_fill_rate": 0.0, "gap_down_count": 0, "gap_down_fill_rate": 0.0}
    df = df.tail(lookback_days).copy()
    prev_close = df["Close"].shift(1)
    gap_up_mask = df["Open"] > prev_close * 1.005
    gap_down_mask = df["Open"] < prev_close * 0.995
    gap_up_filled = gap_up_mask & (df["Low"] <= prev_close)
    gap_down_filled = gap_down_mask & (df["High"] >= prev_close)
    gu = gap_up_mask.sum()
    gd = gap_down_mask.sum()
    return {
        "gap_up_count": int(gu),
        "gap_up_fill_rate": round(gap_up_filled.sum() / gu * 100, 1) if gu > 0 else 0.0,
        "gap_down_count": int(gd),
        "gap_down_fill_rate": round(gap_down_filled.sum() / gd * 100, 1) if gd > 0 else 0.0,
    }
