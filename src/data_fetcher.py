import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


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


def fetch_stock_data(ticker: str, period_years: int = 5) -> pd.DataFrame:
    """Fetch historical OHLCV data for a given stock ticker."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_years * 365)

    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))

    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'. Check the ticker symbol.")

    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.dropna(inplace=True)

    return df


def fetch_market_context(period_years: int = 5) -> pd.DataFrame:
    """Fetch Nifty 50 and India VIX data for market context features."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_years * 365)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    result = pd.DataFrame()

    # Fetch Nifty 50
    try:
        nifty = yf.Ticker("^NSEI").history(start=start_str, end=end_str)
        if not nifty.empty:
            nifty.index = pd.to_datetime(nifty.index).tz_localize(None)
            result["Nifty_Close"] = nifty["Close"]
            result["Nifty_Returns"] = nifty["Close"].pct_change()
    except Exception:
        pass

    # Fetch India VIX
    try:
        vix = yf.Ticker("^INDIAVIX").history(start=start_str, end=end_str)
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


def fetch_multiple_stocks(tickers: list[str], period_years: int = 2) -> pd.DataFrame:
    """Fetch Close prices for multiple tickers, aligned on common dates."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=period_years * 365)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    prices = pd.DataFrame()
    for ticker in tickers:
        try:
            df = yf.Ticker(ticker).history(start=start_str, end=end_str)
            if not df.empty:
                df.index = pd.to_datetime(df.index).tz_localize(None)
                prices[ticker] = df["Close"]
        except Exception:
            continue

    if not prices.empty:
        prices.ffill(limit=3, inplace=True)
        prices.dropna(inplace=True)

    return prices


def get_stock_info(ticker: str) -> dict:
    """Get basic stock information."""
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "name": info.get("longName", ticker),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "market_cap": info.get("marketCap", 0),
        "current_price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
        "currency": info.get("currency", "INR"),
    }


def get_stock_news(ticker: str) -> list[dict]:
    """Get recent news for a stock ticker."""
    stock = yf.Ticker(ticker)
    news = stock.news or []
    results = []
    for item in news[:10]:
        results.append({
            "title": item.get("title", ""),
            "publisher": item.get("publisher", ""),
            "link": item.get("link", ""),
            "published": item.get("providerPublishTime", ""),
        })
    return results


def fetch_intraday_data(ticker: str, interval: str = "5m", period: str = "5d") -> pd.DataFrame:
    """Fetch intraday OHLCV data for scalping analysis.

    Args:
        ticker: Stock ticker symbol
        interval: Candle interval - "1m", "2m", "5m", "15m", "30m", "1h"
        period: How far back - "1d", "5d", "1mo" (yfinance limits: 1m=7d, 5m=60d, 15m=60d)
    """
    stock = yf.Ticker(ticker)
    df = stock.history(interval=interval, period=period)

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


def get_stock_fundamentals(ticker: str) -> dict:
    """Fetch comprehensive fundamental data from yfinance."""
    stock = yf.Ticker(ticker)
    info = stock.info
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
    for t in tickers:
        try:
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
