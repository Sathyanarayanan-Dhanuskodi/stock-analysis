import requests
import streamlit as st


_NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com",
    "Connection": "keep-alive",
}


def _get_nse_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(_NSE_HEADERS)
    session.get("https://www.nseindia.com", timeout=10)
    return session


@st.cache_data(ttl=60, show_spinner=False)
def fetch_nse_option_chain(symbol: str = "NIFTY") -> dict:
    try:
        session = _get_nse_session()
        if symbol in ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"):
            url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
        else:
            url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


def calculate_pcr(option_chain: dict) -> float:
    try:
        records = option_chain.get("records", {}).get("data", [])
        total_ce_oi = sum(r["CE"]["openInterest"] for r in records if "CE" in r)
        total_pe_oi = sum(r["PE"]["openInterest"] for r in records if "PE" in r)
        if total_ce_oi == 0:
            return 0.0
        return round(total_pe_oi / total_ce_oi, 2)
    except Exception:
        return 0.0


def get_oi_sentiment(pcr: float) -> str:
    if pcr >= 1.2:
        return "Bullish"
    elif pcr <= 0.8:
        return "Bearish"
    return "Neutral"


def get_oi_status(price_change_pct: float, oi_change_pct: float) -> str:
    price_up = price_change_pct > 0
    oi_up = oi_change_pct > 0
    if price_up and oi_up:
        return "Long Buildup"
    elif not price_up and oi_up:
        return "Short Buildup"
    elif price_up and not oi_up:
        return "Short Covering"
    else:
        return "Long Unwinding"


def get_oi_status_color(status: str) -> str:
    return {
        "Long Buildup": "#34d399",
        "Short Covering": "#86efac",
        "Short Buildup": "#f87171",
        "Long Unwinding": "#fca5a5",
    }.get(status, "#888")


def get_total_oi(option_chain: dict) -> dict:
    try:
        records = option_chain.get("records", {}).get("data", [])
        total_ce_oi = sum(r["CE"]["openInterest"] for r in records if "CE" in r)
        total_pe_oi = sum(r["PE"]["openInterest"] for r in records if "PE" in r)
        total_ce_chng = sum(r["CE"]["changeinOpenInterest"] for r in records if "CE" in r)
        total_pe_chng = sum(r["PE"]["changeinOpenInterest"] for r in records if "PE" in r)
        return {
            "ce_oi": total_ce_oi,
            "pe_oi": total_pe_oi,
            "ce_chng": total_ce_chng,
            "pe_chng": total_pe_chng,
            "total_oi": total_ce_oi + total_pe_oi,
            "oi_chng": total_ce_chng + total_pe_chng,
        }
    except Exception:
        return {"ce_oi": 0, "pe_oi": 0, "ce_chng": 0, "pe_chng": 0, "total_oi": 0, "oi_chng": 0}


def get_top_strikes(option_chain: dict, top_n: int = 5) -> list[dict]:
    try:
        records = option_chain.get("records", {}).get("data", [])
        strikes = []
        for r in records:
            ce_oi = r["CE"]["openInterest"] if "CE" in r else 0
            pe_oi = r["PE"]["openInterest"] if "PE" in r else 0
            strikes.append({
                "strike": r.get("strikePrice", 0),
                "ce_oi": ce_oi,
                "pe_oi": pe_oi,
            })
        ce_sorted = sorted(strikes, key=lambda x: x["ce_oi"], reverse=True)[:top_n]
        pe_sorted = sorted(strikes, key=lambda x: x["pe_oi"], reverse=True)[:top_n]
        return {"top_ce": ce_sorted, "top_pe": pe_sorted}
    except Exception:
        return {"top_ce": [], "top_pe": []}
