import pandas as pd
import streamlit as st
from src.data_fetcher import fetch_stock_data


@st.cache_data(ttl=3600, show_spinner=False)
def get_monthly_returns(ticker: str, years: int = 8) -> pd.DataFrame:
    df = fetch_stock_data(ticker, period_years=years)
    if df is None or df.empty:
        return pd.DataFrame()
    monthly = df["Close"].resample("ME").last().pct_change().dropna() * 100
    result = pd.DataFrame({
        "Year": monthly.index.year,
        "Month": monthly.index.month,
        "Return": monthly.values,
    })
    pivot = result.pivot(index="Year", columns="Month", values="Return")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot.columns = [month_names[m - 1] for m in pivot.columns]
    return pivot.sort_index(ascending=False)


@st.cache_data(ttl=3600, show_spinner=False)
def get_dow_returns(ticker: str, years: int = 3) -> pd.DataFrame:
    df = fetch_stock_data(ticker, period_years=years)
    if df is None or df.empty:
        return pd.DataFrame()
    daily_returns = df["Close"].pct_change().dropna() * 100
    dow_map = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday"}
    result = pd.DataFrame({
        "DayOfWeek": daily_returns.index.dayofweek,
        "Return": daily_returns.values,
    })
    result = result[result["DayOfWeek"] <= 4]
    summary = result.groupby("DayOfWeek")["Return"].agg(["mean", "count", "std"]).reset_index()
    summary["Day"] = summary["DayOfWeek"].map(dow_map)
    summary.columns = ["DayOfWeek", "AvgReturn", "Count", "StdDev", "Day"]
    return summary[["Day", "AvgReturn", "Count", "StdDev"]]


@st.cache_data(ttl=3600, show_spinner=False)
def get_monthly_stats(ticker: str, years: int = 8) -> pd.DataFrame:
    df = fetch_stock_data(ticker, period_years=years)
    if df is None or df.empty:
        return pd.DataFrame()
    monthly = df["Close"].resample("ME").last().pct_change().dropna() * 100
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    result = pd.DataFrame({
        "Month": monthly.index.month,
        "Return": monthly.values,
    })
    summary = result.groupby("Month")["Return"].agg(["mean", "count", "std"]).reset_index()
    summary["MonthName"] = summary["Month"].apply(lambda m: month_names[m - 1])
    summary["PositiveRate"] = result.groupby("Month")["Return"].apply(
        lambda x: (x > 0).mean() * 100
    ).values
    summary.columns = ["Month", "AvgReturn", "Count", "StdDev", "MonthName", "PositiveRate"]
    return summary[["MonthName", "AvgReturn", "PositiveRate", "Count", "StdDev"]]
