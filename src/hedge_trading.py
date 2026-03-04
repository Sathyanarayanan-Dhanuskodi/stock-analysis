import pandas as pd
import numpy as np
from scipy import stats
from dataclasses import dataclass


@dataclass
class PairTradingSignal:
    stock_a: str
    stock_b: str
    signal: str  # "LONG_A_SHORT_B" | "SHORT_A_LONG_B" | "NEUTRAL"
    z_score: float
    half_life_days: float
    correlation: float


def calculate_correlation_matrix(
    prices_df: pd.DataFrame,
    method: str = "pearson",
    rolling_window: int | None = None,
) -> pd.DataFrame:
    """Compute correlation matrix for a set of stocks."""
    returns = prices_df.pct_change().dropna()

    if rolling_window and len(returns) > rolling_window:
        return returns.tail(rolling_window).corr(method=method)
    return returns.corr(method=method)


def find_hedge_candidates(
    target_ticker: str,
    prices_df: pd.DataFrame,
    min_negative_corr: float = -0.3,
    min_positive_corr: float = 0.7,
) -> dict:
    """Find hedging and pairing candidates for a target stock."""
    corr_matrix = calculate_correlation_matrix(prices_df)

    if target_ticker not in corr_matrix.columns:
        return {"inverse_hedges": [], "pair_candidates": []}

    correlations = corr_matrix[target_ticker].drop(target_ticker, errors="ignore")

    inverse_hedges = []
    pair_candidates = []

    for ticker, corr in correlations.items():
        if corr <= min_negative_corr:
            hr = calculate_hedge_ratio(prices_df[target_ticker], prices_df[ticker])
            inverse_hedges.append({"ticker": ticker, "correlation": round(corr, 3), "hedge_ratio": round(hr, 3)})
        elif corr >= min_positive_corr:
            coint = test_cointegration(prices_df[target_ticker], prices_df[ticker])
            pair_candidates.append({
                "ticker": ticker, "correlation": round(corr, 3),
                "cointegrated": coint["is_cointegrated"], "p_value": round(coint["p_value"], 4),
            })

    inverse_hedges.sort(key=lambda x: x["correlation"])
    pair_candidates.sort(key=lambda x: x["correlation"], reverse=True)

    return {"inverse_hedges": inverse_hedges, "pair_candidates": pair_candidates}


def calculate_beta(
    stock_returns: pd.Series,
    nifty_returns: pd.Series,
    rolling_window: int | None = 252,
) -> dict:
    """Calculate beta of stock vs Nifty 50."""
    aligned = pd.DataFrame({"stock": stock_returns, "nifty": nifty_returns}).dropna()

    if len(aligned) < 30:
        return {"beta": 1.0, "alpha": 0.0, "r_squared": 0.0,
                "beta_interpretation": "Insufficient data", "rolling_beta": None}

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        aligned["nifty"], aligned["stock"]
    )

    beta = slope
    alpha = intercept * 252  # Annualize daily alpha
    r_squared = r_value ** 2

    if beta < 0:
        interpretation = "Inverse to market (rare hedge)"
    elif beta < 0.8:
        interpretation = "Defensive (low volatility)"
    elif beta <= 1.2:
        interpretation = "Market-aligned"
    else:
        interpretation = "Aggressive (high volatility)"

    rolling_beta = None
    if rolling_window and len(aligned) > rolling_window:
        rolling_beta = aligned["stock"].rolling(rolling_window).cov(aligned["nifty"]) / \
                       aligned["nifty"].rolling(rolling_window).var()

    return {
        "beta": round(beta, 3),
        "alpha": round(alpha, 4),
        "r_squared": round(r_squared, 3),
        "beta_interpretation": interpretation,
        "rolling_beta": rolling_beta,
    }


def calculate_hedge_ratio(stock_a: pd.Series, stock_b: pd.Series) -> float:
    """Calculate hedge ratio using OLS regression."""
    aligned = pd.DataFrame({"a": stock_a, "b": stock_b}).dropna()
    if len(aligned) < 30:
        return 1.0
    slope, _, _, _, _ = stats.linregress(aligned["b"], aligned["a"])
    return slope


def test_cointegration(series_a: pd.Series, series_b: pd.Series) -> dict:
    """Engle-Granger cointegration test for pairs trading."""
    try:
        from statsmodels.tsa.stattools import coint
        aligned = pd.DataFrame({"a": series_a, "b": series_b}).dropna()

        if len(aligned) < 60:
            return {"is_cointegrated": False, "p_value": 1.0, "test_statistic": 0, "half_life_days": float("inf")}

        score, p_value, _ = coint(aligned["a"], aligned["b"])

        # Estimate half-life of mean reversion
        hedge_ratio = calculate_hedge_ratio(aligned["a"], aligned["b"])
        spread = aligned["a"] - hedge_ratio * aligned["b"]
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()

        n = min(len(spread_lag), len(spread_diff))
        if n > 10:
            slope, _, _, _, _ = stats.linregress(spread_lag.iloc[-n:], spread_diff.iloc[-n:])
            half_life = -np.log(2) / slope if slope < 0 else float("inf")
        else:
            half_life = float("inf")

        return {
            "is_cointegrated": p_value < 0.05,
            "p_value": p_value,
            "test_statistic": score,
            "half_life_days": round(half_life, 1) if half_life != float("inf") else float("inf"),
        }
    except ImportError:
        return {"is_cointegrated": False, "p_value": 1.0, "test_statistic": 0, "half_life_days": float("inf")}


def generate_pair_trading_signal(
    stock_a: str,
    stock_b: str,
    prices_df: pd.DataFrame,
    lookback: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> PairTradingSignal:
    """Z-score based pairs trading signal."""
    a_prices = prices_df[stock_a].tail(lookback)
    b_prices = prices_df[stock_b].tail(lookback)

    hedge_ratio = calculate_hedge_ratio(a_prices, b_prices)
    spread = a_prices - hedge_ratio * b_prices
    z_score = (spread.iloc[-1] - spread.mean()) / spread.std() if spread.std() > 0 else 0

    corr = a_prices.corr(b_prices)

    coint_result = test_cointegration(a_prices, b_prices)
    half_life = coint_result["half_life_days"]

    if z_score > entry_z:
        signal = "SHORT_A_LONG_B"
    elif z_score < -entry_z:
        signal = "LONG_A_SHORT_B"
    elif abs(z_score) < exit_z:
        signal = "NEUTRAL"
    else:
        signal = "NEUTRAL"

    return PairTradingSignal(
        stock_a=stock_a, stock_b=stock_b, signal=signal,
        z_score=round(z_score, 3), half_life_days=half_life,
        correlation=round(corr, 3),
    )


def calculate_portfolio_var(
    prices_df: pd.DataFrame,
    weights: dict[str, float],
    confidence_level: float = 0.95,
    horizon_days: int = 1,
    method: str = "historical",
) -> dict:
    """Value at Risk (VaR) calculation."""
    returns = prices_df.pct_change().dropna()
    tickers = [t for t in weights if t in returns.columns]
    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()  # Normalize

    portfolio_returns = (returns[tickers] * w).sum(axis=1)

    if method == "parametric":
        mu = portfolio_returns.mean()
        sigma = portfolio_returns.std()
        z = stats.norm.ppf(1 - confidence_level)
        var_1d = -(mu + z * sigma)
    else:  # historical
        var_1d = -np.percentile(portfolio_returns, (1 - confidence_level) * 100)

    var_nd = var_1d * np.sqrt(horizon_days)

    # Conditional VaR (Expected Shortfall)
    threshold = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
    cvar = -portfolio_returns[portfolio_returns <= threshold].mean() if len(portfolio_returns[portfolio_returns <= threshold]) > 0 else var_1d

    return {
        "var_1d": round(var_1d * 100, 3),
        "var_nd": round(var_nd * 100, 3),
        "cvar": round(cvar * 100, 3),
        "method": method,
        "confidence_level": confidence_level,
        "worst_day_return": round(portfolio_returns.min() * 100, 3),
        "horizon_days": horizon_days,
    }


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.065,
) -> dict:
    """Sharpe, Sortino, Calmar ratios and related metrics."""
    if len(returns) < 30:
        return {"sharpe_ratio": 0, "sortino_ratio": 0, "calmar_ratio": 0,
                "annualized_return": 0, "annualized_volatility": 0,
                "max_drawdown": 0, "max_drawdown_duration_days": 0}

    annual_return = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    daily_rf = risk_free_rate / 252

    # Sharpe
    excess = returns - daily_rf
    sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0

    # Sortino (downside deviation only)
    downside = excess[excess < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else annual_vol
    sortino = (annual_return - risk_free_rate) / downside_std if downside_std > 0 else 0

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = abs(drawdown.min())

    # Max drawdown duration
    underwater = drawdown < 0
    if underwater.any():
        groups = (~underwater).cumsum()
        dd_durations = underwater.groupby(groups).sum()
        max_dd_duration = int(dd_durations.max()) if len(dd_durations) > 0 else 0
    else:
        max_dd_duration = 0

    # Calmar
    calmar = (annual_return - risk_free_rate) / max_dd if max_dd > 0 else 0

    return {
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "calmar_ratio": round(calmar, 3),
        "annualized_return": round(annual_return * 100, 2),
        "annualized_volatility": round(annual_vol * 100, 2),
        "max_drawdown": round(max_dd * 100, 2),
        "max_drawdown_duration_days": max_dd_duration,
    }
