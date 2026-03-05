import numpy as np
import pandas as pd

from src.data_fetcher import fetch_stock_data, fetch_market_context
from src.feature_engineering import (
    add_technical_indicators, add_market_context, add_lagged_features,
    get_feature_columns, get_xgboost_feature_columns,
)
from src.model import (
    LOOKBACK, MAX_FORECAST, train_model, save_model, load_saved_model,
    prepare_sequences_single,
)
from src.xgboost_model import (
    train_xgb_model, xgb_recursive_forecast, save_xgb_model, load_xgb_model,
)
from src.ensemble import (
    compute_horizon_weights, blend_predictions, compute_ensemble_confidence,
    auto_tune_weights,
)


def _prepare_data(ticker: str, use_market_context: bool = True):
    """Fetch and prepare data with all features."""
    df = fetch_stock_data(ticker)
    df = add_technical_indicators(df)

    if use_market_context:
        try:
            context_df = fetch_market_context()
            df = add_market_context(df, context_df)
        except Exception:
            pass  # Continue without market context

    return df


def _get_lstm_features(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Get LSTM feature columns and data (no lags needed)."""
    feature_cols = get_feature_columns()
    available = [c for c in feature_cols if c in df.columns]
    cols = ["Close"] + [c for c in available if c != "Close"]
    return cols, df[cols].values


def _inverse_close(scaled_vals, scaler):
    """Inverse transform Close column values."""
    close_scale = scaler.scale_[0]
    return (scaled_vals - scaler.min_[0]) / close_scale


def train_and_predict(
    ticker: str,
    forecast_days: int = 7,
    epochs: int = 80,
    progress_callback=None,
    use_market_context: bool = True,
    xgb_base_weight: float = 0.55,
) -> dict:
    """Full ensemble pipeline: fetch → features → train LSTM + XGBoost → blend → predict."""

    # Phase 1: Data preparation
    df = _prepare_data(ticker, use_market_context)
    df_with_lags = add_lagged_features(df)

    # Phase 2: Train LSTM
    lstm_cols, lstm_data = _get_lstm_features(df)

    def lstm_progress(epoch, total):
        if progress_callback:
            progress_callback(epoch, total, "LSTM")

    lstm_model, scaler, lstm_history = train_model(
        lstm_data, epochs=epochs, progress_callback=lstm_progress,
    )
    save_model(lstm_model, scaler, ticker)

    # Phase 3: Train XGBoost
    if progress_callback:
        progress_callback(0, 1, "XGBoost")

    xgb_feature_cols = get_xgboost_feature_columns()
    available_xgb_cols = [c for c in xgb_feature_cols if c in df_with_lags.columns]

    xgb_model, xgb_metrics = train_xgb_model(df_with_lags, available_xgb_cols)
    save_xgb_model(xgb_model, ticker)

    if progress_callback:
        progress_callback(1, 1, "XGBoost")

    # Phase 4: Backtest evaluation (LSTM)
    scaled_data = scaler.transform(lstm_data)
    X_eval, y_actual_scaled = prepare_sequences_single(scaled_data)
    lstm_backtest_scaled = lstm_model.predict(X_eval, verbose=0)[:, 0]
    lstm_backtest = _inverse_close(lstm_backtest_scaled, scaler)
    actual_prices = _inverse_close(y_actual_scaled, scaler)

    # Backtest evaluation (XGBoost) — single-step on test set
    xgb_backtest_vals = []
    test_start = len(df_with_lags) - len(actual_prices)
    for i in range(test_start, len(df_with_lags)):
        row = df_with_lags[available_xgb_cols].iloc[i:i+1]
        if len(row) > 0:
            pred = xgb_model.predict(row.values)[0]
            xgb_backtest_vals.append(pred)
    xgb_backtest = np.array(xgb_backtest_vals[:len(actual_prices)])

    # Ensemble backtest — auto-tune optimal weight
    n_eval = min(len(lstm_backtest), len(xgb_backtest))

    tune_result = auto_tune_weights(
        actual_prices[:n_eval], lstm_backtest[:n_eval], xgb_backtest[:n_eval],
    )
    optimal_weight = tune_result["optimal_weight"]

    # Use the better of: user-specified weight or auto-tuned weight
    # Auto-tune wins if it found a meaningfully better RMSE
    user_weights = compute_horizon_weights(1, xgb_base_weight=xgb_base_weight)
    user_xgb_w, user_lstm_w = user_weights[0]
    user_ensemble = user_xgb_w * xgb_backtest[:n_eval] + user_lstm_w * lstm_backtest[:n_eval]
    user_rmse = float(np.sqrt(np.mean((actual_prices[:n_eval] - user_ensemble) ** 2)))

    if tune_result["best_rmse"] < user_rmse * 0.98:  # Auto-tune at least 2% better
        best_xgb_weight = optimal_weight
    else:
        best_xgb_weight = xgb_base_weight

    weights_1d = compute_horizon_weights(1, xgb_base_weight=best_xgb_weight)
    xgb_w, lstm_w = weights_1d[0]
    ensemble_backtest = xgb_w * xgb_backtest[:n_eval] + lstm_w * lstm_backtest[:n_eval]

    eval_df = pd.DataFrame({
        "Date": df.index[LOOKBACK:LOOKBACK + n_eval],
        "Actual": actual_prices[:n_eval],
        "Predicted": ensemble_backtest,
        "LSTM": lstm_backtest[:n_eval],
        "XGBoost": xgb_backtest[:n_eval],
    }).set_index("Date")

    # Phase 5: Future predictions
    # LSTM direct multi-step
    last_sequence = scaled_data[-LOOKBACK:]
    lstm_all_preds = lstm_model.predict(
        last_sequence.reshape(1, LOOKBACK, -1), verbose=0
    )[0]
    lstm_future = [_inverse_close(p, scaler) for p in lstm_all_preds[:forecast_days]]

    # XGBoost recursive
    xgb_future = xgb_recursive_forecast(
        xgb_model, df_with_lags, available_xgb_cols, forecast_days,
    )

    # Ensemble blend with auto-tuned weight
    weights = compute_horizon_weights(forecast_days, xgb_base_weight=best_xgb_weight)
    ensemble_future = blend_predictions(lstm_future, xgb_future, weights)

    # Future dates
    last_date = df.index[-1]
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)

    # Confidence intervals
    recent_errors = np.abs(eval_df["Actual"].values[-60:] - eval_df["Predicted"].values[-60:])
    error_std = np.std(recent_errors) if len(recent_errors) > 0 else 0

    confidence = compute_ensemble_confidence(
        lstm_future, xgb_future, ensemble_future,
        error_std, forecast_days, future_dates,
    )

    # Backtest metrics
    from src.ensemble import evaluate_models
    backtest_metrics = evaluate_models(
        actual_prices[:n_eval], lstm_backtest[:n_eval],
        xgb_backtest[:n_eval], ensemble_backtest,
    )

    return {
        "historical_df": df,
        "predictions": ensemble_future,
        "lstm_predictions": lstm_future,
        "xgb_predictions": xgb_future,
        "prediction_dates": future_dates,
        "confidence": confidence,
        "train_metrics": lstm_history,
        "actual_vs_predicted": eval_df,
        "backtest_metrics": backtest_metrics,
        "feature_importance": xgb_metrics.get("feature_importance", {}),
        "ensemble_weights": weights,
        "last_price": df["Close"].iloc[-1],
        "auto_tune": tune_result,
        "used_xgb_weight": best_xgb_weight,
    }


def predict_with_saved_model(
    ticker: str,
    forecast_days: int = 7,
    use_market_context: bool = True,
    xgb_base_weight: float = 0.55,
) -> dict | None:
    """Use saved LSTM + XGBoost models for ensemble prediction without retraining."""
    lstm_result = load_saved_model(ticker)
    if lstm_result is None:
        return None

    lstm_model, scaler = lstm_result
    xgb_model = load_xgb_model(ticker)

    df = _prepare_data(ticker, use_market_context)

    # LSTM prediction
    lstm_cols, lstm_data = _get_lstm_features(df)
    scaled_data = scaler.transform(lstm_data)
    last_sequence = scaled_data[-LOOKBACK:]
    lstm_all_preds = lstm_model.predict(
        last_sequence.reshape(1, LOOKBACK, -1), verbose=0
    )[0]
    lstm_future = [_inverse_close(p, scaler) for p in lstm_all_preds[:forecast_days]]

    # XGBoost prediction (if model exists)
    if xgb_model is not None:
        df_with_lags = add_lagged_features(df)
        xgb_feature_cols = get_xgboost_feature_columns()
        available_xgb_cols = [c for c in xgb_feature_cols if c in df_with_lags.columns]
        xgb_future = xgb_recursive_forecast(
            xgb_model, df_with_lags, available_xgb_cols, forecast_days,
        )
        weights = compute_horizon_weights(forecast_days, xgb_base_weight=xgb_base_weight)
        ensemble_future = blend_predictions(lstm_future, xgb_future, weights)
    else:
        xgb_future = lstm_future  # Fallback to LSTM only
        ensemble_future = lstm_future

    future_dates = pd.bdate_range(
        start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)

    return {
        "predictions": ensemble_future,
        "lstm_predictions": lstm_future,
        "xgb_predictions": xgb_future,
        "prediction_dates": future_dates,
        "last_price": df["Close"].iloc[-1],
        "historical_df": df,
    }
