import numpy as np
import pandas as pd


def compute_horizon_weights(
    forecast_days: int,
    xgb_base_weight: float = 0.55,
    decay_rate: float = 0.025,
    min_xgb_weight: float = 0.20,
) -> list[tuple[float, float]]:
    """Compute per-day (xgb_weight, lstm_weight) pairs.

    XGBoost weight decays with horizon because recursive forecasting
    accumulates error. LSTM's direct output is more reliable at longer horizons.
    """
    weights = []
    for day in range(forecast_days):
        xgb_w = max(min_xgb_weight, xgb_base_weight - decay_rate * day)
        lstm_w = 1.0 - xgb_w
        weights.append((xgb_w, lstm_w))
    return weights


def blend_predictions(
    lstm_preds: list[float],
    xgb_preds: list[float],
    weights: list[tuple[float, float]] | None = None,
) -> list[float]:
    """Produce weighted ensemble predictions."""
    n = min(len(lstm_preds), len(xgb_preds))
    if weights is None:
        weights = compute_horizon_weights(n)

    ensemble = []
    for i in range(n):
        xgb_w, lstm_w = weights[i]
        pred = xgb_w * xgb_preds[i] + lstm_w * lstm_preds[i]
        ensemble.append(pred)
    return ensemble


def compute_ensemble_confidence(
    lstm_preds: list[float],
    xgb_preds: list[float],
    ensemble_preds: list[float],
    historical_error_std: float,
    forecast_days: int,
    prediction_dates: list,
) -> list[dict]:
    """Compute confidence bands using historical error + model disagreement."""
    confidence = []
    for i in range(forecast_days):
        price = ensemble_preds[i]
        # Base uncertainty from historical backtest errors
        base_spread = historical_error_std * (1 + 0.15 * i)
        # Model disagreement widens the band
        disagreement = abs(lstm_preds[i] - xgb_preds[i]) if i < len(lstm_preds) and i < len(xgb_preds) else 0
        total_spread = base_spread + 0.5 * disagreement

        confidence.append({
            "date": prediction_dates[i] if i < len(prediction_dates) else None,
            "predicted": price,
            "lower": price - 2 * total_spread,
            "upper": price + 2 * total_spread,
            "model_disagreement": disagreement,
            "disagreement_pct": (disagreement / price * 100) if price > 0 else 0,
        })
    return confidence


def auto_tune_weights(
    actual: np.ndarray,
    lstm_preds: np.ndarray,
    xgb_preds: np.ndarray,
    weight_range: tuple[float, float] = (0.20, 0.90),
    step: float = 0.05,
) -> dict:
    """Find the optimal xgb_base_weight by grid search on backtest data.

    Tests every weight from weight_range[0] to weight_range[1] in increments of step.
    Returns the weight that minimizes RMSE on the backtest.
    """
    n = min(len(lstm_preds), len(xgb_preds), len(actual))
    if n < 30:
        return {"optimal_weight": 0.55, "best_rmse": float("inf"), "all_results": []}

    actual = actual[:n]
    lstm = lstm_preds[:n]
    xgb = xgb_preds[:n]

    best_weight = 0.55
    best_rmse = float("inf")
    all_results = []

    w = weight_range[0]
    while w <= weight_range[1] + 1e-9:
        ensemble = w * xgb + (1 - w) * lstm
        rmse = float(np.sqrt(np.mean((actual - ensemble) ** 2)))

        # Directional accuracy
        actual_dir = np.diff(actual) > 0
        pred_dir = np.diff(ensemble) > 0
        m = min(len(actual_dir), len(pred_dir))
        dir_acc = float(np.mean(actual_dir[:m] == pred_dir[:m]) * 100) if m > 0 else 0

        all_results.append({
            "xgb_weight": round(w, 2),
            "rmse": round(rmse, 4),
            "directional_accuracy": round(dir_acc, 1),
        })

        if rmse < best_rmse:
            best_rmse = rmse
            best_weight = round(w, 2)

        w += step

    return {
        "optimal_weight": best_weight,
        "best_rmse": round(best_rmse, 4),
        "all_results": all_results,
    }


def evaluate_models(
    actual: np.ndarray,
    lstm_preds: np.ndarray,
    xgb_preds: np.ndarray,
    ensemble_preds: np.ndarray,
) -> dict:
    """Compare model performance on backtest data."""
    def _metrics(actual, predicted):
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mae = np.mean(np.abs(actual - predicted))
        # Directional accuracy
        actual_dir = np.diff(actual) > 0
        pred_dir = np.diff(predicted) > 0
        n = min(len(actual_dir), len(pred_dir))
        dir_acc = np.mean(actual_dir[:n] == pred_dir[:n]) * 100 if n > 0 else 0
        return {"rmse": rmse, "mae": mae, "directional_accuracy": dir_acc}

    results = {
        "lstm": _metrics(actual, lstm_preds),
        "xgb": _metrics(actual, xgb_preds),
        "ensemble": _metrics(actual, ensemble_preds),
    }

    # Improvement over LSTM alone
    if results["lstm"]["rmse"] > 0:
        results["improvement_pct"] = (
            (results["lstm"]["rmse"] - results["ensemble"]["rmse"])
            / results["lstm"]["rmse"] * 100
        )
    else:
        results["improvement_pct"] = 0

    return results
