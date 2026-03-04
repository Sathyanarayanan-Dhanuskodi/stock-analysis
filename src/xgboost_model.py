import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
}


def prepare_xgb_dataset(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "Close",
    forecast_horizon: int = 1,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare flat feature matrix for XGBoost with shifted target."""
    available_cols = [c for c in feature_cols if c in df.columns]
    X = df[available_cols].copy()
    y = df[target_col].shift(-forecast_horizon)

    # Drop rows where target is NaN (last forecast_horizon rows)
    mask = y.notna()
    return X[mask], y[mask]


def train_xgb_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int = 5,
    forecast_horizon: int = 1,
) -> tuple[xgb.XGBRegressor, dict]:
    """Train XGBoost with TimeSeriesSplit cross-validation."""
    X, y = prepare_xgb_dataset(df, feature_cols, forecast_horizon=forecast_horizon)

    if len(X) < 100:
        raise ValueError("Not enough data for XGBoost training.")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []

    # Cross-validation
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        preds = model.predict(X_val)
        rmse = np.sqrt(np.mean((preds - y_val.values) ** 2))
        cv_scores.append(rmse)

    # Train final model on all data with early stopping on last split
    split_idx = int(len(X) * 0.85)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    final_model = xgb.XGBRegressor(**XGB_PARAMS)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Feature importance
    importance = dict(zip(X.columns, final_model.feature_importances_))
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    metrics = {
        "cv_rmse_mean": np.mean(cv_scores),
        "cv_rmse_std": np.std(cv_scores),
        "feature_importance": importance,
    }

    return final_model, metrics


def xgb_recursive_forecast(
    model: xgb.XGBRegressor,
    df: pd.DataFrame,
    feature_cols: list[str],
    forecast_days: int = 30,
) -> list[float]:
    """Multi-step forecast using recursive strategy."""
    available_cols = [c for c in feature_cols if c in df.columns]
    working_df = df.copy()
    predictions = []

    for _ in range(forecast_days):
        features = working_df[available_cols].iloc[-1:].values
        pred = model.predict(features)[0]
        predictions.append(pred)

        # Create next row by copying last row and updating Close + derived features
        new_row = working_df.iloc[-1:].copy()
        new_row.index = new_row.index + pd.Timedelta(days=1)
        new_row["Close"] = pred

        # Update lag features
        for lag_col in available_cols:
            if lag_col.startswith("Close_Lag_"):
                lag_n = int(lag_col.split("_")[-1])
                if lag_n == 1:
                    new_row[lag_col] = working_df["Close"].iloc[-1]
                elif f"Close_Lag_{lag_n - 1}" in working_df.columns:
                    new_row[lag_col] = working_df[f"Close_Lag_{lag_n - 1}"].iloc[-1]
            elif lag_col.startswith("Returns_Lag_"):
                lag_n = int(lag_col.split("_")[-1])
                if lag_n == 1 and "Returns" in working_df.columns:
                    new_row[lag_col] = working_df["Returns"].iloc[-1]

        # Update returns
        if "Returns" in new_row.columns:
            prev_close = working_df["Close"].iloc[-1]
            new_row["Returns"] = (pred - prev_close) / prev_close if prev_close != 0 else 0

        working_df = pd.concat([working_df, new_row])

    return predictions


def get_feature_importance(model: xgb.XGBRegressor, feature_cols: list[str]) -> pd.DataFrame:
    """Return sorted DataFrame of feature importance scores."""
    importance = pd.DataFrame({
        "Feature": feature_cols[:len(model.feature_importances_)],
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    return importance


def save_xgb_model(model: xgb.XGBRegressor, ticker: str):
    """Save XGBoost model using joblib."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    safe_ticker = ticker.replace(".", "_")
    joblib.dump(model, os.path.join(MODELS_DIR, f"{safe_ticker}_xgb_model.joblib"))


def load_xgb_model(ticker: str) -> xgb.XGBRegressor | None:
    """Load previously saved XGBoost model."""
    safe_ticker = ticker.replace(".", "_")
    path = os.path.join(MODELS_DIR, f"{safe_ticker}_xgb_model.joblib")
    if not os.path.exists(path):
        return None
    return joblib.load(path)
