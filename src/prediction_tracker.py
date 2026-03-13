import sqlite3
import os
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from contextlib import contextmanager

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "predictions.db")


@contextmanager
def get_db():
    """Context manager for SQLite connections."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS predictions (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker           TEXT NOT NULL,
                prediction_date  TEXT NOT NULL,
                target_date      TEXT NOT NULL,
                predicted_price  REAL NOT NULL,
                lstm_price       REAL,
                xgb_price        REAL,
                confidence_lower REAL,
                confidence_upper REAL,
                last_known_price REAL NOT NULL,
                actual_price     REAL,
                error_pct        REAL,
                abs_error_pct    REAL,
                direction_correct INTEGER,
                within_confidence INTEGER,
                model_version    INTEGER DEFAULT 1,
                validated_at     TEXT,
                UNIQUE(ticker, prediction_date, target_date)
            );

            CREATE TABLE IF NOT EXISTS model_versions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker          TEXT NOT NULL,
                version         INTEGER NOT NULL,
                trained_at      TEXT DEFAULT (datetime('now')),
                reason          TEXT,
                xgb_weight      REAL,
                backtest_rmse   REAL,
                backtest_dir_acc REAL
            );

            CREATE INDEX IF NOT EXISTS idx_pred_ticker_target
                ON predictions(ticker, target_date);
            CREATE INDEX IF NOT EXISTS idx_pred_unvalidated
                ON predictions(actual_price) WHERE actual_price IS NULL;
        """)


def log_predictions(
    ticker: str,
    prediction_date: date,
    target_dates,
    ensemble_prices: list[float],
    lstm_prices: list[float],
    xgb_prices: list[float],
    confidence: list[dict],
    last_known_price: float,
    model_version: int = 1,
) -> int:
    """Log all predictions from a single prediction run.

    Returns the number of rows inserted.
    """
    rows = []
    pred_date_str = prediction_date.isoformat()

    for i, target_dt in enumerate(target_dates):
        target_str = pd.Timestamp(target_dt).strftime("%Y-%m-%d")
        conf = confidence[i] if i < len(confidence) else {}

        rows.append((
            ticker,
            pred_date_str,
            target_str,
            ensemble_prices[i] if i < len(ensemble_prices) else None,
            lstm_prices[i] if i < len(lstm_prices) else None,
            xgb_prices[i] if i < len(xgb_prices) else None,
            conf.get("lower"),
            conf.get("upper"),
            last_known_price,
            model_version,
        ))

    with get_db() as conn:
        conn.executemany("""
            INSERT OR REPLACE INTO predictions
                (ticker, prediction_date, target_date, predicted_price,
                 lstm_price, xgb_price, confidence_lower, confidence_upper,
                 last_known_price, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)

    return len(rows)


def validate_pending_predictions(ticker: str = None) -> dict:
    """Fetch actual prices for past predictions and fill them in.

    Returns dict with validated_count and any errors.
    """
    import yfinance as yf

    today_str = date.today().isoformat()
    result = {"validated_count": 0, "tickers_updated": [], "errors": []}

    with get_db() as conn:
        if ticker:
            pending = conn.execute("""
                SELECT DISTINCT ticker, MIN(target_date) as min_date, MAX(target_date) as max_date
                FROM predictions
                WHERE actual_price IS NULL AND target_date <= ? AND ticker = ?
                GROUP BY ticker
            """, (today_str, ticker)).fetchall()
        else:
            pending = conn.execute("""
                SELECT DISTINCT ticker, MIN(target_date) as min_date, MAX(target_date) as max_date
                FROM predictions
                WHERE actual_price IS NULL AND target_date <= ?
                GROUP BY ticker
            """, (today_str,)).fetchall()

        for row in pending:
            t = row["ticker"]
            min_date = row["min_date"]
            max_date = row["max_date"]

            try:
                # Fetch actual prices from yfinance
                start = (datetime.strptime(min_date, "%Y-%m-%d") - timedelta(days=3)).strftime("%Y-%m-%d")
                end = (datetime.strptime(max_date, "%Y-%m-%d") + timedelta(days=3)).strftime("%Y-%m-%d")
                hist = yf.Ticker(t).history(start=start, end=end)

                if hist.empty:
                    continue

                hist.index = pd.to_datetime(hist.index).tz_localize(None)

                # Get unvalidated rows for this ticker
                rows_to_update = conn.execute("""
                    SELECT id, target_date, predicted_price, last_known_price,
                           confidence_lower, confidence_upper
                    FROM predictions
                    WHERE ticker = ? AND actual_price IS NULL AND target_date <= ?
                """, (t, today_str)).fetchall()

                for pred_row in rows_to_update:
                    target_dt = pd.Timestamp(pred_row["target_date"])

                    # Find closest trading day close price
                    available_dates = hist.index
                    if len(available_dates) == 0:
                        continue

                    # Use the closest date on or before target
                    mask = available_dates <= target_dt
                    if mask.any():
                        closest_date = available_dates[mask][-1]
                    else:
                        closest_date = available_dates[0]

                    actual = float(hist.loc[closest_date, "Close"])
                    predicted = pred_row["predicted_price"]
                    last_known = pred_row["last_known_price"]

                    error_pct = ((predicted - actual) / actual) * 100 if actual > 0 else 0
                    abs_error = abs(error_pct)

                    # Direction: did predicted and actual move same way from last_known?
                    pred_direction = predicted - last_known
                    actual_direction = actual - last_known
                    direction_correct = 1 if (pred_direction * actual_direction > 0) else 0

                    # Within confidence band?
                    lower = pred_row["confidence_lower"]
                    upper = pred_row["confidence_upper"]
                    within = 1 if (lower is not None and upper is not None
                                   and lower <= actual <= upper) else 0

                    conn.execute("""
                        UPDATE predictions
                        SET actual_price = ?, error_pct = ?, abs_error_pct = ?,
                            direction_correct = ?, within_confidence = ?,
                            validated_at = ?
                        WHERE id = ?
                    """, (actual, round(error_pct, 3), round(abs_error, 3),
                          direction_correct, within,
                          datetime.now().isoformat(), pred_row["id"]))

                    result["validated_count"] += 1

                result["tickers_updated"].append(t)

            except Exception as e:
                result["errors"].append(f"{t}: {str(e)}")

    return result


def get_accuracy_metrics(ticker: str = None, last_n_days: int = 30) -> dict:
    """Compute accuracy metrics for validated predictions."""
    with get_db() as conn:
        if ticker:
            rows = conn.execute("""
                SELECT * FROM predictions
                WHERE ticker = ? AND actual_price IS NOT NULL
                ORDER BY target_date DESC LIMIT ?
            """, (ticker, last_n_days * 30)).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM predictions
                WHERE actual_price IS NOT NULL
                ORDER BY target_date DESC LIMIT ?
            """, (last_n_days * 30,)).fetchall()

        if not rows:
            return {
                "total_predictions": 0, "validated_count": 0,
                "direction_accuracy_pct": 0, "avg_error_pct": 0,
                "median_error_pct": 0, "within_confidence_pct": 0,
            }

        total = len(rows)
        direction_correct = sum(1 for r in rows if r["direction_correct"] == 1)
        within_conf = sum(1 for r in rows if r["within_confidence"] == 1)
        abs_errors = [r["abs_error_pct"] for r in rows if r["abs_error_pct"] is not None]

        # Count total predictions (including unvalidated)
        if ticker:
            total_all = conn.execute(
                "SELECT COUNT(*) as c FROM predictions WHERE ticker = ?", (ticker,)
            ).fetchone()["c"]
        else:
            total_all = conn.execute("SELECT COUNT(*) as c FROM predictions").fetchone()["c"]

        return {
            "total_predictions": total_all,
            "validated_count": total,
            "direction_accuracy_pct": round(direction_correct / total * 100, 1) if total > 0 else 0,
            "avg_error_pct": round(sum(abs_errors) / len(abs_errors), 2) if abs_errors else 0,
            "median_error_pct": round(float(np.median(abs_errors)), 2) if abs_errors else 0,
            "within_confidence_pct": round(within_conf / total * 100, 1) if total > 0 else 0,
        }


def get_prediction_history(ticker: str = None, limit: int = 100) -> pd.DataFrame:
    """Return prediction history as a DataFrame."""
    with get_db() as conn:
        if ticker:
            rows = conn.execute("""
                SELECT ticker, prediction_date, target_date, predicted_price,
                       lstm_price, xgb_price, actual_price, error_pct, abs_error_pct,
                       direction_correct, within_confidence, confidence_lower, confidence_upper,
                       last_known_price
                FROM predictions
                WHERE ticker = ?
                ORDER BY target_date DESC
                LIMIT ?
            """, (ticker, limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT ticker, prediction_date, target_date, predicted_price,
                       lstm_price, xgb_price, actual_price, error_pct, abs_error_pct,
                       direction_correct, within_confidence, confidence_lower, confidence_upper,
                       last_known_price
                FROM predictions
                ORDER BY target_date DESC
                LIMIT ?
            """, (limit,)).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame([dict(r) for r in rows])

    # Ensure numeric columns are float (SQLite can return mixed types)
    numeric_cols = ["predicted_price", "lstm_price", "xgb_price", "actual_price",
                    "error_pct", "abs_error_pct", "confidence_lower", "confidence_upper",
                    "last_known_price", "direction_correct", "within_confidence"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_tracked_tickers() -> list[str]:
    """Return list of tickers that have logged predictions."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT DISTINCT ticker FROM predictions ORDER BY ticker"
        ).fetchall()
    return [r["ticker"] for r in rows]


def should_relearn(ticker: str, min_validated: int = 20) -> tuple[bool, str]:
    """Check if a ticker should trigger relearning.

    Criteria:
    1. At least min_validated validated predictions exist
    2. Direction accuracy < 55% or avg error > 3%
    3. No relearn in last 7 days
    """
    with get_db() as conn:
        # Check validated count
        validated = conn.execute("""
            SELECT COUNT(*) as c FROM predictions
            WHERE ticker = ? AND actual_price IS NOT NULL
        """, (ticker,)).fetchone()["c"]

        if validated < min_validated:
            return False, f"Need {min_validated} validated predictions, have {validated}"

        # Check recent accuracy
        metrics = get_accuracy_metrics(ticker)
        dir_acc = metrics["direction_accuracy_pct"]
        avg_err = metrics["avg_error_pct"]

        if dir_acc >= 55 and avg_err <= 3:
            return False, f"Model performing well (dir acc: {dir_acc}%, err: {avg_err}%)"

        # Check cooldown
        last_retrain = conn.execute("""
            SELECT MAX(trained_at) as last FROM model_versions
            WHERE ticker = ?
        """, (ticker,)).fetchone()["last"]

        if last_retrain:
            last_dt = datetime.fromisoformat(last_retrain)
            if (datetime.now() - last_dt).days < 7:
                return False, f"Cooldown: last retrain was {(datetime.now() - last_dt).days} days ago"

        reason = f"Dir accuracy {dir_acc}%, avg error {avg_err}%"
        return True, reason


def compute_adaptive_weights(ticker: str, lookback: int = 30) -> float:
    """Compute optimal xgb_base_weight based on recent prediction accuracy.

    Returns adjusted xgb_base_weight (0.30 to 0.75).
    """
    with get_db() as conn:
        rows = conn.execute("""
            SELECT lstm_price, xgb_price, actual_price
            FROM predictions
            WHERE ticker = ? AND actual_price IS NOT NULL
                AND lstm_price IS NOT NULL AND xgb_price IS NOT NULL
            ORDER BY target_date DESC
            LIMIT ?
        """, (ticker, lookback)).fetchall()

    if len(rows) < 10:
        return 0.55  # Default, not enough data

    lstm_errors = [abs(float(r["lstm_price"]) - float(r["actual_price"])) / float(r["actual_price"])
                   for r in rows if r["actual_price"] and isinstance(r["actual_price"], (int, float)) and r["actual_price"] > 0
                   and isinstance(r["lstm_price"], (int, float))]
    xgb_errors = [abs(float(r["xgb_price"]) - float(r["actual_price"])) / float(r["actual_price"])
                  for r in rows if r["actual_price"] and isinstance(r["actual_price"], (int, float)) and r["actual_price"] > 0
                  and isinstance(r["xgb_price"], (int, float))]

    if not lstm_errors or not xgb_errors:
        return 0.55

    avg_lstm = sum(lstm_errors) / len(lstm_errors)
    avg_xgb = sum(xgb_errors) / len(xgb_errors)

    max_err = max(avg_lstm, avg_xgb, 1e-8)
    adjustment = 0.20 * (avg_lstm - avg_xgb) / max_err
    new_weight = max(0.30, min(0.75, 0.55 + adjustment))

    return round(new_weight, 2)


def get_current_model_version(ticker: str) -> int:
    """Get the latest model version number for a ticker."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT MAX(version) as v FROM model_versions WHERE ticker = ?", (ticker,)
        ).fetchone()
    return row["v"] if row and row["v"] else 0


def log_model_version(ticker: str, reason: str, metrics: dict, weights: dict) -> int:
    """Log a new model version. Returns new version number."""
    new_version = get_current_model_version(ticker) + 1
    with get_db() as conn:
        conn.execute("""
            INSERT INTO model_versions (ticker, version, reason, xgb_weight,
                                        backtest_rmse, backtest_dir_acc)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (ticker, new_version, reason,
              weights.get("xgb"), metrics.get("rmse"), metrics.get("directional_accuracy")))
    return new_version


def relearn_models(ticker: str, lstm_epochs: int = 15, progress_callback=None) -> dict:
    """Execute the relearning pipeline for a ticker.

    1. Fine-tune LSTM with recent data
    2. Retrain XGBoost from scratch
    3. Compute adaptive ensemble weights
    4. Save updated models and log version
    """
    from src.data_fetcher import fetch_stock_data
    from src.feature_engineering import add_technical_indicators, add_lagged_features, get_feature_columns, get_xgboost_feature_columns
    from src.model import load_saved_model, save_model, fine_tune_model
    from src.xgboost_model import train_xgb_model, save_xgb_model

    if progress_callback:
        progress_callback("Loading models...", 0.1)

    # Load existing LSTM
    lstm_result = load_saved_model(ticker)
    if lstm_result is None:
        return {"error": "No saved LSTM model found. Run full training first."}

    lstm_model, scaler = lstm_result

    # Fetch fresh data
    if progress_callback:
        progress_callback("Fetching fresh data...", 0.2)

    df = fetch_stock_data(ticker)
    df = add_technical_indicators(df)

    # Get LSTM features
    feature_cols = get_feature_columns()
    available = [c for c in feature_cols if c in df.columns]
    cols = ["Close"] + [c for c in available if c != "Close"]
    lstm_data = df[cols].values

    # Fine-tune LSTM
    if progress_callback:
        progress_callback("Fine-tuning LSTM...", 0.4)

    lstm_model, ft_history = fine_tune_model(lstm_model, scaler, lstm_data, epochs=lstm_epochs)
    save_model(lstm_model, scaler, ticker)

    # Retrain XGBoost
    if progress_callback:
        progress_callback("Retraining XGBoost...", 0.7)

    df_with_lags = add_lagged_features(df)
    xgb_feature_cols = get_xgboost_feature_columns()
    available_xgb = [c for c in xgb_feature_cols if c in df_with_lags.columns]
    xgb_model, xgb_metrics = train_xgb_model(df_with_lags, available_xgb)
    save_xgb_model(xgb_model, ticker)

    # Compute adaptive weights
    if progress_callback:
        progress_callback("Computing adaptive weights...", 0.9)

    new_xgb_weight = compute_adaptive_weights(ticker)
    old_version = get_current_model_version(ticker)

    # Log new version
    backtest_metrics = xgb_metrics if isinstance(xgb_metrics, dict) else {}
    new_version = log_model_version(
        ticker, "relearn",
        {"rmse": backtest_metrics.get("test_rmse", 0),
         "directional_accuracy": backtest_metrics.get("directional_accuracy", 0)},
        {"xgb": new_xgb_weight, "lstm": 1 - new_xgb_weight},
    )

    if progress_callback:
        progress_callback("Relearning complete!", 1.0)

    return {
        "old_version": old_version,
        "new_version": new_version,
        "new_xgb_weight": new_xgb_weight,
        "fine_tune_history": ft_history,
    }
