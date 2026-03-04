import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler


LOOKBACK = 90  # Number of past days to use for prediction
MAX_FORECAST = 30  # Maximum forecast horizon (model outputs this many days)
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def build_lstm_model(n_features: int, forecast_horizon: int = MAX_FORECAST) -> Model:
    """Build a multi-step direct LSTM model for stock price prediction.

    Instead of predicting 1 day and feeding it back recursively,
    this model directly predicts all future days at once,
    which eliminates the lag/smoothing problem.
    """
    inputs = Input(shape=(LOOKBACK, n_features))
    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(forecast_horizon)(x)  # Predict all days at once

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="huber", metrics=["mae"])
    return model


def prepare_sequences(data: np.ndarray, lookback: int = LOOKBACK, forecast_horizon: int = MAX_FORECAST):
    """Create sliding window sequences with multi-step targets."""
    X, y = [], []
    for i in range(lookback, len(data) - forecast_horizon + 1):
        X.append(data[i - lookback:i])
        # Target: Close prices for the next forecast_horizon days
        y.append(data[i:i + forecast_horizon, 0])
    return np.array(X), np.array(y)


def prepare_sequences_single(data: np.ndarray, lookback: int = LOOKBACK):
    """Create sliding window sequences with single-step target (for evaluation)."""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def train_model(
    feature_data: np.ndarray,
    epochs: int = 80,
    batch_size: int = 32,
    validation_split: float = 0.2,
    progress_callback=None,
) -> tuple[Model, MinMaxScaler, dict]:
    """
    Train multi-step LSTM model on feature data.

    Args:
        feature_data: numpy array with Close price as first column
        epochs: number of training epochs
        batch_size: training batch size
        validation_split: fraction of data for validation
        progress_callback: optional callable(epoch, total_epochs) for progress updates

    Returns:
        (trained_model, scaler, training_history)
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(feature_data)

    X, y = prepare_sequences(scaled_data)

    if len(X) == 0:
        raise ValueError("Not enough data to create training sequences. Need at least "
                         f"{LOOKBACK + MAX_FORECAST} data points.")

    split_idx = int(len(X) * (1 - validation_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    model = build_lstm_model(n_features=X.shape[2])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ]

    if progress_callback:
        class ProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress_callback(epoch + 1, epochs)
        callbacks.append(ProgressCallback())

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=0,
    )

    return model, scaler, history.history


def save_model(model: Model, scaler: MinMaxScaler, ticker: str):
    """Save model and scaler to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    safe_ticker = ticker.replace(".", "_")
    model.save(os.path.join(MODELS_DIR, f"{safe_ticker}_model.keras"))
    np.save(os.path.join(MODELS_DIR, f"{safe_ticker}_scaler_min.npy"), scaler.data_min_)
    np.save(os.path.join(MODELS_DIR, f"{safe_ticker}_scaler_max.npy"), scaler.data_max_)


def load_saved_model(ticker: str) -> tuple[Model, MinMaxScaler] | None:
    """Load a previously saved model and scaler."""
    safe_ticker = ticker.replace(".", "_")
    model_path = os.path.join(MODELS_DIR, f"{safe_ticker}_model.keras")
    min_path = os.path.join(MODELS_DIR, f"{safe_ticker}_scaler_min.npy")
    max_path = os.path.join(MODELS_DIR, f"{safe_ticker}_scaler_max.npy")

    if not all(os.path.exists(p) for p in [model_path, min_path, max_path]):
        return None

    model = load_model(model_path)
    scaler = MinMaxScaler()
    scaler.data_min_ = np.load(min_path)
    scaler.data_max_ = np.load(max_path)
    scaler.data_range_ = scaler.data_max_ - scaler.data_min_
    scaler.scale_ = 1.0 / scaler.data_range_
    scaler.min_ = -scaler.data_min_ * scaler.scale_
    scaler.n_features_in_ = len(scaler.data_min_)
    scaler.feature_range = (0, 1)

    return model, scaler
