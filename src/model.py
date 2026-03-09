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
    """Build a CNN + LSTM hybrid model with temporal attention.

    Architecture:
    1. 1D CNN layers extract local patterns (candlestick formations, short-term trends)
    2. Bidirectional LSTM captures long-range sequential dependencies
    3. Temporal Attention lets the model focus on the most relevant time steps
    4. Dense head produces multi-step forecast

    This hybrid approach outperforms pure LSTM because:
    - CNN finds local features (2-5 day patterns) that LSTM may dilute over 90 days
    - Attention weighs recent volatile days more heavily than quiet periods
    """
    from tensorflow.keras.layers import (
        Bidirectional, BatchNormalization, Conv1D, MaxPooling1D,
        Multiply, Permute, RepeatVector, Lambda, Flatten, Concatenate,
    )
    import tensorflow.keras.backend as K

    inputs = Input(shape=(LOOKBACK, n_features))

    # === CNN Branch: Local pattern extraction ===
    # Captures short-term patterns like candlestick formations, gaps, spikes
    c = Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(inputs)
    c = Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(c)
    c = Dropout(0.2)(c)
    c = Conv1D(filters=32, kernel_size=5, activation="relu", padding="same")(c)
    c = Dropout(0.2)(c)

    # === Merge CNN features with original input for LSTM ===
    merged = Concatenate(axis=-1)([inputs, c])

    # === LSTM Branch: Sequential pattern learning ===
    x = Bidirectional(LSTM(128, return_sequences=True))(merged)
    x = Dropout(0.25)(x)
    x = LSTM(96, return_sequences=True)(x)
    x = Dropout(0.2)(x)

    # === Temporal Attention Mechanism ===
    # Learns which time steps (days) are most important for prediction
    # e.g., days with high volume or big price moves get higher attention
    attn = Dense(1, activation="tanh")(x)  # (batch, timesteps, 1)
    attn = Flatten()(attn)  # (batch, timesteps)
    attn = Dense(x.shape[1], activation="softmax")(attn)  # attention weights
    attn = RepeatVector(96)(attn)  # (batch, features, timesteps)
    attn = Permute((2, 1))(attn)  # (batch, timesteps, features)
    x = Multiply()([x, attn])  # Apply attention weights

    # Final LSTM to compress attended sequence
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.15)(x)

    # === Dense Head ===
    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(forecast_horizon)(x)

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


def fine_tune_model(
    model: Model,
    scaler: MinMaxScaler,
    feature_data: np.ndarray,
    epochs: int = 15,
    learning_rate: float = 1e-5,
) -> tuple[Model, dict]:
    """Fine-tune an existing LSTM model on recent data with a low learning rate.

    Uses only the most recent ~500 trading days to focus on recent patterns.
    Low learning rate prevents catastrophic forgetting.
    """
    from tensorflow.keras.optimizers import Adam

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="huber", metrics=["mae"])

    scaled_data = scaler.transform(feature_data)
    recent = scaled_data[-500:] if len(scaled_data) > 500 else scaled_data
    X, y = prepare_sequences(recent)

    if len(X) == 0:
        return model, {}

    history = model.fit(
        X, y,
        epochs=epochs,
        batch_size=32,
        validation_split=0.15,
        verbose=0,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    )

    return model, history.history


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
