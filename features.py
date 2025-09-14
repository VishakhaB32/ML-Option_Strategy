import pandas as pd
from sklearn.preprocessing import StandardScaler

def prepare_features(spot_df):
    # -----------------------------
    # 1. Base feature list
    # -----------------------------
    features = [
        "open", "high", "low", "close",
        "macd", "macd_signal", "macd_hist",
        "rsi", "atr", "bb_mid", "bb_upper", "bb_lower",
        "stoch_k", "stoch_d", "ema_fast", "ema_slow"
    ]

    # -----------------------------
    # 2. Engineered features
    # -----------------------------
    spot_df = spot_df.copy()
    spot_df["returns"] = spot_df["close"].pct_change().fillna(0)
    spot_df["volatility"] = spot_df["returns"].rolling(20).std().fillna(0)

    for lag in [1, 2, 5]:
        spot_df[f"close_lag{lag}"] = spot_df["close"].shift(lag).fillna(spot_df["close"])
        spot_df[f"rsi_lag{lag}"] = spot_df["rsi"].shift(lag).fillna(spot_df["rsi"])

    # Extend feature list with engineered ones
    features.extend([
        "returns", "volatility",
        "close_lag1", "close_lag2", "close_lag5",
        "rsi_lag1", "rsi_lag2", "rsi_lag5"
    ])

    # -----------------------------
    # 3. Handle NaNs
    # -----------------------------
    X = spot_df[features].ffill().fillna(0)

    # -----------------------------
    # 4. Scale
    # -----------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, X_scaled, scaler, features
