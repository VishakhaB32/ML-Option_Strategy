import pandas as pd
import numpy as np

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

# -------------------------
# Core Indicators
# -------------------------
def macd(df, fast=12, slow=26, signal=9):
    ema_fast = ema(df['close'], fast)
    ema_slow = ema(df['close'], slow)
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    df['macd'] = macd_line
    df['macd_signal'] = macd_signal
    df['macd_hist'] = df['macd'] - df['macd_signal']
    return df

def rsi(df, length=14):
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ma_up = up.rolling(length).mean()
    ma_down = down.rolling(length).mean()
    rs = ma_up / (ma_down + 1e-9)
    df['rsi'] = 100 - (100/(1+rs))
    return df

def atr(df, length=14):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(length).mean()
    return df

def bollinger(df, length=20, mult=2):
    ma = df['close'].rolling(length).mean()
    std = df['close'].rolling(length).std()
    df['bb_mid'] = ma
    df['bb_upper'] = ma + mult*std
    df['bb_lower'] = ma - mult*std
    return df

def stochastic(df, k_period=14, d_period=3):
    low_min = df['low'].rolling(k_period).min()
    high_max = df['high'].rolling(k_period).max()
    df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-9)
    df['stoch_d'] = df['stoch_k'].rolling(d_period).mean()
    return df

# -------------------------
# Extra Indicators
# -------------------------
def adx(df, length=14):
    """Average Directional Index (trend strength)."""
    up_move = df['high'].diff()
    down_move = df['low'].diff().abs()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(length).mean()

    plus_di = 100 * pd.Series(plus_dm).rolling(length).mean() / (atr14 + 1e-9)
    minus_di = 100 * pd.Series(minus_dm).rolling(length).mean() / (atr14 + 1e-9)
    dx = 100 * (plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-9)
    df['adx'] = dx.rolling(length).mean()
    return df

def cci(df, length=20):
    """Commodity Channel Index."""
    tp = (df['high'] + df['low'] + df['close']) / 3
    ma = tp.rolling(length).mean()
    md = (tp - ma).abs().rolling(length).mean()
    df['cci'] = (tp - ma) / (0.015 * md + 1e-9)
    return df

def supertrend(df, period=10, multiplier=3):
    """SuperTrend indicator."""
    hl2 = (df['high'] + df['low']) / 2
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    atr_ = tr.rolling(period).mean()

    upperband = hl2 + multiplier * atr_
    lowerband = hl2 - multiplier * atr_
    trend = [True] * len(df)
    for i in range(1, len(df)):
        if df['close'].iloc[i] > upperband.iloc[i-1]:
            trend[i] = True
        elif df['close'].iloc[i] < lowerband.iloc[i-1]:
            trend[i] = False
        else:
            trend[i] = trend[i-1]
            if trend[i] and lowerband.iloc[i] < lowerband.iloc[i-1]:
                lowerband.iloc[i] = lowerband.iloc[i-1]
            if not trend[i] and upperband.iloc[i] > upperband.iloc[i-1]:
                upperband.iloc[i] = upperband.iloc[i-1]

    df['supertrend'] = np.where(trend, 1, -1)
    return df

# -------------------------
# Lagged Features
# -------------------------
def add_lagged_features(df, lags=[1,2,3]):
    """Add lagged features of close & RSI for better ML learning."""
    for lag in lags:
        df[f'close_lag{lag}'] = df['close'].shift(lag)
        df[f'rsi_lag{lag}'] = df['rsi'].shift(lag)
    return df

# -------------------------
# Master Function
# -------------------------
def compute_all(df):
    """Compute all supported indicators."""
    df = df.copy()
    df = macd(df)
    df = rsi(df)
    df = atr(df)
    df = bollinger(df)
    df = stochastic(df)
    df['ema_fast'] = ema(df['close'], 8)
    df['ema_slow'] = ema(df['close'], 21)

    # ✅ New Indicators
    df = adx(df)
    df = cci(df)
    df = supertrend(df)
    df = add_lagged_features(df)

    return df

# ✅ Wrapper for backtest.py
def add_indicators(df):
    """Alias for compute_all, so backtest.py works seamlessly."""
    return compute_all(df)
