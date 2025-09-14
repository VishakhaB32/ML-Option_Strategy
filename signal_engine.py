import numpy as np
import pandas as pd

# -------------------------
# Individual Indicator Signals
# -------------------------
def indicator_signal_rsi(rsi_val, low=30, high=70):
    if np.isnan(rsi_val): return 0
    if rsi_val < low: return 1       # Oversold -> Buy
    if rsi_val > high: return -1     # Overbought -> Sell
    return 0

def indicator_signal_macd(macd_hist):
    if np.isnan(macd_hist): return 0
    return 1 if macd_hist > 0 else -1 if macd_hist < 0 else 0

def indicator_signal_ema(ema_fast, ema_slow):
    if np.isnan(ema_fast) or np.isnan(ema_slow): return 0
    return 1 if ema_fast > ema_slow else -1 if ema_fast < ema_slow else 0

def indicator_signal_bollinger(close, bb_mid, bb_lower, bb_upper):
    if np.isnan(close) or np.isnan(bb_mid): return 0
    if close < bb_lower: return 1    # oversold → Buy
    if close > bb_upper: return -1   # overbought → Sell
    return 0

def indicator_signal_adx(adx_val, threshold=25):
    """ADX > threshold means strong trend. Returns 1 for trend following."""
    if np.isnan(adx_val): return 0
    return 1 if adx_val > threshold else 0

def indicator_signal_cci(cci_val, low=-100, high=100):
    """CCI: Buy if below -100, Sell if above +100."""
    if np.isnan(cci_val): return 0
    if cci_val < low: return 1
    if cci_val > high: return -1
    return 0

def indicator_signal_supertrend(supertrend_val):
    """Supertrend is already +1 (uptrend) or -1 (downtrend)."""
    if np.isnan(supertrend_val): return 0
    return 1 if supertrend_val == 1 else -1 if supertrend_val == -1 else 0


# -------------------------
# Composite Scoring Engine
# -------------------------
def compute_composite(df, weights=None):
    """
    Create a weighted ensemble score based on multiple technical indicators.
    Returns composite_score and composite_signal.
    """
    if weights is None:
        weights = {
            'inhouse': 0.3,
            'rsi': 0.1,
            'macd': 0.1,
            'ema': 0.1,
            'boll': 0.1,
            'adx': 0.1,
            'cci': 0.1,
            'supertrend': 0.2
        }

    scores = []
    for _, row in df.iterrows():
        s = 0.0

        # ✅ In-house signal from raw dataset (if available)
        inhouse = 0
        sig = str(row.get('signal')).lower()
        if sig == 'buy': inhouse = 1
        elif sig == 'sell': inhouse = -1
        s += weights['inhouse'] * inhouse

        # ✅ Technical Indicators
        s += weights['rsi'] * indicator_signal_rsi(row.get('rsi', np.nan))
        s += weights['macd'] * indicator_signal_macd(row.get('macd_hist', np.nan))
        s += weights['ema'] * indicator_signal_ema(row.get('ema_fast', np.nan), row.get('ema_slow', np.nan))
        s += weights['boll'] * indicator_signal_bollinger(
            row.get('close', np.nan), row.get('bb_mid', np.nan),
            row.get('bb_lower', np.nan), row.get('bb_upper', np.nan)
        )
        s += weights['adx'] * indicator_signal_adx(row.get('adx', np.nan))
        s += weights['cci'] * indicator_signal_cci(row.get('cci', np.nan))
        s += weights['supertrend'] * indicator_signal_supertrend(row.get('supertrend', np.nan))

        scores.append(s)

    df['composite_score'] = scores

    # ✅ Dynamic thresholding for more flexible signal generation
    df['composite_signal'] = df['composite_score'].apply(
        lambda x: 'Buy' if x > 0.25 else ('Sell' if x < -0.25 else 'Hold')
    )
    return df


# ✅ Wrapper alias so backtest.py works
def build_composite_signal(df, weights=None):
    """Wrapper for compute_composite (used in backtest.py)."""
    return compute_composite(df, weights)

