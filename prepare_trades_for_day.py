# prepare_trades_for_day.py
import pandas as pd
import polars as pl
import os

# --- Logging setup ---
LOG_FILE = os.path.join("logs", "missing_snapshots.log")
os.makedirs("logs", exist_ok=True)

def log_no_snapshot(day, ts, signal, expiry, spot_price, strategy="composite", reason="No option snapshot"):
    """
    Log missing snapshot info to file instead of console flood.
    """
    msg = f"{strategy} | {day} {ts} {signal}: {reason} (expiry={expiry}, spot={spot_price})\n"
    with open(LOG_FILE, "a") as f:
        f.write(msg)


def find_atm_strike(spot_price, strikes):
    """Find nearest strike rounded to nearest 50."""
    rounded_strikes = [int(round(s / 50) * 50) for s in strikes]
    return min(rounded_strikes, key=lambda x: abs(x - spot_price))


def prepare_trades_for_day(
    spot, options, date_str, signal_col="composite_signal",
    capital=200000, sl=0.015, tp=0.03,
    force_exit="15:15", strategy_label="composite"
):
    """
    Extracts trades for a given day based on signal_col and simulates exits.
    - Buy signal -> Short ATM PUT
    - Sell signal -> Short ATM CALL
    """

    # --- Filter spot for this date ---
    spot_day = spot[spot["datetime"].dt.date == pd.to_datetime(date_str).date()]

    # --- Define trading session start/end ---
    start = pd.to_datetime(f"{date_str} 09:15:00").to_pydatetime().replace(tzinfo=None)
    end   = pd.to_datetime(f"{date_str} 15:30:00").to_pydatetime().replace(tzinfo=None)
    force_exit = pd.to_datetime(f"{date_str} {force_exit}:00").to_pydatetime().replace(tzinfo=None)

    # --- Force expiry_date to date only ---
    options = options.with_columns([
        pl.col("expiry_date").dt.date().alias("expiry_date"),
        pl.col("datetime").dt.replace_time_zone(None).alias("datetime")
    ])

    options_day = options.filter(
        (pl.col("datetime") >= start) &
        (pl.col("datetime") <= end)
    ).collect()

    merged_trades = []
    daily_skips = 0  # count missing cases per day

    for _, row in spot_day.iterrows():
        signal = str(row.get(signal_col, "")).strip()
        if signal not in ["Buy", "Sell"]:
            continue

        expiry = pd.to_datetime(row["closest_expiry"]).date()
        row_dt = pd.to_datetime(row["datetime"]).to_pydatetime().replace(tzinfo=None)
        spot_price = row["close"]

        # --- Wider ±10min window ---
        subset = options_day.filter(
            (pl.col("expiry_date") == expiry) &
            (pl.col("datetime") >= row_dt - pd.Timedelta(minutes=10)) &
            (pl.col("datetime") <= row_dt + pd.Timedelta(minutes=10))
        )

        if subset.height == 0:
            log_no_snapshot(date_str, row_dt, signal, expiry, spot_price, strategy_label, reason="No option snapshot")
            daily_skips += 1
            continue

        # ATM strike
        strikes = subset.select("strike_price").unique().to_series().to_list()
        atm_strike = find_atm_strike(spot_price, strikes)

        opt_type = "PE" if signal == "Buy" else "CE"
        trade = subset.filter(
            (pl.col("strike_price") == atm_strike) & (pl.col("option_type") == opt_type)
        )
        if trade.height == 0:
            log_no_snapshot(date_str, row_dt, signal, expiry, spot_price, strategy_label, reason=f"ATM {atm_strike} {opt_type} not found")
            daily_skips += 1
            continue

        entry_price = trade.select("close").to_series().to_list()[0]

        # --- Exit logic ---
        trade_slice = options_day.filter(
            (pl.col("expiry_date") == expiry) &
            (pl.col("strike_price") == atm_strike) &
            (pl.col("option_type") == opt_type) &
            (pl.col("datetime") > row_dt) &
            (pl.col("datetime") <= force_exit)
        ).sort("datetime")

        exit_time, exit_price = force_exit, None
        for t in trade_slice.iter_rows(named=True):
            price = t["close"]
            if price >= entry_price * (1 + sl):
                exit_time, exit_price = t["datetime"], price
                break
            elif price <= entry_price * (1 - tp):
                exit_time, exit_price = t["datetime"], price
                break

        if exit_price is None:
            exit_price = trade_slice.tail(1)["close"].to_list()[0] if trade_slice.height > 0 else entry_price

        pnl = entry_price - exit_price

        merged_trades.append({
            "signal": signal,
            "strike_price": atm_strike,
            "option_type": opt_type,
            "entry_time": row_dt,
            "entry_price": entry_price,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "exit_reason": (
                "SL" if exit_price >= entry_price * (1 + sl) else
                "TP" if exit_price <= entry_price * (1 - tp) else
                "Force Exit"
            ),
            "expiry_date": expiry,
            "gross_pnl": round(pnl, 2),
            "expenses": 20.5,
            "interest": 0.0,
            "net_pnl": round(pnl - 20.5, 2)
        })

    # --- Daily summary print ---
    if daily_skips > 0:
        print(f"[{strategy_label}] {date_str}: {daily_skips} snapshots missing (logged to {LOG_FILE})")

    return pd.DataFrame(merged_trades)


