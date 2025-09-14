import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib

from data_prep import load_spot_data, load_options_data
from prepare_trades_for_day import prepare_trades_for_day
from indicators import add_indicators
from signal_engine import build_composite_signal

# -----------------------------
# Parameters
# -----------------------------
CAPITAL = 200_000
STOP_LOSS = -0.015   # -1.5%
TAKE_PROFIT = 0.03   # +3%
FORCE_EXIT = "15:15"

# ensure folders exist
os.makedirs("results", exist_ok=True)
os.makedirs("model_predictions", exist_ok=True)


def run_backtest(spot, options, signal_col, label, outdir,
                 start_date="2023-01-02", end_date="2023-01-31",
                 save_plots=True):
    """
    Run per-day backtest. `spot` must include the column named `signal_col`.
    prepare_trades_for_day expects a column named 'composite_signal', so we
    pass a renamed copy. Signals are forced to clean strings.
    """

    all_trades = []
    date_range = pd.date_range(start=start_date, end=end_date, freq="B")

    # 🔧 Force signal to plain strings (avoid Series truth-value errors)
    if signal_col in spot.columns:
        spot[signal_col] = spot[signal_col].astype(str).str.strip()

    # Quick distribution check
    if signal_col not in spot.columns:
        print(f"⚠️ Signal column '{signal_col}' not present in spot dataframe.")
    else:
        print(f"📊 {label} signal distribution (sample):")
        print(spot[signal_col].value_counts().to_dict())

    for d in tqdm(date_range, desc=f"Backtesting {label}"):
        day_str = d.strftime("%Y-%m-%d")
        try:
            # copy & rename to composite_signal
            spot_temp = spot.copy()
            spot_temp["composite_signal"] = spot_temp[signal_col].astype(str).str.strip()

            # trades = prepare_trades_for_day(
            #     spot_temp, options, day_str,
            #     capital=CAPITAL, sl=STOP_LOSS, tp=TAKE_PROFIT, force_exit=FORCE_EXIT
            # )
            trades = prepare_trades_for_day(
            spot,                # no renaming here
            options, 
            day_str,
            capital=CAPITAL, 
            sl=STOP_LOSS, 
            tp=TAKE_PROFIT,
            force_exit=FORCE_EXIT,
            signal_col=signal_col,   # ✅ let function pick correct signal
            strategy_label=label
            )
       


            if trades is not None and len(trades) > 0:
                all_trades.append(trades)
        except Exception as e:
            print(f"[WARN] Skipping {day_str}: {e}")

    if not all_trades:
        print(f"⚠️ No trades generated for {label} strategy in given period.")
        return None

    df_trades = pd.concat(all_trades, ignore_index=True)
    out_trades_path = os.path.join(outdir, f"{label}_trades.csv")
    df_trades.to_csv(out_trades_path, index=False)
    print(f"Saved trades -> {out_trades_path}")

    # -----------------------------
    # Equity curve
    # -----------------------------
    if "net_pnl" not in df_trades.columns:
        if "gross_pnl" in df_trades.columns:
            df_trades["net_pnl"] = df_trades["gross_pnl"]
        else:
            df_trades["net_pnl"] = 0.0

    df_trades["cum_pnl"] = df_trades["net_pnl"].cumsum()
    df_trades["equity"] = CAPITAL + df_trades["cum_pnl"]

    time_index_col = "exit_time" if "exit_time" in df_trades.columns else "entry_time"
    equity_curve = df_trades.set_index(pd.to_datetime(df_trades[time_index_col]))["equity"]

    if save_plots:
        # Equity curve plot
        plt.figure(figsize=(12, 6))
        plt.plot(equity_curve, label=f"{label} Equity Curve")
        plt.title(f"{label} Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Equity (₹)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(outdir, f"{label}_equity_curve.png"))
        plt.close()

        # Drawdown plot
        rolling_max = equity_curve.cummax()
        drawdown = equity_curve - rolling_max
        plt.figure(figsize=(12, 6))
        plt.plot(drawdown, label="Drawdown")
        plt.title(f"{label} Drawdown")
        plt.xlabel("Date")
        plt.ylabel("Drawdown (₹)")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(outdir, f"{label}_drawdown.png"))
        plt.close()

    # -----------------------------
    # Metrics
    # -----------------------------
    total_return = (equity_curve.iloc[-1] - CAPITAL) / CAPITAL
    max_dd = (equity_curve - equity_curve.cummax()).min()
    sharpe = (
        df_trades["net_pnl"].mean() / df_trades["net_pnl"].std() * (252 ** 0.5)
        if df_trades["net_pnl"].std() > 0 else 0
    )
    metrics = {
        "Strategy": label,
        "Total Return %": round(total_return * 100, 2),
        "Max Drawdown": round(float(max_dd), 2),
        "Sharpe Ratio": round(float(sharpe), 2),
        "Final Equity": round(float(equity_curve.iloc[-1]), 2),
        "Total Trades": len(df_trades)
    }
    
    pd.DataFrame([metrics]).to_csv(os.path.join(outdir, f"{label}_metrics.csv"), index=False)

    print(f"✅ {label} backtest complete! Results saved in '{outdir}/'")
    return metrics


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    # ---- Load Spot & Options ----
    spot = load_spot_data()           # pandas DataFrame
    spot = add_indicators(spot)       # compute indicators (macd, rsi, etc.)
    spot = build_composite_signal(spot)  # add composite_signal
    options = load_options_data()     # Polars lazy frame or DataFrame depending on your loader

    # ---- Load ML Bundle (model + scaler + features) ----
    bundle_path = "model_result/best_model_bundle.pkl"
    if not os.path.exists(bundle_path):
        print(f"❗ Model bundle not found: {bundle_path}. ML backtest will be skipped.")
        # Run only composite then exit
        metrics_composite = run_backtest(
            spot, options, "composite_signal", "composite", "results",
            start_date="2023-02-02", end_date="2023-02-27", save_plots=False
        )
        sys.exit(0)

    bundle = joblib.load(bundle_path)
    model = bundle.get("model")
    scaler = bundle.get("scaler")
    features = bundle.get("features")

    if model is None or scaler is None or features is None:
        print("❗ Invalid model bundle (missing model/scaler/features). Aborting ML backtest.")
        sys.exit(1)

    # -----------------------------
    # Recreate missing engineered features for backtest (robust)
    # -----------------------------
    # ensure returns/volatility exist
    spot = spot.copy()
    if "returns" not in spot.columns:
        spot["returns"] = spot["close"].pct_change().fillna(0)
    if "volatility" not in spot.columns:
        spot["volatility"] = spot["returns"].rolling(20).std().fillna(0)

    # Ensure lag features are always created
    for lag in [1, 2, 5]:
        cl = f"close_lag{lag}"
        rl = f"rsi_lag{lag}"
        if cl not in spot.columns:
            spot[cl] = spot["close"].shift(lag)
        if rl not in spot.columns:
            spot[rl] = spot["rsi"].shift(lag)
        # fill NA with current value (start of series)
        spot[cl] = spot[cl].fillna(spot["close"])
        spot[rl] = spot[rl].fillna(spot["rsi"])

    # -----------------------------
    # Prepare scaled features for ML (safe)
    # -----------------------------
    missing_cols = [c for c in features if c not in spot.columns]
    if missing_cols:
        print("⚠️ Warning: Missing features in spot data:", missing_cols)
        # create missing features as zeros so scaler doesn't fail
        for c in missing_cols:
            spot[c] = 0.0

    # fill and transform
    X_for_model = spot[features].ffill().fillna(0)
    try:
        X_scaled = scaler.transform(X_for_model)
    except Exception as e:
        print("❗ Error transforming features for ML model:", e)
        sys.exit(1)

    # Predict and safe-map to strings
    y_pred = model.predict(X_scaled)
    mapping_back = {0: "Sell", 1: "Hold", 2: "Buy"}
    spot["ml_signal"] = pd.Series(y_pred).map(mapping_back).astype(str)

    # Debug: show distribution
    print("✅ Unique ML signals in backtest set:", spot["ml_signal"].unique())
    print("📊 ML signal counts:\n", spot["ml_signal"].value_counts().to_dict())

    # If ML predicts only Hold or empty, log and continue (no trades)
    if spot["ml_signal"].nunique() == 1 and spot["ml_signal"].unique()[0] == "Hold":
        print("⚠️ ML predicted only 'Hold' over the period. No trades expected.")

    # ---- Run Backtests ----
    # Composite → save only metrics/trades (no plots)
    metrics_composite = run_backtest(
        spot, options, "composite_signal", "composite", "results",
        start_date="2023-02-02", end_date="2023-02-05", save_plots=False
    )

    # ML → full results with plots saved under model_predictions/
    metrics_ml = run_backtest(
        spot, options, "ml_signal", "ml", "model_predictions",
        start_date="2023-02-02", end_date="2023-02-05", save_plots=True
    )

    # ---- Save comparison (only if both ran) ----
    if metrics_composite and metrics_ml:
        comp_path = "results/composite_trades.csv"
        ml_path = "model_predictions/ml_trades.csv"
        # ensure files exist
        if os.path.exists(comp_path) and os.path.exists(ml_path):
            pd.DataFrame([metrics_composite, metrics_ml]).to_csv(
                "model_predictions/comparison_metrics.csv", index=False
            )
            # load trades and plot combined curve (model_predictions only)
            comp_trades = pd.read_csv(comp_path, parse_dates=["exit_time"])
            ml_trades = pd.read_csv(ml_path, parse_dates=["exit_time"])
            comp_trades["cum_pnl"] = comp_trades.get("net_pnl", comp_trades.get("gross_pnl", 0)).cumsum()
            comp_trades["equity"] = CAPITAL + comp_trades["cum_pnl"]
            ml_trades["cum_pnl"] = ml_trades.get("net_pnl", ml_trades.get("gross_pnl", 0)).cumsum()
            ml_trades["equity"] = CAPITAL + ml_trades["cum_pnl"]

            plt.figure(figsize=(12, 6))
            plt.plot(ml_trades.set_index("exit_time")["equity"], label="ML Strategy")
            plt.plot(comp_trades.set_index("exit_time")["equity"], label="Composite Strategy")
            plt.title("Equity Curve Comparison: Composite vs ML")
            plt.xlabel("Date")
            plt.ylabel("Equity (₹)")
            plt.legend()
            plt.grid(True)
            plt.savefig("model_predictions/comparison_equity_curve.png")
            plt.close()
            print("✅ Comparison equity curve saved in model_predictions/comparison_equity_curve.png")
        else:
            print("⚠️ Could not find trades CSVs for comparison:", comp_path, ml_path)

    print("All done.")