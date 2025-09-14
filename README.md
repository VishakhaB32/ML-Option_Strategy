# 📈 Options Trading Backtest & ML Strategy

This repository implements a modular backtesting framework for **NIFTY Options strategies** using both **rule-based technical indicators** and a **Machine Learning model**.

We combine multiple technical indicators into a **composite signal**, and also train an ML classifier to predict Buy/Sell/Hold. Both approaches are backtested on **2023 Spot + Options data**.

---

## 🔧 Environment Setup & Dependencies

Clone the repository and install dependencies:

```bash
git clone <your-repo-url>
cd strategy-backtest
pip install -r requirements.txt
```

**Key Python libraries used:**

* `pandas`, `numpy` → data manipulation
* `matplotlib` → plotting equity/drawdown
* `joblib` → load/save ML models
* `scikit-learn` → ML training & preprocessing
* `polars` → efficient options data handling
* `tqdm` → progress bars

---

## ▶️ Script Usage

### Run Backtest

```bash
python backtest.py
```

This will:

* Load `spot_with_signals_2023.csv` and `options_data_2023.csv`
* Compute technical indicators
* Build composite signal
* Run backtests for:

  * **Composite Strategy**
  * **ML Strategy** (if trained model bundle is available)
* Save results in `results/` and `model_predictions/`

---

## 📊 Indicators Implemented

Indicators are computed in [`indicators.py`](indicators.py):

* **MACD** (12/26 EMA difference with signal line)
* **RSI** (14-period momentum oscillator)
* **ATR** (Average True Range for volatility)
* **Bollinger Bands** (20-period SMA ± 2σ)
* **Stochastic Oscillator** (%K and %D)
* **ADX** (trend strength)
* **CCI** (Commodity Channel Index)
* **SuperTrend** (ATR-based trend-following)
* **EMA crossover** (fast = 8, slow = 21)
* **Lagged features** of Close & RSI for ML model

---

## ⚡ Composite Signal Logic

Implemented in [`signal_engine.py`](signal_engine.py).
Each indicator contributes a **+1 (Buy)**, **-1 (Sell)**, or **0 (Hold)**.

Weighted ensemble (default weights):

* In-house signal → 0.3
* RSI → 0.1
* MACD → 0.1
* EMA crossover → 0.1
* Bollinger Bands → 0.1
* ADX → 0.1
* CCI → 0.1
* SuperTrend → 0.2

Final **composite signal**:

* `Buy` if score > +0.25
* `Sell` if score < -0.25
* `Hold` otherwise

---

## 🤖 Machine Learning Model

* Features: Technical indicators + lagged values (close, RSI, returns, volatility).
* Target: Buy / Hold / Sell classification.
* Trained using scikit-learn (best model saved in `model_result/best_model_bundle.pkl`).
* Model predictions stored in `spot["ml_signal"]`.

---

## 🧪 Backtest Parameters

* **Capital**: ₹200,000
* **Trade Entry**:

  * Buy signal → Sell ATM PUT
  * Sell signal → Sell ATM CALL
* **Stop Loss**: 1.5%
* **Take Profit**: 3%
* **Force Exit**: 15:15 IST
* **Expenses**: flat ₹20.5 per trade
* **Slippage / Margin Interest**: ignored

---

## 📈 Interpreting Results

### Outputs

Each backtest generates:

* `trades.csv` → trade log (entry, exit, strike, option type, P\&L)
* `metrics.csv` → performance metrics:

  * **Total Return %**
  * **Max Drawdown**
  * **Sharpe Ratio**
  * **Final Equity**
  * **Total Trades**
* `equity_curve.png` → cumulative equity over time
* `drawdown.png` → drawdowns relative to peak equity

### Example:

* **Positive equity curve** → strategy is profitable.
* **Sharpe Ratio > 1** → risk-adjusted performance is acceptable.
* **Max Drawdown** → measures worst historical dip.

---

## 📂 Deliverables

This repository contains:

```
strategy-backtest/
├── data/
│   ├── spot_with_signals_2023.csv
│   ├── options_data_2023.csv
├── indicators.py
├── signal_engine.py
├── model.py
├── backtest.py
├── prepare_trades_for_day.py
├── utils.py
├── requirements.txt
├── README.md
├── results/                # Composite backtest results
│   ├── equity_curve.png
│   ├── drawdown.png
│   ├── metrics.csv
│   └── trades.csv
└── model_predictions/      # ML backtest results
    ├── equity_curve.png
    ├── drawdown.png
    ├── metrics.csv
    └── trades.csv
```
