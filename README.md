# NIFTY Options Intraday Backtesting Project

## Overview
This project implements and backtests **three intraday trading strategies** on **NSE Nifty index options** over a one-year period:

1. **Mean Reversion Strategy**  
2. **Directional Strategy**  
3. **Semi-Directional Strategy**

The objective is to analyse performance across different market behaviours and evaluate both individual and combined portfolio performance using key quantitative metrics.

---

## Data

### 🔹 Why Synthetic Data?
I initially attempted to fetch real-time or historical intraday NIFTY options data from public APIs like **Yahoo Finance (`yfinance`)**, **NSEPy**, and other Python libraries.  
However, none of these sources provided reliable **minute-level option chain data**.

To ensure backtesting logic could still be validated end-to-end, I generated **synthetic yet realistic datasets** for both the underlying index and its options.  
The synthetic data mimics intraday volatility, volume, and price behaviour consistent with NIFTY options.

---

### 🔹 Dataset Description

#### 1️⃣ `underlying_minute.csv`
Contains 1-minute interval OHLCV data for the NIFTY index.  
| Column | Description |
|:--------|:-------------|
| `datetime` | Timestamp (1-minute frequency) |
| `open` | Opening price of the minute |
| `high` | Highest price within the minute |
| `low` | Lowest price within the minute |
| `close` | Closing price of the minute |
| `volume` | Simulated trade volume |

---

#### 2️⃣ `options_minute.csv`
Simulated 1-minute OHLCV data for multiple NIFTY option contracts.  
| Column | Description |
|:--------|:-------------|
| `datetime` | Timestamp (1-minute frequency) |
| `symbol` | Option identifier (e.g., `NIFTY03OCT2421850CE`) |
| `strike` | Option strike price |
| `expiry` | Expiry date (weekly/monthly) |
| `option_type` | CE for Call / PE for Put |
| `open`, `high`, `low`, `close` | Option prices for the minute |
| `volume` | Simulated trade volume |

---

## Strategy Descriptions

### 1. Mean Reversion
- Based on the principle that prices revert to their short-term mean.  
- Buys oversold options and sells overbought options relative to a moving average band.  
- Works well in range-bound markets.

### 2. Directional
- Captures intraday trends using short-term momentum or breakout signals.  
- Trades in the direction of the underlying’s intraday trend.  
- Effective in trending markets.

### 3. Semi-Directional
- Hybrid of the two above — partially delta-hedged directional exposure.  
- Adjusts position sizing dynamically based on volatility and intraday bias.

---

## Performance Metrics
Each strategy computes and logs:
- **CAGR (Compounded Annual Growth Rate)**
- **MDD (Maximum Drawdown)**
- **Sharpe Ratio**
- **Calmar Ratio**
- **Win Rate**
- **Average Profit/Loss per trade**
- **Number of trades**
- **Equity curve visualization**

---

## Combined Portfolio
All three strategies are blended into a single portfolio.  
The combined equity curve is generated, targeting:
- **Calmar Ratio ≥ 5**, and  
- Diversified exposure across market regimes (mean-reverting, trending, and mixed).

---

## Repository Structure

```
.
├── data/
│   ├── underlying_minute.csv
│   ├── options_minute.csv
├── backtest.py
├── analysis.ipynb
├── README.md
└── reports/
    ├── individual_equity_curves.png
    ├── combined_equity_curve.png
    └── performance_report.pdf
```

---

## Notes
- The synthetic data is for demonstration and validation purposes only.  
  Replace it with real NIFTY option chain data if available from a paid data provider.
- All calculations assume **no transaction cost** and **instant execution** for simplicity.
- For realistic deployment, incorporate **slippage**, **broker fees**, and **liquidity filters**.

---

## Future Improvements
- Integrate live NSE data using official APIs or paid data vendors.
