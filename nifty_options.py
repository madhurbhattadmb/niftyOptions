import os
from datetime import datetime, timedelta
import math
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OPTIONS_CSV = "options_minute.csv"
UNDERLYING_CSV = "underlying_minute.csv"
START_DATE = None   # set to None to use full range from CSVs
END_DATE = None
CAPITAL = 100000.0
SLIPPAGE_PCT = 0.0005
FEE_PER_TRADE = 50.0
CONTRACT_MULTIPLIER = 50  # NIFTY option lot size (may change with time)

# Strategy params
MEANREV_Z = 2.0
MEANREV_LOOKBACK = 12
DIRECTIONAL_FAST = 6
DIRECTIONAL_SLOW = 18
SEMI_OTM_PCT = 0.02
SEMI_RV_WINDOW = 12

OUT_DIR = "backtest_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def read_data():
    if not os.path.exists(OPTIONS_CSV):
        raise FileNotFoundError(f"{OPTIONS_CSV} not found. Provide option minute OHLC data.")
    if not os.path.exists(UNDERLYING_CSV):
        raise FileNotFoundError(f"{UNDERLYING_CSV} not found. Provide underlying minute bars.")
    opts = pd.read_csv(OPTIONS_CSV, parse_dates=['datetime'])
    under = pd.read_csv(UNDERLYING_CSV, parse_dates=['datetime'])
    # standardise column names
    under = under.rename(columns={c: c.lower() for c in under.columns})
    under = under.set_index('datetime').sort_index()
    opts['datetime'] = pd.to_datetime(opts['datetime'])
    opts = opts.sort_values(['datetime','symbol','strike','option_type'])
    # apply date filters
    if START_DATE:
        under = under[under.index.date >= START_DATE]
        opts = opts[opts['datetime'].dt.date >= START_DATE]
    if END_DATE:
        under = under[under.index.date <= END_DATE]
        opts = opts[opts['datetime'].dt.date <= END_DATE]
    return under, opts

def choose_weekly_expiry(d):
    # pick next Thursday on or after d
    d0 = d
    while d0.weekday() != 3:
        d0 = d0 + timedelta(days=1)
    return d0

def nearest_strike(spot, step=50):
    return int(round(spot / step) * step)

def option_price_at(opts_df, ts, strike, option_type, expiry):
    # filter
    mask = (
        (opts_df['strike'] == int(strike)) &
        (opts_df['option_type'] == option_type) &
        (opts_df['expiry'] == pd.to_datetime(expiry).date()) &
        (opts_df['datetime'] <= ts)
    )
    sub = opts_df[mask]
    if sub.empty:
        return None
    row = sub.sort_values('datetime').iloc[-1]
    # use mid of open/close if available else close
    if 'open' in row and 'close' in row and not math.isnan(row['open']) and not math.isnan(row['close']):
        return float((row['open'] + row['close']) / 2.0)
    return float(row.get('close', np.nan))

class Position:
    def __init__(self, option_type, strike, expiry, side, size, entry_price, entry_ts):
        self.option_type = option_type
        self.strike = int(strike)
        self.expiry = pd.to_datetime(expiry).date()
        self.side = side  # 'buy' or 'sell'
        self.size = size
        self.entry_price = entry_price
        self.entry_ts = entry_ts
        self.exit_price = None
        self.exit_ts = None

    def pnl(self, exit_price=None):
        price_close = exit_price if exit_price is not None else self.exit_price
        if price_close is None:
            return 0.0
        # For buyer: P&L = (exit - entry) * multiplier * size * (1 for buy, -1 for sell)
        mult = CONTRACT_MULTIPLIER
        direction = 1 if self.side == 'buy' else -1
        return (price_close - self.entry_price) * mult * self.size * direction

class BacktestEngine:
    def __init__(self, under, opts, starting_capital=CAPITAL):
        self.under = under.copy()
        self.opts = opts.copy()
        self.capital = starting_capital
        self.starting_capital = starting_capital
        self.open_positions = []
        self.closed_trades = []  # list of dicts with trade details
        self.equity_times = []
        self.equity_values = []

    def run(self, strategy_fn, name):
        print(f"Running: {name}")
        # iterate minute bars in order
        for ts, row in self.under.iterrows():
            orders = strategy_fn(self, ts)
            # execute orders (market fill at mid + slippage)
            for o in orders:
                self._execute_order(ts, o)
            # mark to market
            self._record_equity(ts)
            # close end-of-day positions
            if ts.time().hour == 15 and ts.time().minute >= 15:  # assume market close around 15:15
                self._close_eod(ts)
        # final EOD close if any
        if len(self.open_positions) > 0:
            self._close_eod(self.under.index[-1])
        eq = pd.Series(self.equity_values, index=self.equity_times).sort_index()
        results = self._metrics(eq, name)
        # save trades and equity
        trades_df = pd.DataFrame(self.closed_trades)
        trades_df.to_csv(os.path.join(OUT_DIR, f"trades_{name}.csv"), index=False)
        eq.to_csv(os.path.join(OUT_DIR, f"equity_{name}.csv"))
        return results

    def _execute_order(self, ts, order):
        # order: dict with keys type(buy/sell), option_type, strike, expiry, size
        price = option_price_at(self.opts, ts, order['strike'], order['option_type'], order['expiry'])
        if price is None or math.isnan(price):
            return False
        slippage = price * SLIPPAGE_PCT
        exec_price = price + slippage if order['type'] == 'buy' else price - slippage
        # money flow: buyers pay premium -> capital reduces, sellers receive premium -> capital increases
        cash_flow = exec_price * CONTRACT_MULTIPLIER * order.get('size', 1) * (1 if order['type'] == 'buy' else -1)
        # buyer pays positive cash, so subtract from capital
        self.capital -= cash_flow
        # fees
        self.capital -= FEE_PER_TRADE
        pos = Position(order['option_type'], order['strike'], order['expiry'], order['type'], order.get('size',1), exec_price, ts)
        self.open_positions.append(pos)
        # log
        #print(f"{ts} EXEC {order['type']} {order['option_type']} {order['strike']} @ {exec_price:.2f}")
        return True

    def _record_equity(self, ts):
        # mark-to-market open positions
        total_unreal = 0.0
        for p in list(self.open_positions):
            mid = option_price_at(self.opts, ts, p.strike, p.option_type, p.expiry)
            if mid is None or math.isnan(mid):
                mid = p.entry_price
            # buyer: unrealized = (mid - entry) * mult * size
            dir = 1 if p.side == 'buy' else -1
            total_unreal += (mid - p.entry_price) * CONTRACT_MULTIPLIER * p.size * dir
        total_equity = self.capital + total_unreal
        self.equity_times.append(ts)
        self.equity_values.append(total_equity)

    def _close_eod(self, ts):
        # close all open positions at last available mid price at or before ts
        for p in list(self.open_positions):
            mid = option_price_at(self.opts, ts, p.strike, p.option_type, p.expiry)
            if mid is None or math.isnan(mid):
                mid = p.entry_price
            # execute closing trade => reverse side
            close_side = 'sell' if p.side == 'buy' else 'buy'
            slippage = mid * SLIPPAGE_PCT
            exec_price = mid + slippage if close_side == 'buy' else mid - slippage
            cash_flow = exec_price * CONTRACT_MULTIPLIER * p.size * (1 if close_side == 'buy' else -1)
            self.capital -= cash_flow
            self.capital -= FEE_PER_TRADE
            p.exit_price = exec_price
            p.exit_ts = ts
            pnl = p.pnl()
            self.closed_trades.append({
                'entry_ts': p.entry_ts, 'exit_ts': p.exit_ts, 'option_type': p.option_type, 'strike': p.strike,
                'expiry': p.expiry, 'side': p.side, 'size': p.size, 'entry_price': p.entry_price, 'exit_price': p.exit_price,
                'pnl': pnl
            })
            self.open_positions.remove(p)
        # after closing, record equity
        self._record_equity(ts)

    def _metrics(self, equity_series, name):
        # equity_series: pd.Series indexed by timestamp
        equity_series = equity_series.sort_index()
        # resample to daily close
        daily = equity_series.resample('D').last().dropna()
        if daily.empty:
            raise RuntimeError(f"No daily equity points for {name}")
        returns = daily.pct_change().dropna()
        total_days = (daily.index[-1] - daily.index[0]).days
        total_years = total_days / 365.25 if total_days>0 else 1/365.25
        start_val = daily.iloc[0]
        end_val = daily.iloc[-1]
        cagr = (end_val / start_val) ** (1 / total_years) - 1 if total_years>0 else 0.0
        # max drawdown
        cum = (1 + returns).cumprod()
        cum = pd.concat([pd.Series([1.0], index=[daily.index[0]]), cum])
        peak = cum.cummax()
        drawdown = (cum - peak) / peak
        maxdd = drawdown.min()
        # annualized Sharpe (assume 0 rf)
        ann_ret = returns.mean() * 252
        ann_vol = returns.std() * math.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
        # Sortino
        neg = returns[returns < 0]
        down_vol = neg.std() * math.sqrt(252) if not neg.empty else 0.0
        sortino = ann_ret / down_vol if down_vol > 0 else np.nan
        # trade stats
        trades_df = pd.DataFrame(self.closed_trades)
        num_trades = len(trades_df)
        win_rate = (trades_df['pnl'] > 0).mean() if num_trades>0 else np.nan
        avg_pnl = trades_df['pnl'].mean() if num_trades>0 else np.nan
        avg_win = trades_df[trades_df['pnl']>0]['pnl'].mean() if num_trades>0 and (trades_df['pnl']>0).any() else np.nan
        avg_loss = trades_df[trades_df['pnl']<=0]['pnl'].mean() if num_trades>0 and (trades_df['pnl']<=0).any() else np.nan
        # Calmar = annual return / abs(max drawdown)
        calmar = cagr / abs(maxdd) if maxdd < 0 else np.nan
        results = {
            'name': name,
            'start_date': daily.index[0].date(),
            'end_date': daily.index[-1].date(),
            'start_value': float(start_val),
            'end_value': float(end_val),
            'cagr': float(cagr),
            'max_drawdown': float(maxdd),
            'sharpe': float(sharpe) if not np.isnan(sharpe) else None,
            'sortino': float(sortino) if not np.isnan(sortino) else None,
            'num_trades': int(num_trades),
            'win_rate': float(win_rate) if not np.isnan(win_rate) else None,
            'avg_pnl_per_trade': float(avg_pnl) if not np.isnan(avg_pnl) else None,
            'avg_win': float(avg_win) if not np.isnan(avg_win) else None,
            'avg_loss': float(avg_loss) if not np.isnan(avg_loss) else None,
            'calmar': float(calmar) if not np.isnan(calmar) else None,
            'equity': equity_series
        }
        # save summary
        summary_lines = [f"Metric,{k},{v}" for k,v in results.items() if k!='equity']
        with open(os.path.join(OUT_DIR, f"summary_{name}.csv"), 'w') as f:
            f.write('\n'.join(summary_lines))
        print(f"{name}: CAGR {results['cagr']:.2%}, MDD {results['max_drawdown']:.2%}, Sharpe {results['sharpe']}")
        return results

def mean_reversion_strategy(bt: BacktestEngine, ts):
    orders = []
    # require lookback
    idx = bt.under.index.get_loc(ts)
    if idx < MEANREV_LOOKBACK:
        return orders
    recent = bt.under['close'].iloc[idx-MEANREV_LOOKBACK:idx]
    rets = recent.pct_change().dropna()
    if len(rets) < 3 or rets.std() == 0:
        return orders
    z = (rets.iloc[-1] - rets.mean()) / rets.std()
    if abs(z) >= MEANREV_Z:
        spot = bt.under['close'].loc[ts]
        strike = nearest_strike(spot)
        expiry = choose_weekly_expiry(ts.date())
        if z > 0:
            # bought PE
            orders.append({'type':'buy','option_type':'PE','strike':strike,'expiry':expiry,'size':1})
        else:
            orders.append({'type':'buy','option_type':'CE','strike':strike,'expiry':expiry,'size':1})
    return orders

def directional_strategy(bt: BacktestEngine, ts):
    orders = []
    idx = bt.under.index.get_loc(ts)
    if idx < DIRECTIONAL_SLOW + 1:
        return orders
    fast = bt.under['close'].iloc[idx-DIRECTIONAL_FAST:idx].mean()
    slow = bt.under['close'].iloc[idx-DIRECTIONAL_SLOW:idx].mean()
    prev_fast = bt.under['close'].iloc[idx-DIRECTIONAL_FAST-1:idx-1].mean()
    prev_slow = bt.under['close'].iloc[idx-DIRECTIONAL_SLOW-1:idx-1].mean()
    if (prev_fast <= prev_slow) and (fast > slow):
        spot = bt.under['close'].loc[ts]
        strike = nearest_strike(spot)
        expiry = choose_weekly_expiry(ts.date())
        orders.append({'type':'buy','option_type':'CE','strike':strike,'expiry':expiry,'size':1})
    elif (prev_fast >= prev_slow) and (fast < slow):
        spot = bt.under['close'].loc[ts]
        strike = nearest_strike(spot)
        expiry = choose_weekly_expiry(ts.date())
        orders.append({'type':'buy','option_type':'PE','strike':strike,'expiry':expiry,'size':1})
    return orders

def semi_directional_strategy(bt: BacktestEngine, ts):
    orders = []
    idx = bt.under.index.get_loc(ts)
    if idx < SEMI_RV_WINDOW:
        return orders
    rets = bt.under['close'].iloc[idx-SEMI_RV_WINDOW:idx].pct_change().dropna()
    rv = rets.std() * math.sqrt(252*78) if len(rets)>0 else 0
    spot = bt.under['close'].loc[ts]
    strike_up = nearest_strike(spot*(1+SEMI_OTM_PCT))
    strike_dn = nearest_strike(spot*(1-SEMI_OTM_PCT))
    expiry = choose_weekly_expiry(ts.date())
    # naive rule: if realized vol low, sell strangle
    if rv < 0.6:
        orders.append({'type':'sell','option_type':'CE','strike':strike_up,'expiry':expiry,'size':1})
        orders.append({'type':'sell','option_type':'PE','strike':strike_dn,'expiry':expiry,'size':1})
    return orders

def combine_equities(eqs, weights):
    # eqs: list of pd.Series aligned by timestamp (can have different indices)
    # resample all to daily last, normalize to capital slices
    daily_eqs = [e.resample('D').last().dropna() for e in eqs]
    # align dates
    all_idx = pd.Index(sorted(set().union(*[d.index for d in daily_eqs])))
    aligned = [d.reindex(all_idx).fillna(method='ffill').fillna(method='bfill') for d in daily_eqs]
    # scale each equity so that initial value = weight * total_capital
    total_cap = CAPITAL
    scaled = []
    for s, w in zip(aligned, weights):
        init = s.iloc[0]
        factor = (total_cap * w) / init if init != 0 else 0
        scaled.append(s * factor)
    combined = sum(scaled)
    return combined

def search_weights_for_calmar(eqs, target_calmar=5.0, steps=11):
    # simple grid over weights that sum to 1 (discretized)
    best = None
    best_w = None
    grid = [i/(steps-1) for i in range(steps)]
    combos = [c for c in itertools.product(grid, repeat=len(eqs)) if abs(sum(c)-1.0) < 1e-6]
    print(f"Searching {len(combos)} weight combinations...")
    for w in combos:
        comb = combine_equities(eqs, w)
        # compute calmar
        daily = comb.resample('D').last().dropna()
        if len(daily) < 2:
            continue
        returns = daily.pct_change().dropna()
        total_days = (daily.index[-1] - daily.index[0]).days
        years = total_days/365.25 if total_days>0 else 1/365.25
        cagr = (daily.iloc[-1]/daily.iloc[0]) ** (1/years) - 1
        cum = (1+returns).cumprod()
        peak = cum.cummax()
        drawdown = (cum - peak)/peak
        maxdd = drawdown.min()
        calmar = cagr / abs(maxdd) if maxdd<0 else np.nan
        if best is None or (not math.isnan(calmar) and calmar > best):
            best = calmar
            best_w = w
            if calmar >= target_calmar:
                break
    return best_w, best

def write_report(perf_list, combined_perf=None, combined_weights=None):
    lines = []
    lines.append("# Backtest Analysis Report")
    lines.append(f"Generated: {datetime.now()}\n")
    for p in perf_list:
        lines.append(f"## Strategy: {p['name']}")
        lines.append(f"Period: {p['start_date']} to {p['end_date']}")
        lines.append(f"Start value: {p['start_value']:.2f}, End value: {p['end_value']:.2f}")
        lines.append(f"CAGR: {p['cagr']:.2%}")
        lines.append(f"Max Drawdown: {p['max_drawdown']:.2%}")
        lines.append(f"Sharpe: {p['sharpe']}")
        lines.append(f"Sortino: {p['sortino']}")
        lines.append(f"Calmar: {p['calmar']}")
        lines.append(f"Trades: {p['num_trades']}, Win rate: {p['win_rate']}, Avg PnL/trade: {p['avg_pnl_per_trade']:.2f}\n")
        # quick insight
        if p['num_trades'] < 10:
            lines.append("*Insight: Low number of trades — statistics may not be stable.*\n")
        if p['cagr'] > 0.5:
            lines.append("*Insight: Very high CAGR — verify data and slippage assumptions.*\n")
        if p['max_drawdown'] < -0.3:
            lines.append("*Insight: Large drawdown — consider risk controls and stops.*\n")
    if combined_perf is not None:
        lines.append("## Combined Portfolio")
        lines.append(f"Weights: {combined_weights}")
        lines.append(f"CAGR: {combined_perf['cagr']:.2%}")
        lines.append(f"Max Drawdown: {combined_perf['max_drawdown']:.2%}")
        lines.append(f"Calmar: {combined_perf['calmar']}")
        lines.append(f"Sharpe: {combined_perf['sharpe']}\n")
        # strengths/weaknesses
        lines.append("### Strengths:")
        lines.append("- Diversification reduces idiosyncratic strategy risk; combined equity smoother than individuals.")
        lines.append("### Weaknesses:")
        lines.append("- Strategies share underlying exposure to NIFTY (not fully independent). Semi-directional short premium can produce tails.")
    with open(os.path.join(OUT_DIR, 'analysis_report.md'), 'w') as f:
        f.write('\n'.join(lines))

def main():
    under, opts = read_data()
    # Run three strategies separately
    bt_mean = BacktestEngine(under, opts, starting_capital=CAPITAL)
    res_mean = bt_mean.run(mean_reversion_strategy, 'MeanReversion')

    bt_dir = BacktestEngine(under, opts, starting_capital=CAPITAL)
    res_dir = bt_dir.run(directional_strategy, 'Directional')

    bt_semi = BacktestEngine(under, opts, starting_capital=CAPITAL)
    res_semi = bt_semi.run(semi_directional_strategy, 'SemiDirectional')

    # plot individual equity curves
    plt.figure(figsize=(12,6))
    for r in [res_mean, res_dir, res_semi]:
        eq = r['equity']
        eq.plot(label=r['name'])
    plt.legend(); plt.title('Individual Equity Curves'); plt.xlabel('Time'); plt.ylabel('Equity')
    plt.savefig(os.path.join(OUT_DIR,'equity_individual.png'))

    # combine by searching weights
    eqs = [res_mean['equity'], res_dir['equity'], res_semi['equity']]
    best_w, best_calmar = search_weights_for_calmar(eqs, target_calmar=5.0, steps=7)
    if best_w is None:
        print("No feasible weight vector found that sums to 1 with given discretization. Using equal weights.")
        best_w = (1/3, 1/3, 1/3)
    combined_eq = combine_equities(eqs, best_w)
    combined_eq.to_csv(os.path.join(OUT_DIR, 'equity_combined.csv'))

    # compute combined metrics
    # reuse BacktestEngine._metrics by creating a temporary engine with closed_trades empty and equity set
    class DummyEngine:
        def __init__(self, equity, closed_trades):
            self.closed_trades = closed_trades
        def _metrics(self, equity_series, name):
            # reuse the function from BacktestEngine by binding
            return BacktestEngine._metrics(self, equity_series, name)

    dummy = DummyEngine(combined_eq, [])
    combined_perf = BacktestEngine._metrics(dummy, combined_eq, 'CombinedPortfolio')

    # plot combined
    plt.figure(figsize=(10,5))
    combined_eq.plot(label=f"Combined (weights={best_w}, calmar={combined_perf['calmar']:.2f})")
    plt.legend(); plt.title('Combined Equity Curve'); plt.savefig(os.path.join(OUT_DIR,'equity_combined.png'))

    # write analysis report
    perf_list = [res_mean, res_dir, res_semi]
    write_report(perf_list, combined_perf, combined_weights=best_w)

    print('\nBacktest complete. Outputs saved to folder:', OUT_DIR)

if __name__ == '__main__':
    main()