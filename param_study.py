"""Favourite strategy, $1 stake: parameter grid chosen on half 1 (18.08-04.09), tested on half 2 (05.09-25.09).
Parameters: entry time, favourite price range, model veto, favourite-streak, Binance momentum, spread, hours, coins, exits."""
import sys, numpy as np, pandas as pd, itertools, time
from alib import *
from a_incidents import in_inc
pd.set_option('display.width', 250); pd.set_option('display.max_columns', None)
OUT = D
MID = np.datetime64('2026-09-05T00:00', 'ms').astype('int64')
TAUS = [600, 480, 360, 300, 240, 180, 120, 90, 60]; DELAY = 1
SLS = [0.3, 0.5, 0.7]            # stop-loss: sell when favourite's bid <= level
def buy_shares(px): return 1.0 / (px * (1 + 0.07 * (1 - px)))          # $1 covers price + taker fee
def sell_value(sh, b): return sh * b * (1 - 0.07 * (1 - b))              # proceeds after taker fee on the sale
rows = []
t0 = time.time()
for coin in COINS:
    P = load(coin); y = P['y']; st = P['start']; b = P['bid'].astype(float); a = P['ask'].astype(float); mid = (a + b) / 2
    fair = P['fair'].astype(float); S = P['S']; inc = in_inc(st); hour = pd.to_datetime(st, unit='ms').hour.values
    for tau in TAUS:
        k = 900 - tau; m0 = mid[:, k]; up = m0 > 0.5; sgn = np.where(up, 1, -1)
        px = np.where(up, a[:, k + DELAY], 1 - b[:, k + DELAY])
        ok = np.isfinite(m0) & np.isfinite(px) & (a[:, k] - b[:, k] <= 0.10) & (m0 != 0.5) & (px > 0) & (px < 1) & ~inc
        win = np.where(up, y, 1 - y); sh = buy_shares(px)
        look = min(300, k)
        seg = mid[:, k - look:k]; favseg = np.where(up[:, None], seg, 1 - seg)
        fb = np.where(up[:, None], b[:, k + DELAY:], 1 - a[:, k + DELAY:])            # favourite's bid path after entry
        d = pd.DataFrame(dict(coin=coin, start=st, tau=tau, px=px, win=win, pnl=sh * win - 1,
            model_edge=(np.where(up, fair[:, k], 1 - fair[:, k]) - px - fee(px)) * 100,
            bn_mom=sgn * (S[:, k] / S[:, k - look] - 1) * 1e4, streak=(favseg > 0.5).all(1),
            spread=(a[:, k] - b[:, k]) * 100, hour=hour))
        for sl in SLS:                                                                  # stop-loss exits
            hit = np.nan_to_num(fb, nan=1.0) <= sl; any_ = hit.any(1); j = np.argmax(hit, 1)
            ex = fb[np.arange(len(fb)), j]
            d[f'pnl_sl{sl}'] = np.where(any_, sell_value(sh, np.nan_to_num(ex, nan=0.0)) - 1, d.pnl)
        hit = np.nan_to_num(fb, nan=0.0) >= 0.99; any_ = hit.any(1); j = np.argmax(hit, 1)   # take-profit at 0.99
        d['pnl_tp99'] = np.where(any_, sell_value(sh, fb[np.arange(len(fb)), j]) - 1, d.pnl)
        rows.append(d[ok])
A = pd.concat(rows, ignore_index=True); A['half'] = np.where(A.start < MID, 1, 2)
A.to_parquet(f'{OUT}/grid_trades.parquet'); print('built', len(A), 'rows in', round(time.time() - t0), 's', flush=True)
