"""The favourite strategy on the 5-minute markets (BTC, ETH, SOL). Rules are the 15m rules frozen on 25.09 with every
time parameter scaled by 1/3 (window 900 s -> 300 s); nothing was re-optimised on 5m data.
Also: the same 15m strategy on the same three coins for comparison, a small time-parameter sensitivity (information only)."""
import numpy as np, pandas as pd
from alib import *
from favlib import features, conditions, first_entry, summary, MID_TS
from a_daystop import apply_stop
pd.set_option('display.width', 250); pd.set_option('display.max_columns', None)
C3 = ['btc', 'eth', 'sol']
BASE = dict(price=(0.80, 0.98), veto=0.0, spread=2.0)
RULES = {  # name: (tf, streak s, entry from s before close, entry until s before close, hours)
    'A15: 15m, 6:00->3:00, 12-20 UTC': ('15m', 300, 360, 180, (12, 20)),
    'B15: 15m, 8:00->3:00, all hours': ('15m', 300, 480, 180, None),
    'A5: 5m, 2:00->1:00, 12-20 UTC': ('5m', 100, 120, 60, (12, 20)),
    'B5: 5m, 2:40->1:00, all hours': ('5m', 100, 160, 60, None)}

def trades(tf, streak, e_from, e_to, hours, coins=C3, FS=None):
    out = []
    for c in coins:
        F = FS[(c, tf, streak)] if FS is not None and (c, tf, streak) in FS else features(c, streak, tf)
        if FS is not None: FS[(c, tf, streak)] = F
        cfg = dict(BASE, streak=streak, hours=hours)
        out.append(first_entry(F, conditions(F, cfg, use_hours=hours is not None), F['K'] - e_from, F['K'] - e_to))
    T = pd.concat(out); return T[T.win.isin([0, 1])]

def bank(T, lim=None, dep=17.0):
    K_, bk = apply_stop(T, lim, None, dep)
    bal = dep; low = dep
    for _, g in K_.sort_values('start').groupby('start'): bal += g.pnl.sum(); low = min(low, bal)
    return dict(final=bk['final'], max_dd=bk['max_dd'], min_balance=low)

if __name__ == '__main__':
    FS = {}
    m5 = pd.read_parquet(f'{D}/meta_5m.parquet'); m15 = pd.read_parquet(f'{D}/meta_15m.parquet')
    print('Gamma volume per window, median $:', {c: (round(m5[m5.coin == c].vol.median()), round(m15[m15.coin == c].vol.median())) for c in C3}, '(5m, 15m)')
    rows = []; bk = []; per_coin = []; TT = {}
    for name, (tf, st, ef, et, hrs) in RULES.items():
        T = trades(tf, st, ef, et, hrs, FS=FS); TT[name] = T
        s = summary(T, name); mu, t, n = cl_t(T.pnl, T.start)
        s.update(all_n=n, all_win=T.win.mean() * 100, avg_px=T.px.mean(), all_roi=mu * 100, all_t=t, pnl=T.pnl.sum())
        rows.append(s)
        days = (T.start.max() - T.start.min()) / 86_400_000
        for lim in (None, 2.0):
            bk.append(dict(variant=name, day_limit='-$2' if lim else 'none', trades_per_day=len(T) / days, **bank(T, lim)))
        for c in C3:
            x = T[T.coin == c]; mu_, t_, n_ = cl_t(x.pnl, x.start)
            per_coin.append(dict(variant=name, coin=c, n=n_, win=x.win.mean() * 100, roi=mu_ * 100, t=t_))
    R = pd.DataFrame(rows)
    print('\n== per trade, $1 each (train = 18.08-04.09, test = 05.09-25.09) ==')
    print(R[['variant', 'all_n', 'all_win', 'avg_px', 'all_roi', 'all_t', 'pnl', 'train_roi%', 'train_t', 'test_roi%', 'test_t', 'test_per_day']].round(2).to_string(index=False))
    print('\n== deposit $17, whole period 18.08-25.09 ==')
    print(pd.DataFrame(bk).round(2).to_string(index=False))
    print('\n== by coin ==')
    print(pd.DataFrame(per_coin).pivot(index='variant', columns='coin', values=['n', 'win', 'roi', 't']).round(2).to_string())
    # sensitivity of the 5m time parameters (information only - not used to pick anything)
    sens = []
    for st in (60, 100, 150):
        for ef, et in ((90, 30), (120, 60), (160, 60), (200, 60), (240, 120)):
            T = trades('5m', st, ef, et, None, FS=FS); s = summary(T, f'streak {st}s, entry {ef}->{et}s')
            sens.append(s)
    print('\n== 5m sensitivity, all hours (information only) ==')
    print(pd.DataFrame(sens)[['variant', 'train_n', 'train_roi%', 'train_t', 'test_n', 'test_roi%', 'test_t']].round(2).to_string(index=False))
    for name, T in TT.items(): T.to_parquet(f'{D}/trades_{name.split(":")[0]}.parquet')
