"""Strategy 2 (a_strat.run) with venue-incident windows separated out (INCIDENTS.json from the archive, +-30 min),
then by week and by halves on normal windows only; plus the check that the last-seconds "knife-edge" anomaly
comes from frozen books during incidents.  out: data/strat_trades_noinc.parquet"""
import numpy as np, pandas as pd, json
from alib import *
from a_strat import run, good_windows, MID
pd.set_option('display.width', 250)

inc = json.load(open(f'{D}/INCIDENTS.json'))['incidents']
IV = [(pd.Timestamp(i['started_utc']).value // 10**6 - 30 * 60_000, pd.Timestamp(i['ended_utc']).value // 10**6 + 30 * 60_000) for i in inc]

def in_inc(st, dur=900_000):
    """window [start, start+dur) overlaps an incident (+-30 min buffer)"""
    m = np.zeros(len(st), bool)
    for a, b in IV: m |= (st + dur > a) & (st < b)
    return m

def summ(d):
    mu, t, n = cl_t(d.pnl, d.start); return pd.Series({'ev_c': mu * 100, 't': t, 'n': n})

if __name__ == '__main__':
    print(len(IV), 'incidents; total hours', round(sum(b - a for a, b in IV) / 3.6e6, 1))
    allt = []
    for coin in COINS:
        P = load(coin); good = good_windows(P)
        print(coin, 'windows in incidents', in_inc(P['start']).sum())
        for theta in (0.06, 0.10, 0.15):
            for kmin, kmax in [(0, 840), (840, 900)]:
                for delay in (1, 2):
                    t, _ = run(P, theta, delay, kmin, kmax, good)
                    t['coin'] = coin; t['theta'] = theta; t['part'] = f'{kmin}-{kmax}'; t['delay'] = delay
                    t['inc'] = in_inc(t.start.values); allt.append(t)
    T = pd.concat(allt); T['week'] = pd.to_datetime(T.start, unit='ms').dt.to_period('W-SUN').astype(str).str[:10]
    print('\n== incident windows vs normal (delay 2) ==')
    print(T[T.delay == 2].groupby(['theta', 'part', 'inc']).apply(summ).round(2).unstack('inc').to_string())
    N = T[~T.inc].copy()
    print('\n== normal windows only, by week (delay 2) ==')
    print(N[N.delay == 2].groupby(['theta', 'part', 'week']).apply(summ).round(2).unstack(['theta', 'part']).to_string())
    N['half'] = np.where(N.start < MID, '1st', '2nd')
    print('\n== normal windows only: halves (split 2026-09-05) ==')
    print(N.groupby(['theta', 'part', 'delay', 'half']).apply(summ).round(2).unstack('half').to_string())
    N.to_parquet(f'{D}/strat_trades_noinc.parquet')

    # last-seconds anomaly: favorite priced 0.50-0.60 three seconds before the end
    print('\n== knife-edge books 3 s before the end ==')
    tr = pd.read_parquet(f'{D}/trades_15m.parquet')
    for coin in COINS:
        P = load(coin); b = P['bid'].astype(float); a = P['ask'].astype(float); y = P['y']; st = P['start']
        k = 897; mid = (a[:, k] + b[:, k]) / 2; fav = np.where(mid >= .5, a[:, k], 1 - b[:, k]); yf = np.where(mid >= .5, y, 1 - y)
        s = (fav >= .5) & (fav < .6) & (a[:, k] - b[:, k] <= .1); im = in_inc(st)
        last = tr[(tr.coin == coin) & tr.m.isin(set(P['m'][s])) & (tr.t >= 885) & (tr.t < 900)]
        print(f"{coin}: windows {s.sum()}, in incidents {(s & im).sum()}, favorite win rate in incidents "
              f"{yf[s & im].mean() if (s & im).any() else np.nan:.2f}, trades in last 15 s: {len(last)}")
