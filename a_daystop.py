"""Daily loss limit / losing-streak stop on top of the favourite strategies ($1 per trade, $17 deposit).
Positions of a window resolve at its end, before the next window's entries, so the day's P&L is updated per window."""
import numpy as np, pandas as pd
from alib import *
from favlib import *
pd.set_option('display.width', 250); pd.set_option('display.max_columns', None)

def apply_stop(T, day_limit=None, streak_stop=None, dep=17.0):
    """returns kept trades and bankroll stats; day = UTC day of the window start"""
    T = T.sort_values(['start', 'coin']); keep = []; bal = peak = dep; mdd = 0
    day = None; dpnl = 0; lstreak = 0; stopped = False
    for s, g in T.groupby('start', sort=True):
        d = pd.Timestamp(s, unit='ms').date()
        if d != day: day, dpnl, lstreak, stopped = d, 0.0, 0, False
        if stopped: continue
        n = int(min(len(g), bal // 1.0))
        if n == 0: break
        g = g.iloc[:n]; keep.append(g)
        p = g.pnl.sum(); bal += p; dpnl += p; peak = max(peak, bal); mdd = max(mdd, peak - bal)
        lstreak = lstreak + 1 if p < 0 else 0
        if (day_limit is not None and dpnl <= -day_limit) or (streak_stop is not None and lstreak >= streak_stop): stopped = True
    K_ = pd.concat(keep) if keep else T.iloc[:0]
    return K_, dict(final=bal, max_dd=mdd)

if __name__ == '__main__':
    FS = {c: features(c) for c in COINS}
    base = []; trig = []; trig_h = []
    for c, F in FS.items():
        k = 600; m = np.isfinite(F['pxe'][:, k]) & (F['mid'][:, k] != 0.5) & (F['spread'][:, k] <= 10) & ~F['inc']
        base.append(take(F, np.where(m)[0], np.full(m.sum(), k)))
        trig.append(first_entry(F, conditions(F, use_hours=False), K - 480, K - 180))
        trig_h.append(first_entry(F, conditions(F, use_hours=True), K - 360, K - 180))
    strategies = {'fixed 5 min, no filters': pd.concat(base), 'first 8->3 min + conditions': pd.concat(trig),
                  'first 6->3 min + conditions, 12-20 UTC': pd.concat(trig_h)}
    rules = [('no stop', None, None), ('day limit -$1', 1, None), ('day limit -$2', 2, None), ('day limit -$3', 3, None),
             ('day limit -$5', 5, None), ('stop after 2 losing windows', None, 2), ('stop after 3 losing windows', None, 3)]
    for name, T in strategies.items():
        rows = []
        for lab, dl, ss in rules:
            out = {'rule': lab}
            for h, hl in ((T.start < MID_TS, 'train'), (T.start >= MID_TS, 'test')):
                Kt, bk = apply_stop(T[h], dl, ss)
                mu, t, n = cl_t(Kt.pnl, Kt.start) if len(Kt) > 2 else (np.nan, np.nan, len(Kt))
                out.update({f'{hl}_n': n, f'{hl}_roi%': mu * 100, f'{hl}_t': t, f'{hl}_pnl$': Kt.pnl.sum(), f'{hl}_final$': bk['final'], f'{hl}_maxDD$': bk['max_dd']})
            rows.append(out)
        print(f'\n== {name} ==  ($1 per trade, deposit $17 restarted at the start of each half)')
        print(pd.DataFrame(rows).round(2).to_string(index=False))
    # are losses clustered in time? P(losing window | previous window lost) vs overall
    T = strategies['fixed 5 min, no filters'].sort_values('start')
    w = T.groupby('start').pnl.sum(); lose = (w < 0).astype(int)
    print('\nfixed 5 min: P(losing window) = %.3f, P(losing window | previous lost) = %.3f' % (lose.mean(), lose[lose.shift(1) == 1].mean()))
    dd = T.assign(day=pd.to_datetime(T.start, unit='ms').dt.date, h=pd.to_datetime(T.start, unit='ms').dt.hour)
    a = dd[dd.h < 12].groupby('day').pnl.sum(); b = dd[dd.h >= 12].groupby('day').pnl.sum()
    print('corr(P&L 00-12 UTC, P&L 12-24 UTC same day) = %.3f over %d days' % (a.corr(b.reindex(a.index)), len(a)))
