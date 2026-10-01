"""Binance aggressive (taker) flow: imbalance = (taker buys - taker sells) / volume over the last W seconds.
1) does it add information about the outcome beyond the Polymarket price and the Binance model?  (logit, both halves)
2) does it predict the Binance move until the end of the window?
3) as a filter for the favourite strategies (fixed 5 min without filters; trigger variants)."""
import numpy as np, pandas as pd
from alib import *
from favlib import *
pd.set_option('display.width', 250); pd.set_option('display.max_columns', None)
WS = (10, 30, 60, 300)

def flow_tools(coin):
    b = pd.read_parquet(f'{D}/bn1s/{SYM[coin]}.parquet'); t0 = b.t.values[0] // 1000
    cv = np.r_[0, np.cumsum(b.v.values.astype(float))]; ctb = np.r_[0, np.cumsum(b.tb.values.astype(float))]
    lc = np.log(b.c.values.astype(float))
    def imb(t_s, W):                                   # bars opened in [t-W, t-1]
        j1 = t_s - t0; j0 = j1 - W; ok = (j0 >= 0) & (j1 < len(cv))
        j1c = np.clip(j1, 0, len(cv) - 1); j0c = np.clip(j0, 0, len(cv) - 1)
        V = cv[j1c] - cv[j0c]; B = ctb[j1c] - ctb[j0c]
        return np.where(ok & (V > 0), (2 * B - V) / np.where(V > 0, V, 1), np.nan)
    def ret(t_s, e_s):                                 # log return from price at t to price at e (closes of bars opened t-1, e-1)
        i = t_s - 1 - t0; j = e_s - 1 - t0; ok = (i >= 0) & (j < len(lc))
        return np.where(ok, lc[np.clip(j, 0, len(lc) - 1)] - lc[np.clip(i, 0, len(lc) - 1)], np.nan)
    return imb, ret

def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))

if __name__ == '__main__':
    reg = []; corr = []; Tfix = []; Ttrig = []
    for coin in COINS:
        F = features(coin); P = F['P']; st = P['start']; imb, ret = flow_tools(coin)
        for k in (420, 600):
            t_s = st // 1000 + k; e_s = st // 1000 + K
            d = pd.DataFrame({'coin': coin, 'start': st, 'k': k, 'y': F['y'], 'mid': F['mid'][:, k], 'fair': P['fair'][:, k].astype(float),
                              'fut_ret': ret(t_s, e_s), 'inc': F['inc']})
            for W in WS: d[f'imb{W}'] = imb(t_s, W)
            reg.append(d)
        # fixed 5 min before close, no filters (big sample)
        k = 600; c = np.isfinite(F['pxe'][:, k]) & (F['mid'][:, k] != 0.5) & (F['spread'][:, k] <= 10) & ~F['inc']
        T = take(F, np.where(c)[0], np.full(c.sum(), k)); Tfix.append(T)
        # trigger: first moment 8->3 min with the chosen conditions, all hours
        T2 = first_entry(F, conditions(F, use_hours=False), K - 480, K - 180); Ttrig.append(T2)
        for T_ in (T, T2):
            sg = np.where(T_.side == 'Up', 1, -1); ts = T_.start.values // 1000 + T_.k.values
            for W in WS: T_[f'ff{W}'] = sg * imb(ts, W)          # flow toward the favourite
    R = pd.concat(reg, ignore_index=True); R = R[~R.inc].dropna()
    print('1) logit  y ~ logit(mid) + logit(model) + imb60 + imb300   (t-stats clustered by window)')
    for k in (420, 600):
        for h, lab in ((R.start < MID_TS, 'train'), (R.start >= MID_TS, 'test')):
            x = R[(R.k == k) & h]
            X = np.column_stack([np.ones(len(x)), logit(x.mid), logit(x.fair), x.imb60, x.imb300])
            bta, tt = logit_fit(X, x.y.values, x.start.values)
            print(f'  {(K - k) // 60} min before close, {lab:5s}: imb60 coef {bta[3]:+.3f} (t {tt[3]:+.2f}), imb300 coef {bta[4]:+.3f} (t {tt[4]:+.2f}), n={len(x)}')
    print('\n2) corr(imbalance over last W s, Binance return until the end of the window):')
    for k in (420, 600):
        x = R[R.k == k]
        print(f'  {(K - k) // 60} min:', ' | '.join(f'W={W}s: {np.corrcoef(x[f"imb{W}"], x.fut_ret)[0, 1]:+.3f}' for W in WS))
    print('\n3) favourite strategies split by flow toward the favourite (last 60 s):')
    for name, T in (('fixed 5 min, no filters', pd.concat(Tfix)), ('first 8->3 min + conditions', pd.concat(Ttrig))):
        T = T.dropna(subset=['ff60'])
        T['q'] = pd.qcut(T.ff60.rank(method='first'), 5, labels=['Q1 sell', 'Q2', 'Q3', 'Q4', 'Q5 buy'])
        print(f'\n  {name}:')
        print(pd.DataFrame([summary(d, f'{q}  [{d.ff60.min():+.2f}..{d.ff60.max():+.2f}]') for q, d in T.groupby('q', observed=True)]
                           + [summary(T[T.ff60 > 0], 'flow toward fav > 0'), summary(T[T.ff60 <= 0], 'flow toward fav <= 0')]).round(2).to_string(index=False))
