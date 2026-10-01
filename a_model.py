"""B: Binance TWAP-model fair value vs market mid (Brier, logistic), D: lead-lag of mid vs fair."""
import numpy as np, pandas as pd
from alib import *
pd.set_option('display.width', 250)
def logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
rows = []; ll = []
for coin in COINS:
    P = load(coin); y = P['y']; st = P['start']; fair = P['fair'].astype(float)
    bid = P['bid'].astype(float); ask = P['ask'].astype(float); mid = (bid + ask) / 2
    good = np.isfinite(P['P0']) & (np.abs(P['P0'] / P['ptb'] - 1) < 1e-3)   # start-only sanity check (no look-ahead)
    # model sanity: fair at last second vs outcome
    for tau in [840, 600, 300, 180, 120, 60, 30, 10, 3]:
        k = 900 - tau
        ok = good & np.isfinite(mid[:, k]) & np.isfinite(fair[:, k]) & (ask[:, k] - bid[:, k] <= 0.10)
        f = fair[ok, k]; m = mid[ok, k]; yy = y[ok]
        X = np.column_stack([np.ones(ok.sum()), logit(m), logit(f)])
        try:
            bb, tt = logit_fit(X, yy, st[ok]); bm, bf, tm, tf = bb[1], bb[2], tt[1], tt[2]
        except Exception: bm = bf = tm = tf = np.nan
        rows.append(dict(coin=coin, tau=tau, n=ok.sum(), brier_mid=np.mean((m - yy) ** 2), brier_fair=np.mean((f - yy) ** 2),
                         brier_blend=np.mean(((m + f) / 2 - yy) ** 2), b_mid=bm, t_mid=tm, b_fair=bf, t_fair=tf,
                         mean_abs_gap=np.mean(np.abs(f - m))))
    # lead-lag: corr(d mid[k->k+1], d fair[k-L -> k-L+1]) over k in [60, 840]
    dm = np.diff(mid, axis=1); df = np.diff(fair, axis=1)
    sel = good & np.isfinite(mid).all(1) & np.isfinite(fair).all(1)
    for L in range(-3, 11):
        ks = np.arange(60, 840)
        a = dm[sel][:, ks].ravel(); b = df[sel][:, ks - L].ravel()
        ll.append(dict(coin=coin, lag=L, corr=np.corrcoef(a, b)[0, 1]))
r = pd.DataFrame(rows); print(r.round(4).to_string(index=False))
l = pd.DataFrame(ll).pivot(index='lag', columns='coin', values='corr'); print('\ncorr( dMid[t,t+1], dFair[t-L,t-L+1] ):\n', l.round(3))
