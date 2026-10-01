"""A: market calibration by time remaining; EV of buying favorite / underdog at the ask after fees."""
import numpy as np, pandas as pd, sys
from alib import *
pd.set_option('display.width', 250)
rows = []
for coin in COINS:
    P = load(coin); y = P['y']; st = P['start']
    for tau in [870, 840, 720, 600, 480, 360, 300, 240, 180, 120, 90, 60, 45, 30, 20, 10, 5, 3]:
        k = 900 - tau; b = P['bid'][:, k].astype(float); a = P['ask'][:, k].astype(float)
        ok = np.isfinite(b) & np.isfinite(a) & (a > b) & (a - b <= 0.10)
        mid = (a + b) / 2
        up_fav = mid >= 0.5
        fav_ask = np.where(up_fav, a, 1 - b); yf = np.where(up_fav, y, 1 - y); dog_ask = np.where(up_fav, 1 - b, a)
        for lo, hi in [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95), (0.95, 0.98), (0.98, 0.995)]:
            s = ok & (fav_ask >= lo) & (fav_ask < hi)
            if s.sum() < 20: continue
            pnl_f = yf[s] - fav_ask[s] - fee(fav_ask[s])
            pnl_d = (1 - yf[s]) - dog_ask[s] - fee(dog_ask[s])
            mf, tf, n = cl_t(pnl_f, st[s]); md, td, _ = cl_t(pnl_d, st[s])
            rows.append(dict(coin=coin, tau=tau, bucket=f'{lo:.2f}-{hi:.3f}', n=n, fav_ask=fav_ask[s].mean(), win=yf[s].mean(),
                             ev_fav=mf * 100, t_fav=tf, ev_dog=md * 100, t_dog=td))
r = pd.DataFrame(rows)
r.to_parquet(f'{D}/res_calib.parquet')
allc = r.copy()
for tau in [840, 600, 300, 180, 120, 60, 30, 10, 3]:
    x = allc[allc.tau == tau]
    if len(x): print(f'\n== tau {tau}s remaining ==\n', x[['coin', 'bucket', 'n', 'fav_ask', 'win', 'ev_fav', 't_fav', 'ev_dog', 't_dog']].round(3).to_string(index=False))
