"""C: taker strategy: buy the side where Binance-TWAP fair value beats the Polymarket quote by > theta after fee.
Decision at second k with quote visible at k; order arrives at k+delay: fills only if the price then is <= decision price
(marketable limit), pays the price prevailing at arrival. One entry per window per side; hold to resolution."""
import numpy as np, pandas as pd, itertools
from alib import *
pd.set_option('display.width', 250)
MID = np.datetime64('2026-09-05T00:00', 'ms').astype('int64')
def run(P, theta, delay, kmin, kmax, good):
    y = P['y']; st = P['start']; fair = P['fair'].astype(float); bid = P['bid'].astype(float); ask = P['ask'].astype(float)
    N = len(y); out = []; nsig = 0
    ks = np.arange(kmin, min(kmax, 900 - delay))
    for side in ('up', 'dn'):
        px = ask if side == 'up' else 1 - bid                     # price to buy this side
        pf = fair if side == 'up' else 1 - fair
        edge = pf[:, ks] - px[:, ks] - fee(px[:, ks])
        sig = (edge > theta) & np.isfinite(edge) & good[:, None] & ((ask - bid)[:, ks] <= 0.10)
        has = sig.any(1); first = np.argmax(sig, 1)
        for i in np.where(has)[0]:
            k = ks[first[i]]; lim = px[i, k]; pe = px[i, k + delay]; nsig += 1
            if not np.isfinite(pe) or pe > lim + 1e-9: continue
            yy = y[i] if side == 'up' else 1 - y[i]
            out.append((st[i], side, k, pf[i, k], lim, pe, yy - pe - fee(pe)))
    return pd.DataFrame(out, columns=['start', 'side', 'k', 'fair', 'lim', 'px', 'pnl']), nsig
def good_windows(P):
    return np.isfinite(P['P0']) & (np.abs(P['P0'] / P['ptb'] - 1) < 1e-3)   # start-only sanity check (no look-ahead)

if __name__ == '__main__':
    rows = []
    for coin in COINS:
        P = load(coin)
        good = good_windows(P)
        for theta, delay, (kmin, kmax) in itertools.product([0.03, 0.06, 0.10, 0.15], [1, 2, 5], [(0, 840), (840, 900), (0, 900)]):
            t, nsig = run(P, theta, delay, kmin, kmax, good)
            if len(t) < 10: continue
            mu, tt, n = cl_t(t.pnl, t.start)
            h1 = t[t.start < MID]; h2 = t[t.start >= MID]
            m1, t1, n1 = cl_t(h1.pnl, h1.start); m2, t2, n2 = cl_t(h2.pnl, h2.start)
            rows.append(dict(coin=coin, theta=theta, delay=delay, ks=f'{kmin}-{kmax}', n=n, fill_rate=n / max(1, nsig), px=t.px.mean(),
                             ev_c=mu * 100, t=tt, ev1_c=m1 * 100, t1=t1, ev2_c=m2 * 100, t2=t2, win=(t.pnl > 0).mean()))
    r = pd.DataFrame(rows); r.to_parquet(f'{D}/res_strat.parquet')
    print(r.round(3).to_string(index=False))
