"""Which factors separate good from bad 'favourite at T-5min' trades? Quintiles of each factor -> realized edge (c/contract),
cluster-t by window start, and both halves (split 2026-09-05). Factors use only information available at entry."""
import sys, numpy as np, pandas as pd
from alib import *
import a_fav5m as F
pd.set_option('display.width', 250); pd.set_option('display.max_columns', None)
K = 600; MID = np.datetime64('2026-09-05T00:00', 'ms').astype('int64')
tr = pd.read_parquet(f'{D}/trades_15m.parquet', columns=['m', 'a', 'coin', 'side', 'usd', 't', 'up'])
rows = []
for coin in COINS:
    P = load(coin); T = F.trades(coin); T = T[~T.inc].copy(); i = T.index.values
    up = (T.side == 'Up').values; sgn = np.where(up, 1, -1)
    fair = P['fair'][i, K].astype(float); T['fair_fav'] = np.where(up, fair, 1 - fair)
    T['model_edge_c'] = (T.fair_fav - T.px - fee(T.px)) * 100                       # Binance-TWAP model vs price paid
    T['sigma_bp'] = P['sig'][i, K].astype(float) * 1e4                                  # realized vol per second (5s returns, 30m+4h)
    S = P['S']; T['bn_mom60_bp'] = sgn * (S[i, K] / S[i, K - 60] - 1) * 1e4              # Binance move last 60 s toward favourite
    T['bn_mom300_bp'] = sgn * (S[i, K] / S[i, K - 300] - 1) * 1e4
    b = P['bid'].astype(float); a = P['ask'].astype(float); mid = (a + b) / 2
    midfav = lambda k: np.where(up, mid[i, k], 1 - mid[i, k])
    T['pm_mom120_c'] = (midfav(K) - midfav(K - 120)) * 100                               # favourite price change last 2 min
    T['spread_c'] = (a[i, K] - b[i, K]) * 100
    # how long has it been the favourite (seconds of the last 300 with favourite mid > 0.5)
    T['fav_streak_s'] = (np.where(up[:, None], mid[i, K - 300:K], 1 - mid[i, K - 300:K]) > 0.5).sum(1)
    # Polymarket taker flow toward the favourite in the 60 s before entry
    m = P['m'][i]; T['m'] = m
    x = tr[(tr.coin == coin) & (tr.t >= K - 60) & (tr.t < K)].copy()
    bull_up = ((x.a == x.up) & (x.side == 'BUY')) | ((x.a != x.up) & (x.side == 'SELL'))
    x['flow_up'] = np.where(bull_up, x.usd, -x.usd)
    f = x.groupby('m').flow_up.sum()
    T['pm_flow60_$'] = sgn * T.m.map(f).fillna(0).values
    T['hour_utc'] = pd.to_datetime(T.start, unit='ms').dt.hour
    rows.append(T)
A = pd.concat(rows, ignore_index=True); A['half'] = np.where(A.start < MID, 1, 2)
A.to_parquet(f'{D}/fav300_features.parquet')
print('trades', len(A), 'baseline edge %.2fc' % A.edge_c.mean())
def qtab(col, q=5):
    A['_q'] = pd.qcut(A[col].rank(method='first'), q, labels=[f'Q{j+1}' for j in range(q)])
    out = []
    for qq, d in A.groupby('_q', observed=True):
        mu, t, n = cl_t(d.edge_c, d.start)
        m1, t1, _ = cl_t(d[d.half == 1].edge_c, d[d.half == 1].start); m2, t2, _ = cl_t(d[d.half == 2].edge_c, d[d.half == 2].start)
        out.append(dict(q=qq, lo=d[col].min(), hi=d[col].max(), n=n, win=d.win.mean() * 100, px=d.px.mean(), edge_c=mu, t=t, h1=m1, t1=t1, h2=m2, t2=t2))
    print(f'\n--- {col} ---'); print(pd.DataFrame(out).round(2).to_string(index=False))
for col in ['model_edge_c', 'sigma_bp', 'bn_mom60_bp', 'bn_mom300_bp', 'pm_mom120_c', 'pm_flow60_$', 'fav_streak_s', 'spread_c']:
    qtab(col)
# hour of day (6 blocks)
A['hblk'] = (A.hour_utc // 4) * 4
out = []
for h, d in A.groupby('hblk'):
    mu, t, n = cl_t(d.edge_c, d.start); out.append(dict(hours_utc=f'{h:02d}-{h+4:02d}', n=n, edge_c=mu, t=t))
print('\n--- hour of day ---'); print(pd.DataFrame(out).round(2).to_string(index=False))
# cost side: upper bound if you could buy at the favourite's BID with no fee (maker, touch = fill)
