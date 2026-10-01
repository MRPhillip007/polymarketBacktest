"""Descriptive stats: volume, trades, spreads through the window, quote activity."""
import numpy as np, pandas as pd, pyarrow.parquet as pq, glob
from alib import *
pd.set_option('display.width', 250)
mt = pd.read_parquet(f'{D}/meta_15m.parquet')
# trades
parts = []
for f in sorted(glob.glob(f'{D}/tr/*.parquet')):
    if pq.read_metadata(f).num_rows == 0: continue
    t = pq.read_table(f, columns=['timestamp', 'market', 'asset_id', 'price', 'size', 'side'])
    parts.append(t.to_pandas())
tr = pd.concat(parts, ignore_index=True)
tr['m'] = tr.market.map(bytes.hex); tr['a'] = tr.asset_id.map(bytes.hex)
tr = tr.merge(mt[['m', 'coin', 'start', 'up', 'p_up_final']], on='m')
tr['price'] = tr.price.astype(float); tr['size'] = tr['size'].astype(float); tr['usd'] = tr.price * tr['size']
tr['t'] = (tr.timestamp - tr.start).dt.total_seconds()
tr.drop(columns=['market', 'asset_id']).to_parquet(f'{D}/trades_15m.parquet')
g = tr.groupby('coin')
print('trades:', len(tr))
print(pd.DataFrame({'windows': mt.groupby('coin').size(), 'vol_gamma_med$': mt.groupby('coin').vol.median(), 'vol_gamma_mean$': mt.groupby('coin').vol.mean(),
      'trades/window': g.size() / mt.groupby('coin').size(), 'notional_arch$': g.usd.sum() / mt.groupby('coin').size(), 'trade_med$': g.usd.median(),
      'trade_p90$': g.usd.quantile(.9), 'up_rate': mt.groupby('coin').p_up_final.mean()}).round(2))
print('\ntrade timing (s from window start) quantiles:\n', g.t.quantile([.01, .1, .5, .9, .99]).unstack().round(0))
tr['hour'] = tr.timestamp.dt.hour
print('\nnotional by UTC hour (share, btc):\n', (tr[tr.coin == 'btc'].groupby('hour').usd.sum() / tr[tr.coin == 'btc'].usd.sum()).round(3).to_dict())
# spreads through the window from panels
rows = []
for coin in COINS:
    P = load(coin); b = P['bid'].astype(float); a = P['ask'].astype(float)
    for tau in [890, 840, 600, 300, 120, 60, 30, 10, 3]:
        k = 900 - tau; s = a[:, k] - b[:, k]; mid = (a[:, k] + b[:, k]) / 2
        rows.append(dict(coin=coin, tau=tau, spread_med=np.nanmedian(s), spread_p90=np.nanpercentile(s, 90), has_quote=np.isfinite(s).mean(),
                         mid_extreme=np.nanmean((mid < .05) | (mid > .95)), ev_per_s=np.nanmean(P['nev'][:, k])))
print(pd.DataFrame(rows).pivot(index='tau', columns='coin').round(3).sort_index(ascending=False))
