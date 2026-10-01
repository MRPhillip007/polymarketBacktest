"""How well Binance spot 1s reproduces Chainlink priceToBeat/finalPrice (from Gamma), for several averaging variants.
Result: mean close of bars opened in [t-61, t-2] matches best (change error ~0.3 bp median, outcome agrees 98.9-99.4%),
spot close at t agrees only 93-95% -> confirms the TWAP-60s rule."""
import pandas as pd, numpy as np
from alib import D
pd.set_option('display.width', 250)
mt = pd.read_parquet(f'{D}/meta_15m.parquet').dropna(subset=['ptb', 'final'])
mt = mt[mt.start < pd.Timestamp('2026-09-24T23:30Z')]
res = []
for coin, sym in [('btc', 'BTCUSDT'), ('eth', 'ETHUSDT'), ('sol', 'SOLUSDT'), ('xrp', 'XRPUSDT')]:
    b = pd.read_parquet(f'{D}/bn1s/{sym}.parquet'); c = b.set_index('t').c
    assert (np.diff(c.index.values) == 1000).all()
    cs = np.r_[0, np.cumsum(c.values)]; t0 = c.index.values[0]
    def twap(ts_ms, a, bb):   # mean close of bars with open time in [ts+a, ts+bb) seconds
        i = (ts_ms + a * 1000 - t0) // 1000; j = (ts_ms + bb * 1000 - t0) // 1000
        return (cs[j] - cs[i]) / (j - i)
    m = mt[mt.coin == coin]
    s = m.start.dt.tz_convert(None).values.astype('datetime64[ms]').astype('int64'); e = s + 900_000
    for name, (a, bb) in {'twap[-60,0)': (-60, 0), 'twap[-61,-1)': (-61, -1), 'twap[-59,1)': (-59, 1),
                          'twap[-30,0)': (-30, 0), 'spot close@t': (-1, 0)}.items():
        ep = twap(s, a, bb); ef = twap(e, a, bb)
        dchg = ((ef / ep) - (m.final.values / m.ptb.values)) * 1e4
        res.append(dict(coin=coin, variant=name, level_bias_bp=np.median((ep / m.ptb.values - 1) * 1e4),
                        chg_err_med_bp=np.median(np.abs(dchg)), chg_err_p95_bp=np.percentile(np.abs(dchg), 95),
                        outcome_agree=((ef >= ep) == (m.final.values >= m.ptb.values)).mean(), n=len(m)))
print(pd.DataFrame(res).round(3).to_string(index=False))
