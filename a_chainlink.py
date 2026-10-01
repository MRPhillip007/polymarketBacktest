"""Chainlink instead of Binance. Exact Chainlink TWAP-60s is known every 5 minutes (priceToBeat of the 5m markets),
i.e. 10 and 5 minutes before the end of every 15m window. Compare fair-probability models at those two moments:
  bn      : Binance spot vs Binance proxy of the start price (the model used so far, from the panels)
  cl      : Chainlink TWAP at t vs exact Chainlink start price
  cl+bn   : Chainlink TWAP at t moved by Binance's own spot-vs-TWAP gap (what a live Chainlink feed + Binance gives)
and use each as the veto in the favourite strategy."""
import numpy as np, pandas as pd
from scipy.stats import norm
from alib import *
from favlib import *
pd.set_option('display.width', 250); pd.set_option('display.max_columns', None)

m5 = pd.read_parquet(f'{D}/meta_5m.parquet').dropna(subset=['ptb'])
CL = pd.Series(m5.ptb.values.astype(float), index=pd.MultiIndex.from_arrays(
    [m5.coin.values, m5.start.dt.tz_convert(None).values.astype('datetime64[ms]').astype('int64')]))

def brier(p, y): ok = np.isfinite(p); return np.mean((p[ok] - y[ok]) ** 2)
rows = []; strat = []; drift = []
for coin in COINS:
    F = features(coin); P = F['P']; y = F['y']; st = P['start']
    b = pd.read_parquet(f'{D}/bn1s/{SYM[coin]}.parquet'); t0 = b.t.values[0] // 1000; cs = np.r_[0, np.cumsum(b.c.values)]
    ix = lambda sec: (sec - t0).astype(np.int64)
    cond = conditions(F)
    for k in (300, 600):
        tau = K - k; t_ms = st + k * 1000; t_s = t_ms // 1000
        Ct = CL.reindex(pd.MultiIndex.from_arrays([np.full(len(st), coin), t_ms])).values
        Cs = P['ptb']; S = P['S'][:, k]; sig = P['sig'][:, k].astype(float)
        inr = (ix(t_s - 61) >= 0) & (ix(t_s - 1) < len(cs))                    # windows covered by the Binance file
        j1 = np.clip(ix(t_s - 1), 0, len(cs) - 1); j0 = np.clip(ix(t_s - 61), 0, len(cs) - 1)
        twap_bn = np.where(inr, (cs[j1] - cs[j0]) / 60, np.nan)                  # Binance mean of closes, bars opened t-61..t-2
        f_bn = P['fair'][:, k].astype(float)
        f_cl = norm.cdf(np.log(Ct / Cs) / (sig * np.sqrt(tau - 20)))             # TWAP lags spot ~30 s -> extra sigma^2*20
        f_clbn = norm.cdf(np.log(Ct * S / twap_bn / Cs) / (sig * np.sqrt(tau - 40)))
        mid = F['mid'][:, k]; ok = np.isfinite(Ct) & np.isfinite(mid) & ~F['inc']
        for name, f in (('market mid', mid), ('bn (Binance)', f_bn), ('cl (Chainlink)', f_cl), ('cl+bn', f_clbn)):
            for h, lab in ((st < MID_TS, 'train'), (st >= MID_TS, 'test')):
                m = ok & h; rows.append(dict(coin=coin, min_before=tau // 60, model=name, half=lab, brier=brier(f[m], y[m]), n=m.sum()))
        # how much does the 'distance to the start price' differ between Chainlink and Binance? (bp)
        drift.append(pd.DataFrame({'coin': coin, 'tau': tau, 'd_bp': (np.log(Ct / Cs) - np.log(twap_bn / P['P0'])) * 1e4}))
        # favourite strategy at k with the conditions, veto from each model
        base = cond[:, k].copy()
        up = F['up'][:, k]; pxd = F['pxd'][:, k]
        for name, f in (('bn', f_bn), ('cl+bn', f_clbn)):
            veto = (np.where(up, f, 1 - f) - pxd - fee(pxd)) * 100 > 0
            # conditions() already contains the Binance veto; rebuild without it, then apply this model's veto
            c = (F['pxd'][:, k] >= 0.80) & (F['pxd'][:, k] < 0.98) & np.isfinite(F['pxe'][:, k]) & (F['spread'][:, k] <= 2) \
                & F['streak'][:, k] & ~F['inc'] & np.isfinite(Ct) & veto
            T = take(F, np.where(c)[0], np.full(c.sum(), k)); T['veto'] = name; T['tau'] = tau; strat.append(T)
R = pd.DataFrame(rows)
print('Brier score (lower = better), all 4 coins pooled:')
print(R.groupby(['min_before', 'half', 'model']).apply(lambda d: np.average(d.brier, weights=d.n)).unstack('model').round(5).to_string())
Dr = pd.concat(drift).dropna()
print('\n|Chainlink distance - Binance distance| to the start price, bp:',
      Dr.groupby('tau').d_bp.apply(lambda x: x.abs().quantile([.5, .9, .99]).round(2).to_dict()).to_dict())
S_ = pd.concat(strat)
print('\nFavourite at 10 / 5 min before close, conditions + veto from each model (all hours):')
print(pd.DataFrame([summary(d, f'{int(tau) // 60} min, veto={v}') for (tau, v), d in S_.groupby(['tau', 'veto'])]).round(2).to_string(index=False))
