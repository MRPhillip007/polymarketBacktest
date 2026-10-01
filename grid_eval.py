"""Grid over entry parameters: select on half 1, evaluate on half 2 (walk-forward). $1 stake, pnl per trade in $ (= ROI)."""
import sys, numpy as np, pandas as pd, itertools
from alib import *
pd.set_option('display.width', 250); pd.set_option('display.max_columns', None)
A = pd.read_parquet(f'{D}/grid_trades.parquet')
PR = [(lo, hi) for lo in (0.5, 0.6, 0.7, 0.8, 0.9) for hi in (0.9, 0.95, 0.98, 1.0) if hi > lo]
VETO = [None, -5.0, -1.8, 0.0]
res = []
for tau, d in A.groupby('tau'):
    pnl = d.pnl.values; tr = (d.half == 1).values; te = ~tr; px = d.px.values
    dims = dict(
        price={f'{lo}-{hi}': (px >= lo) & (px < hi) for lo, hi in PR},
        veto={str(v): (np.ones(len(d), bool) if v is None else d.model_edge.values > v) for v in VETO},
        streak={'any': np.ones(len(d), bool), 'full': d.streak.values},
        mom={'any': np.ones(len(d), bool), '>0': d.bn_mom.values > 0},
        spread={'any': np.ones(len(d), bool), '<=2': d.spread.values <= 2},
        hours={'all': np.ones(len(d), bool), '12-20': (d.hour.values >= 12) & (d.hour.values < 20)})
    keys = list(dims)
    for combo in itertools.product(*[list(dims[k].items()) for k in keys]):
        m = np.logical_and.reduce([c[1] for c in combo])
        a, b = m & tr, m & te
        na, nb = a.sum(), b.sum()
        if na < 150: continue
        res.append(dict(tau=tau, **{k: c[0] for k, c in zip(keys, combo)}, n_train=na, train=pnl[a].mean(), n_test=nb,
                        test=pnl[b].mean() if nb else np.nan))
R = pd.DataFrame(res); R.to_parquet(f'{D}/grid_results.parquet')
print('configs evaluated:', len(R))
print('Spearman corr(train, test) across configs: %.3f' % R[['train', 'test']].corr(method='spearman').iloc[0, 1])
top = R.sort_values('train', ascending=False).head(30)
print('\nTop-30 by TRAIN (half 1): mean train %.2f%%, mean test %.2f%%, share with test>0: %.0f%%' %
      (top.train.mean() * 100, top.test.mean() * 100, (top.test > 0).mean() * 100))
print((top.assign(train=top.train * 100, test=top.test * 100)).head(15).round(2).to_string(index=False))
base = R[(R.price == '0.5-1.0') & (R.veto == 'None') & (R.streak == 'any') & (R['mom'] == 'any') & (R.spread == 'any') & (R.hours == 'all')]
print('\nBaseline (no filters) by entry time, ROI per $1 trade, %:')
print(base[['tau', 'n_train', 'train', 'n_test', 'test']].assign(train=base.train * 100, test=base.test * 100).round(2).to_string(index=False))
# marginal effect of each parameter: average test ROI across configs holding it fixed (robust view, not a selection)
print('\nMarginal view: mean TEST ROI (%) over all configs with that setting')
for k in ['tau', 'price', 'veto', 'streak', 'mom', 'spread', 'hours']:
    g = R.groupby(k).test.mean() * 100
    print(f'{k:7s}', ' | '.join(f'{i}: {v:+.2f}' for i, v in g.items()))
