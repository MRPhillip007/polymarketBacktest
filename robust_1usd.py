"""Pick settings by TRAIN-half marginal averages only, evaluate on the test half; exits, coins; $17 deposit with $1 stake."""
import sys, numpy as np, pandas as pd
from alib import *
pd.set_option('display.width', 250); pd.set_option('display.max_columns', None)
R = pd.read_parquet(f'{D}/grid_results.parquet'); A = pd.read_parquet(f'{D}/grid_trades.parquet')
print('Marginal view on TRAIN (mean train ROI %, over configs):')
pick = {}
for k in ['tau', 'price', 'veto', 'streak', 'mom', 'spread', 'hours']:
    g = R.groupby(k).train.mean() * 100; pick[k] = g.idxmax()
    print(f'{k:7s}', ' | '.join(f'{i}: {v:+.2f}' for i, v in g.items()), '  -> pick', pick[k])
def mask(d, p):
    lo, hi = map(float, p['price'].split('-'))
    m = (d.tau == p['tau']) & (d.px >= lo) & (d.px < hi)
    if p['veto'] != 'None': m &= d.model_edge > float(p['veto'])
    if p['streak'] == 'full': m &= d.streak
    if p['mom'] == '>0': m &= d.bn_mom > 0
    if p['spread'] == '<=2': m &= d.spread <= 2
    if p['hours'] == '12-20': m &= (d.hour >= 12) & (d.hour < 20)
    return m
base = dict(tau=300, price='0.5-1.0', veto='None', streak='any', mom='any', spread='any', hours='all')
def rep(name, d, col='pnl'):
    out = {'config': name}
    for h, lab in ((1, 'train'), (2, 'test')):
        x = d[d.half == h]; mu, t, n = cl_t(x[col], x.start)
        out.update({f'{lab}_n': n, f'{lab}_win%': x.win.mean() * 100, f'{lab}_roi%': mu * 100, f'{lab}_t': t})
    return out
print('\nchosen on train:', pick)
B = A[mask(A, base)]; C = A[mask(A, pick)]
print('\n== baseline vs chosen config (ROI per $1 trade) ==')
print(pd.DataFrame([rep('baseline T-5min', B), rep('chosen on train', C)]).round(2).to_string(index=False))
print('\n== exits (same entries) ==')
rows = []
for name, d in (('baseline', B), ('chosen', C)):
    for col in ['pnl', 'pnl_sl0.3', 'pnl_sl0.5', 'pnl_sl0.7', 'pnl_tp99']:
        r = rep(f'{name} {col.replace("pnl_", "") if col != "pnl" else "hold"}', d, col); rows.append(r)
print(pd.DataFrame(rows).round(2)[['config', 'train_roi%', 'train_t', 'test_roi%', 'test_t']].to_string(index=False))
print('\n== chosen config by coin ==')
print(pd.DataFrame([rep(c, C[C.coin == c]) for c in COINS]).round(2).to_string(index=False))

def bankroll(d, dep=17.0, stake=1.0, col='pnl'):
    bal = peak = dep; mdd = 0; low = dep; n = 0
    for s, g in d.sort_values('start').groupby('start'):
        k = int(min(len(g), bal // stake))
        if k == 0: return dict(final=bal, profit=bal - dep, min_bal=low, max_dd=mdd, trades=n, bust='yes')
        bal += g[col].iloc[:k].sum() * stake; n += k; peak = max(peak, bal); mdd = max(mdd, peak - bal); low = min(low, bal)
    return dict(final=bal, profit=bal - dep, min_bal=low, max_dd=mdd, trades=n, bust='no')
print('\n== $17 deposit, $1 per trade ==')
rows = []
for name, d in (('baseline, all 4 coins', B), ('baseline, BTC only', B[B.coin == 'btc']), ('baseline, ETH only', B[B.coin == 'eth']),
                ('chosen, all 4 coins', C), ('chosen, BTC only', C[C.coin == 'btc'])):
    for per, x in (('full 18.08-25.09', d), ('test 05.09-25.09', d[d.half == 2])):
        rows.append(dict(strategy=name, period=per, **bankroll(x)))
print(pd.DataFrame(rows).round(2).to_string(index=False))
# Monte Carlo: resample whole days (keeps within-day clustering), 30-day paths, chosen config & baseline
rng = np.random.default_rng(7)
def mc(d, days=30, n=4000, dep=17.0):
    d = d.assign(day=pd.to_datetime(d.start, unit='ms').dt.date)
    daily = [g.sort_values('start').pnl.values for _, g in d.groupby('day')]
    fin = []; bust = 0
    for _ in range(n):
        bal = dep; dead = False
        for i in rng.integers(0, len(daily), days):
            for p in daily[i]:
                if bal < 1: dead = True; break
                bal += p
            if dead: break
        bust += dead; fin.append(bal)
    fin = np.array(fin)
    return dict(p_bust=bust / n, median_final=np.median(fin), p_profit=(fin > dep).mean(), p5=np.percentile(fin, 5), p95=np.percentile(fin, 95))
print('\n== Monte Carlo 30 days, $17 / $1, days resampled from the TEST half ==')
print(pd.DataFrame([dict(strategy='baseline all 4', **mc(B[B.half == 2])), dict(strategy='chosen all 4', **mc(C[C.half == 2])),
                    dict(strategy='chosen BTC', **mc(C[(C.half == 2) & (C.coin == 'btc')]))]).round(2).to_string(index=False))
