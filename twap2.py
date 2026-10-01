"""TWAP model v2 for Polymarket Up/Down (Chainlink TWAP-60s rule).
P(Up) = T_nu( d / sqrt(k * var_hat * h + sb^2) )
  d       = ln(E[final TWAP | now] / start TWAP)      (Binance 1s prices, realized part of the final minute included)
  h       = V / 3600, V = sum_ij min(d_i, d_j) over the not-yet-known seconds of the final minute (Brownian horizon)
  var_hat = HAR forecast of per-second variance from realized variance over 1m / 5m / 30m / 4h / 1d
  sb      = Binance-vs-Chainlink noise, nu = tail thickness (Student-t), k = variance scale
Also a 'blend': logistic of logit(market mid) and logit(model), weights depending on time left.
All parameters are fitted on the first half (before 2026-09-05) only.
usage: python twap2.py            -> fit, evaluate, write data/twap2_params.json and data/fair2_<tf>_<coin>.npy"""
import json, numpy as np, pandas as pd
from scipy import stats, optimize
from alib import *
from favlib import MID_TS
from a_incidents import in_inc
pd.set_option('display.width', 250); pd.set_option('display.max_columns', None)
SYM = {'btc': 'BTCUSDT', 'eth': 'ETHUSDT', 'sol': 'SOLUSDT', 'xrp': 'XRPUSDT'}
SETS = {'15m': ['btc', 'eth', 'sol', 'xrp'], '5m': ['btc', 'eth', 'sol']}
RV = ['rv1_60', 'rv1_300', 'rv5_300', 'rv5_1800', 'rv5_14400', 'rv5_86400']

def bn_arrays(sym):
    b = pd.read_parquet(f'{D}/bn1s/{sym}.parquet'); c = b.c.values.astype(float); lc = np.log(c)
    r1 = np.r_[0.0, np.diff(lc)]; r5 = np.r_[np.zeros(5), lc[5:] - lc[:-5]]
    return dict(t0=int(b.t.values[0] // 1000), c=c, cs=np.r_[0.0, np.cumsum(c)], c1=np.r_[0.0, np.cumsum(r1 ** 2)], c5=np.r_[0.0, np.cumsum(r5 ** 2)])

def state(B, s, K, k):
    """model inputs at second k of windows starting at s (arrays over windows)"""
    ix = lambda sec: (sec - B['t0']).astype(np.int64)
    t = s + k; e = s + K; n = len(B['c'])
    ok = (ix(s - 86400 - 10) >= 0) & (ix(e) < n)
    cl = lambda j: np.clip(j, 0, n)
    P0 = (B['cs'][cl(ix(s - 1))] - B['cs'][cl(ix(s - 61))]) / 60
    S = B['c'][np.clip(ix(t - 1), 0, n - 1)]
    last = np.minimum(t - 1, e - 2); r = np.clip(last - (e - 61) + 1, 0, 60)
    sum_real = np.where(r > 0, B['cs'][cl(ix(last) + 1)] - B['cs'][cl(ix(e - 61))], 0.0)
    nf = 60 - r; EF = (sum_real + nf * S) / 60
    a = np.where(r > 0, 1, np.maximum((e - 61) - (t - 1), 1))
    V = nf ** 2 * a + (nf - 1) * nf * (2 * nf - 1) / 6
    j = cl(ix(t))                                               # sums over bars opened up to t-1
    rv = {'rv1_60': (B['c1'][j] - B['c1'][cl(j - 60)]) / 60, 'rv1_300': (B['c1'][j] - B['c1'][cl(j - 300)]) / 300}
    for W in (300, 1800, 14400, 86400): rv[f'rv5_{W}'] = (B['c5'][j] - B['c5'][cl(j - W)]) / W / 5
    Fbn = (B['cs'][cl(ix(e - 1))] - B['cs'][cl(ix(e - 61))]) / 60
    return ok, np.log(EF / P0), V / 3600.0, rv, np.log(Fbn / EF)

def samples(coin, tf, step):
    P = load(coin, tf); K = P['bid'].shape[1]; B = bn_arrays(SYM[coin]); s = P['start'] // 1000
    ks = sorted(set(list(range(step, K - 5, step)) + [K - 50, K - 40, K - 30, K - 20, K - 10, K - 5]))
    inc = in_inc(P['start'], K * 1000); y = P['y']; out = []
    for k in ks:
        ok, d, h, rv, eps = state(B, s, K, k)
        mid = (P['bid'][:, k].astype(float) + P['ask'][:, k].astype(float)) / 2
        m = ok & ~inc & np.isin(y, [0, 1]) & np.isfinite(d)
        out.append(pd.DataFrame(dict(coin=coin, tf=tf, start=P['start'][m], k=k, tau=K - k, y=y[m], d=d[m], h=h[m], eps=eps[m],
                                     mid=mid[m], fair_old=P['fair'][m, k].astype(float), **{c: v[m] for c, v in rv.items()})))
    return pd.concat(out, ignore_index=True)

def var_hat(df, w): return np.maximum(df[RV].values @ w, 1e-14)

def fit(tr):
    # 1) HAR weights by QLIKE on realized horizon variance of the Binance final TWAP (eps^2 / h)
    x = tr[tr.h > 0].sample(min(600_000, (tr.h > 0).sum()), random_state=1)
    q = x.eps.values ** 2 / x.h.values; X = x[RV].values
    f = lambda w: np.mean(q / np.maximum(X @ w, 1e-14) + np.log(np.maximum(X @ w, 1e-14)))
    w = optimize.minimize(f, np.full(len(RV), 1 / len(RV)), bounds=[(0, None)] * len(RV), method='L-BFGS-B').x
    # 2) link: Student-t with variance scale k and Chainlink noise sb, by Bernoulli likelihood on outcomes
    vh = var_hat(tr, w); d = tr.d.values; h = tr.h.values; y = tr.y.values
    def nll(p):
        k, sb, nu = np.exp(p[0]), np.exp(p[1]), 2 + np.exp(p[2])
        P = np.clip(stats.t.cdf(d / np.sqrt(k * vh * h + sb ** 2), nu), 1e-6, 1 - 1e-6)
        return -np.mean(y * np.log(P) + (1 - y) * np.log(1 - P))
    p = optimize.minimize(nll, [0.0, np.log(0.5e-4), np.log(4.0)], method='Nelder-Mead', options=dict(maxiter=600, xatol=1e-4, fatol=1e-7)).x
    par = dict(w=w.tolist(), k=float(np.exp(p[0])), sb=float(np.exp(p[1])), nu=float(2 + np.exp(p[2])))
    # 3) blend with the market: logistic on logit(mid), logit(model) and their interaction with ln(time left)
    tr = tr.assign(p_new=predict(tr, par))
    Xb = blend_X(tr); m = np.isfinite(Xb).all(1)
    bb, _ = logit_fit(Xb[m], tr.y.values[m], tr.start.values[m])
    par['blend'] = bb.tolist()
    return par

def predict(df, par):
    vh = var_hat(df, np.array(par['w']))
    return stats.t.cdf(df.d.values / np.sqrt(par['k'] * vh * df.h.values + par['sb'] ** 2), par['nu'])

def lg(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))
def blend_X(df):
    lt = np.log(df.tau.values.astype(float)); a, b = lg(df.mid.values), lg(df.p_new.values)
    return np.column_stack([np.ones(len(df)), a, b, a * lt, b * lt])
def predict_blend(df, par): return 1 / (1 + np.exp(-(blend_X(df) @ np.array(par['blend']))))

def matrix(coin, tf, par):
    """model v2 probability at every second of every window -> data/fair2_<tf>_<coin>.npy"""
    P = load(coin, tf); K = P['bid'].shape[1]; B = bn_arrays(SYM[coin]); s = P['start'] // 1000
    M = np.full((len(s), K), np.nan, np.float32); w = np.array(par['w'])
    for k in range(K):
        ok, d, h, rv, _ = state(B, s, K, k)
        vh = np.maximum(np.column_stack([rv[c] for c in RV]) @ w, 1e-14)
        p = stats.t.cdf(d / np.sqrt(par['k'] * vh * h + par['sb'] ** 2), par['nu'])
        M[:, k] = np.where(ok & np.isfinite(d), p, np.nan)
    np.save(f'{D}/fair2_{tf}_{coin}.npy', M)

def scores(df, cols):
    y = df.y.values; out = {}
    for c in cols:
        p = np.clip(df[c].values, 1e-4, 1 - 1e-4)
        out[c + '_brier'] = np.mean((p - y) ** 2); out[c + '_logloss'] = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return pd.Series(out)

if __name__ == '__main__':
    S_ = pd.concat([samples(c, tf, 15 if tf == '15m' else 10) for tf, cs in SETS.items() for c in cs], ignore_index=True)
    S_ = S_[np.isfinite(S_[RV]).all(1) & np.isfinite(S_.eps)]
    tr = S_[S_.start < MID_TS]; te = S_[S_.start >= MID_TS]
    print('samples: train', len(tr), 'test', len(te))
    par = fit(tr)
    print('fitted on the first half:', json.dumps({k: (np.round(v, 6).tolist() if isinstance(v, list) else round(v, 6)) for k, v in par.items()}))
    json.dump(par, open(f'{D}/twap2_params.json', 'w'), indent=1)
    for d_ in (tr, te):
        d_['p_new'] = predict(d_, par); d_['p_blend'] = predict_blend(d_, par)
    te = te[np.isfinite(te.mid) & np.isfinite(te.fair_old)]
    te['bucket'] = pd.cut(te.tau, [0, 10, 30, 60, 120, 300, 600, 900], right=True)
    cols = ['mid', 'fair_old', 'p_new', 'p_blend']
    print('\n== TEST half (05.09-25.09): Brier / log-loss, lower = better ==')
    tab = te.groupby(['tf', 'bucket'], observed=True).apply(lambda d: scores(d, cols))
    show = pd.DataFrame({'n': te.groupby(['tf', 'bucket'], observed=True).size()})
    for c in cols: show[c] = tab[c + '_brier']
    for c in ['fair_old', 'p_new', 'p_blend']: show[f'{c} vs mid %'] = (1 - tab[c + '_brier'] / tab['mid_brier']) * 100
    print(show.round(5).to_string())
    print('\nby coin (test half, all times):')
    t2 = te.groupby(['tf', 'coin']).apply(lambda d: scores(d, cols))
    print(pd.DataFrame({c: (1 - t2[c + '_brier'] / t2['mid_brier']) * 100 for c in ['fair_old', 'p_new', 'p_blend']}).round(1).to_string())
    # calibration of the new model on the test half
    te['pb'] = pd.cut(te.p_new, [0, .02, .1, .3, .5, .7, .9, .98, 1])
    print('\ncalibration of p_new on the test half (mean predicted vs realized):')
    print(te.groupby('pb', observed=True).agg(n=('y', 'size'), pred=('p_new', 'mean'), real=('y', 'mean')).round(3).T.to_string())
    for tf, cs in SETS.items():
        for c in cs: matrix(c, tf, par)
    print('\nsaved data/twap2_params.json and data/fair2_<tf>_<coin>.npy')
