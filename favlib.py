"""Per-second features for 'buy the favourite' strategies ($1 market order, hold to resolution).
Everything used in a condition is known at the decision second k; the order fills at the favourite's ask at k+1."""
import numpy as np, pandas as pd
from alib import *
from a_incidents import in_inc

K = 900
MID_TS = np.datetime64('2026-09-05T00:00', 'ms').astype('int64')      # train = before, test = from this date
SYM = {'btc': 'BTCUSDT', 'eth': 'ETHUSDT', 'sol': 'SOLUSDT', 'xrp': 'XRPUSDT'}
# configuration picked on the first half only (see robust_1usd.py)
CHOSEN = dict(price=(0.80, 0.98), veto=0.0, streak=300, spread=2.0, hours=(12, 20))

def shares(px): return 1.0 / (px * (1 + 0.07 * (1 - px)))              # $1 covers price + taker fee 0.07*p*(1-p)

def model_prob(P, coin, tf, model, mid):
    """'old' = panel model, 'new' = twap2 model, 'blend' = twap2 blended with the market mid (params in twap2_params.json)"""
    if model == 'old': return P['fair'].astype(float)
    new = np.load(f'{D}/fair2_{tf}_{coin}.npy').astype(float)
    if model == 'new': return new
    import json
    bb = np.array(json.load(open(f'{D}/twap2_params.json'))['blend'])
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    lt = np.log(np.arange(mid.shape[1], 0, -1, dtype=float))[None, :]                     # ln(seconds left)
    am, an = lg(mid), lg(new)
    return 1 / (1 + np.exp(-(bb[0] + bb[1] * am + bb[2] * an + bb[3] * am * lt + bb[4] * an * lt)))

def features(coin, streak_len=300, tf='15m', model='old'):
    P = load(coin, tf); b = P['bid'].astype(float); a = P['ask'].astype(float); mid = (a + b) / 2
    Kp = b.shape[1]                                                     # 900 for 15m windows, 300 for 5m
    up = mid > 0.5; dn = mid < 0.5
    pxd = np.where(up, a, 1 - b)                                        # favourite's ask at the decision second
    pxe = np.full_like(pxd, np.nan)
    pxe[:, :-1] = np.where(up[:, :-1], a[:, 1:], 1 - b[:, 1:])          # same side's ask one second later (fill price)
    fair = model_prob(P, coin, tf, model, mid)
    medge = (np.where(up, fair, 1 - fair) - pxd - fee(pxd)) * 100      # Binance-TWAP model minus price paid, c/contract
    k = np.arange(Kp); L = streak_len
    cu = np.concatenate([np.zeros((len(mid), 1)), np.cumsum(up, 1)], 1)
    cd = np.concatenate([np.zeros((len(mid), 1)), np.cumsum(dn, 1)], 1)
    lo = np.clip(k - L, 0, None)
    streak = (k >= L) & np.where(up, (cu[:, k] - cu[:, lo]) == L, (cd[:, k] - cd[:, lo]) == L)
    return dict(P=P, coin=coin, up=up, pxd=pxd, pxe=pxe, medge=medge, streak=streak, spread=(a - b) * 100,
                hour=pd.to_datetime(P['start'], unit='ms').hour.values, inc=in_inc(P['start'], Kp * 1000), y=P['y'], start=P['start'], mid=mid, K=Kp)

def conditions(F, cfg=CHOSEN, use_hours=True):
    """boolean [N, K]: all entry conditions hold at second k"""
    lo, hi = cfg['price']
    c = (F['pxd'] >= lo) & (F['pxd'] < hi) & np.isfinite(F['pxe']) & (F['spread'] <= cfg['spread']) & (F['mid'] != 0.5)
    if cfg.get('veto') is not None: c &= F['medge'] > cfg['veto']
    if cfg.get('streak'): c &= F['streak']
    c &= ~F['inc'][:, None]
    if use_hours and cfg.get('hours'):
        h0, h1 = cfg['hours']; c &= ((F['hour'] >= h0) & (F['hour'] < h1))[:, None]
    return c

def take(F, rows, ks):
    """trades for window rows at seconds ks: $1 each"""
    rows = np.asarray(rows); ks = np.asarray(ks)
    upk = F['up'][rows, ks]; px = F['pxe'][rows, ks]; y = F['y'][rows]
    win = np.where(upk, y, 1 - y)
    return pd.DataFrame(dict(coin=F['coin'], row=rows, start=F['start'][rows], k=ks, side=np.where(upk, 'Up', 'Down'),
                             px=px, win=win, pnl=shares(px) * win - 1))

def first_entry(F, cond, kmin, kmax):
    """enter at the first second in [kmin, kmax) where cond holds; one trade per window"""
    c = cond[:, kmin:kmax]; has = c.any(1); j = np.argmax(c, 1)
    rows = np.where(has)[0]
    return take(F, rows, kmin + j[rows])

def fixed_entry(F, cond, k):
    rows = np.where(cond[:, k])[0]
    return take(F, rows, np.full(len(rows), k))

def summary(T, label=''):
    out = {'variant': label}
    for h, lab in ((T.start < MID_TS, 'train'), (T.start >= MID_TS, 'test')):
        x = T[h]; mu, t, n = cl_t(x.pnl, x.start) if len(x) > 2 else (np.nan, np.nan, len(x))
        days = 17.75 if lab == 'train' else 20.45
        out.update({f'{lab}_n': n, f'{lab}_per_day': n / days, f'{lab}_win%': x.win.mean() * 100 if len(x) else np.nan,
                    f'{lab}_roi%': mu * 100, f'{lab}_t': t})
    return out
