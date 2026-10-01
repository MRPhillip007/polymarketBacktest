"""Build 1-second panels per coin: Up-token best bid/ask at each second of every window,
plus Binance-based fair probability under the Chainlink TWAP-60s rule.
usage: python panel.py btc,eth,sol,xrp [15m|5m]
out: data/panel_<coin>.npz (15m) or data/panel5m_<coin>.npz (5m, quotes from data/bbo5m)
"""
import pandas as pd, numpy as np, pyarrow.parquet as pq, glob, os, sys
from scipy.stats import norm
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
SYM = {'btc': 'BTCUSDT', 'eth': 'ETHUSDT', 'sol': 'SOLUSDT', 'xrp': 'XRPUSDT'}
K = 900
BASIS = 0.5e-4   # sd of Binance-proxy vs Chainlink TWAP change (measured ~0.3bp median abs, 1bp p95)

def load_up_bbo(mt, bbo_dir='bbo'):
    """all Up-token BBO rows as numpy: ai = row index into mt, ts ms, bid, ask (sorted by ai, ts)"""
    import pyarrow as pa, pyarrow.compute as pc
    up = pa.array([bytes.fromhex(x) for x in mt.up], type=pa.binary())
    parts = []
    for f in sorted(glob.glob(f'{D}/{bbo_dir}/*.parquet')):
        if pq.read_metadata(f).num_rows == 0: continue           # hour without target markets
        t = pq.read_table(f, columns=['timestamp', 'asset_id', 'best_bid', 'best_ask'])
        t = t.filter(pc.is_in(t['asset_id'].cast(pa.binary()), value_set=up))
        parts.append(pa.table({'ai': pc.index_in(t['asset_id'].cast(pa.binary()), value_set=up),
                               'ts': t['timestamp'].cast(pa.timestamp('ms')).cast(pa.int64()),
                               'bid': pc.cast(t['best_bid'], pa.float64()), 'ask': pc.cast(t['best_ask'], pa.float64())}))
    t = pa.concat_tables(parts)
    x = {c: t[c].to_numpy() for c in ('ai', 'ts', 'bid', 'ask')}
    o = np.lexsort((x['ts'], x['ai']))
    return {c: v[o] for c, v in x.items()}

def build(coin, mt_all, bbo, out='panel_{coin}.npz'):
    mt = mt_all[mt_all.coin == coin].sort_values('start').reset_index().rename(columns={'index': 'orig'})
    s0 = mt.start.dt.tz_convert(None).values.astype('datetime64[ms]').astype('int64')
    N = len(mt); bid = np.full((N, K), np.nan, np.float32); ask = np.full((N, K), np.nan, np.float32); nev = np.zeros((N, K), np.int16)
    gidx = mt_all.index.get_indexer(mt.orig)                  # row index into mt_all == ai
    grid_off = np.arange(K, dtype=np.int64) * 1000
    lo = np.searchsorted(bbo['ai'], gidx, 'left'); hi = np.searchsorted(bbo['ai'], gidx, 'right')
    for i in range(N):
        if hi[i] <= lo[i]: continue
        ts = bbo['ts'][lo[i]:hi[i]]; gt = s0[i] + grid_off
        j = np.searchsorted(ts, gt, side='right') - 1           # last event with ts <= grid time
        ok = j >= 0
        bid[i, ok] = bbo['bid'][lo[i]:hi[i]][j[ok]]; ask[i, ok] = bbo['ask'][lo[i]:hi[i]][j[ok]]
        c = np.searchsorted(ts, gt + 1000, side='right') - np.searchsorted(ts, gt, side='right')
        nev[i] = np.minimum(c, 32767)
    # Binance
    b = pd.read_parquet(f'{D}/bn1s/{SYM[coin]}.parquet'); t0 = b.t.values[0] // 1000; c = b.c.values.astype(float)
    lc = np.log(c); T = len(c)
    def cidx(sec): return (sec - t0).astype(np.int64)           # index of bar with open time `sec`
    # S(t) known at t = close of bar opened at t-1  -> idx(t-1)
    sec0 = s0 // 1000
    have = (cidx(sec0 - 4 * 3600) >= 0) & (cidx(sec0 + K) < T)
    S = np.full((N, K), np.nan); P0 = np.full(N, np.nan); F = np.full(N, np.nan); sig = np.full((N, K), np.nan)
    cs = np.r_[0, np.cumsum(c)]
    r5 = np.r_[np.full(5, np.nan), lc[5:] - lc[:-5]]                 # 5s log return ending at bar j
    r5sq = np.nan_to_num(r5 ** 2); cr = np.r_[0, np.cumsum(r5sq)]
    for i in np.where(have)[0]:
        s = sec0[i]; e = s + K
        P0[i] = (cs[cidx(np.array([s - 1]))[0]] - cs[cidx(np.array([s - 61]))[0]]) / 60      # bars s-61..s-2
        F[i] = (cs[cidx(np.array([e - 1]))[0]] - cs[cidx(np.array([e - 61]))[0]]) / 60
        tt = s + np.arange(K); jj = cidx(tt - 1)
        S[i] = c[jj]
        # sigma^2 per second from 5s returns: short 30min & long 4h windows ending at t
        vs = (cr[jj + 1] - cr[jj + 1 - 1800]) / 1800 / 5; vl = (cr[jj + 1] - cr[jj + 1 - 14400]) / 14400 / 5
        sig[i] = np.sqrt(0.5 * vs + 0.5 * vl)
    # fair P(Up) under TWAP-60 rule
    k = np.arange(K); tau = K - k                                     # seconds remaining
    e_rel = K                                                         # final bars: open times e-61..e-2 (relative to s: 839..898)
    fair = np.full((N, K), np.nan)
    for i in np.where(have)[0]:
        s = sec0[i]; e = s + K
        t = s + k
        # realized final-window bars: open times from e-61 to min(t-2, e-2)  (bar opened at t-2 closed at t-1 <= t) -> use closes known at t
        last_known = np.minimum(t - 1, e - 2)
        r = np.clip(last_known - (e - 61) + 1, 0, 60)
        sum_real = np.where(r > 0, cs[np.clip(cidx(last_known) + 1, 0, None)] - cs[cidx(np.array([e - 61]))[0]], 0.0)
        nf = 60 - r
        EF = (sum_real + nf * S[i]) / 60
        a = np.where(r > 0, 1, (e - 61) - (t - 1))                   # seconds from now to first future bar close in the final window
        a = np.maximum(a, 1)
        V = nf ** 2 * a + (nf - 1) * nf * (2 * nf - 1) / 6            # sum_ij min(d_i,d_j)
        sd = np.sqrt((sig[i] * np.sqrt(np.maximum(V, 1e-9)) / 60) ** 2 + BASIS ** 2) * EF   # price units, + Chainlink-vs-Binance basis noise
        z = (EF - P0[i]) / np.where(sd > 0, sd, np.nan)
        fair[i] = np.where(nf == 0, (EF >= P0[i]).astype(float), norm.cdf(z))
    y = mt.p_up_final.values.astype(float)
    np.savez_compressed(f'{D}/' + out.format(coin=coin), m=mt.m.values, start=s0, y=y, bid=bid, ask=ask, nev=nev, fair=fair.astype(np.float32),
                        S=S.astype(np.float64), P0=P0, F=F, sig=sig.astype(np.float32), ptb=mt.ptb.values.astype(float),
                        final=mt.final.values.astype(float), vol=mt.vol.values)
    print(coin, 'windows', N, 'with quotes', np.isfinite(bid[:, K // 2]).sum(), 'with binance', have.sum(), flush=True)

if __name__ == '__main__':
    tf = sys.argv[2] if len(sys.argv) > 2 else '15m'
    K = {'15m': 900, '5m': 300}[tf]                               # rebinds the module-level window length
    mt = pd.read_parquet(f'{D}/meta_{tf}.parquet')
    bbo = load_up_bbo(mt, 'bbo' if tf == '15m' else f'bbo{tf}'); print('bbo up rows', len(bbo['ts']), flush=True)
    for coin in sys.argv[1].split(','):
        build(coin, mt, bbo, 'panel_{coin}.npz' if tf == '15m' else f'panel{tf}_{{coin}}.npz')
