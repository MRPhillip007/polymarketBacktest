"""Backtest with order-book liquidity: every entry is filled against the real ask levels of the favourite token.
  python a_liquidity.py plan   -> trades of the tested variants -> data/liq_trades.parquet
  python a_liquidity.py pull   -> book snapshots (asks) of those tokens around the entries -> data/book15/<hour>.parquet
  python a_liquidity.py sim    -> fills for $1 / $5 / $20 orders, report
Fill model: latest book snapshot at or before the arrival second (entry second + 1 s); levels cheaper than the best ask
at arrival are dropped (already gone); if the snapshot has no level at that best ask (a new order arrived after the
snapshot) it is assumed to hold the Polymarket minimum of 5 shares; the order takes levels up to a price cap = decision
price + 1c (<= 0.99): if the ask has already moved above the cap, there is no trade; partial fills allowed; fee
0.07*p*(1-p) per share. Snapshots older than 30 s (or missing) are replaced by the best level with 5 shares."""
import sys, os, time, datetime as dt, numpy as np, pandas as pd
import pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
import concurrent.futures as cf
from alib import *
from favlib import features, conditions, first_entry, MID_TS
pd.set_option('display.width', 250); pd.set_option('display.max_columns', None)
OUT = f'{D}/book15'
COINS4 = ['btc', 'eth', 'sol', 'xrp']
VARIANTS = {f'{r} {m} veto>{v:g}c': (m, v, ef, et, hrs) for r, ef, et, hrs in (('A15', 360, 180, (12, 20)), ('B15', 480, 180, None))
            for m in ('old', 'blend') for v in (0.0, 1.0)}

def plan():
    mt = pd.read_parquet(f'{D}/meta_15m.parquet'); mt['s'] = mt.start.dt.tz_convert(None).values.astype('datetime64[ms]').astype('int64')
    rows = []
    for c in COINS4:
        Fs = {m: features(c, 300, '15m', m) for m in ('old', 'blend')}
        for name, (m, v, ef, et, hrs) in VARIANTS.items():
            F = Fs[m]; cfg = dict(price=(0.80, 0.98), veto=v, spread=2.0, streak=300, hours=hrs)
            T = first_entry(F, conditions(F, cfg, use_hours=hrs is not None), 900 - ef, 900 - et)
            T['pxd'] = F['pxd'][T.row.values, T.k.values]; T['variant'] = name; rows.append(T[T.win.isin([0, 1])])
    T = pd.concat(rows, ignore_index=True)
    T = T.merge(mt[['coin', 's', 'up', 'dn']].rename(columns={'s': 'start'}), on=['coin', 'start'], how='left')
    T['token'] = np.where(T.side == 'Up', T.up, T.dn); T['t_exec'] = T.start + (T.k + 1) * 1000
    T.drop(columns=['up', 'dn']).to_parquet(f'{D}/liq_trades.parquet')
    print(T.groupby('variant').size().to_string()); print('unique entries', T[['token', 't_exec']].drop_duplicates().shape[0])

def pull_hour(h, need):
    import rng, pull
    p = f'{OUT}/{pull.hk(h)}.parquet'
    if os.path.exists(p): return 0
    sp, pf = rng.open_remote(pull.url(h))
    et = rng.rg_stats(pf, 'event_type'); mk = rng.rg_stats(pf, 'market')
    mkts = [bytes.fromhex(x) for x in need.m.unique()]
    g = [i for i, (a, _) in enumerate(et) if a == 'book' and any(mk[i][0] <= x <= mk[i][1] for x in mkts)]
    nb = rng.prefetch(sp, pf, g, ['timestamp', 'asset_id', 'asks'], gap=65536) if g else 0
    keep = []
    if g:
        tb = pf.read_row_groups(g, columns=['timestamp', 'asset_id', 'asks'])
        tb = tb.filter(pc.is_in(tb['asset_id'].cast(pa.binary()), value_set=pa.array([bytes.fromhex(x) for x in need.token.unique()], type=pa.binary())))
        df = pd.DataFrame({'ts': tb['timestamp'].cast(pa.timestamp('ms')).cast(pa.int64()).to_numpy(),
                           'token': [x.hex() for x in tb['asset_id'].cast(pa.binary()).to_pylist()]})
        asks = tb['asks'].to_pylist()
        for tok, gg in need.groupby('token'):                            # keep snapshots from 60 s before each entry
            idx = np.where(df.token.values == tok)[0]
            if not len(idx): continue
            ts = df.ts.values[idx]; sel = np.zeros(len(idx), bool)
            for te in gg.t_exec.values: sel |= (ts <= te) & (ts > te - 60_000)
            for i in idx[sel]:
                lv = asks[i] or []
                keep.append((int(df.ts.values[i]), tok, [float(l['price']) for l in lv], [float(l['size']) for l in lv]))
    out = pd.DataFrame(keep, columns=['ts', 'token', 'px', 'sz'])
    out.to_parquet(p + '.tmp'); os.replace(p + '.tmp', p)
    return nb

def pull_all(workers=3, since=None):
    import pull
    os.makedirs(OUT, exist_ok=True)
    T = pd.read_parquet(f'{D}/liq_trades.parquet')
    if since: T = T[T.t_exec >= pd.Timestamp(since, tz='UTC').value // 10**6]
    mt = pd.read_parquet(f'{D}/meta_15m.parquet'); tok2m = pd.concat([mt[['up', 'm']].rename(columns={'up': 'token'}), mt[['dn', 'm']].rename(columns={'dn': 'token'})])
    N = T[['token', 't_exec']].drop_duplicates().merge(tok2m, on='token')
    N['hour'] = pd.to_datetime(N.t_exec, unit='ms').dt.floor('h')
    items = [(h.to_pydatetime(), g) for h, g in N.groupby('hour')]
    t0 = time.time(); tot = 0; errs = []
    with cf.ThreadPoolExecutor(workers) as ex:
        fut = {ex.submit(pull_hour, h, g): h for h, g in items}
        for i, f in enumerate(cf.as_completed(fut), 1):
            try: tot += f.result()
            except Exception as e: errs.append((str(fut[f]), repr(e)[:150]))
            if i % 24 == 0 or i == len(items):
                el = time.time() - t0; print(f'{i}/{len(items)} {el:.0f}s {tot / 1e9:.2f}GB {tot / 1e6 / max(el, 1):.1f}MB/s errs={len(errs)}', flush=True)
    print('errors', len(errs)); [print(' ', e) for e in errs[:10]]

MIN_NEW = 5.0   # a level that is not in the (older) snapshot came from a new limit order: at least 5 shares (Polymarket minimum)

def fill(levels_px, levels_sz, best_ask, cap, stake):
    """walk the ask side up to `cap`: returns (dollars spent incl. fee, shares bought, best level was unknown)"""
    best_ask, cap = round(float(best_ask), 4), round(float(cap), 4)          # panel prices are float32 -> back to the 0.001 grid
    px = np.round(np.asarray(levels_px, float), 4); sz = np.asarray(levels_sz, float); o = np.argsort(px); px, sz = px[o], sz[o]
    m = px >= best_ask - 1e-9; px, sz = px[m], sz[m]                      # levels cheaper than the best ask at arrival are gone
    unknown = not (len(px) and abs(px[0] - best_ask) < 1e-9)
    if unknown: px, sz = np.r_[best_ask, px], np.r_[MIN_NEW, sz]           # new best level after the snapshot
    m = px <= cap + 1e-9; px, sz = px[m], sz[m]
    spent = 0.0; sh = 0.0
    for p, s_ in zip(px, sz):
        unit = p * (1 + 0.07 * (1 - p)); take = min(s_ * unit, stake - spent); sh += take / unit; spent += take
        if spent >= stake - 1e-9: break
    return spent, sh, unknown

def sim():
    """test half only (book snapshots were downloaded for 05.09-25.09): ideal fill vs price-cap only vs full order book"""
    from a_daystop import apply_stop
    T = pd.read_parquet(f'{D}/liq_trades.parquet'); T['hour'] = pd.to_datetime(T.t_exec, unit='ms').dt.strftime('%Y-%m-%dT%H')
    have = {f[:13] for f in os.listdir(OUT) if f.endswith('.parquet')}
    T = T[(T.start >= MID_TS) & T.hour.isin(have)].reset_index(drop=True)
    books = {h: pd.read_parquet(f'{OUT}/{h}.parquet') for h in T.hour.unique()}
    snap = []
    for r in T.itertuples():
        b = books[r.hour]; x = b[(b.token == r.token) & (b.ts <= r.t_exec)]
        snap.append(x.ts.idxmax() if len(x) and r.t_exec - x.ts.max() <= 30_000 else -1)
    T['snap'] = snap
    pxr = np.round(T.px.values, 4); capr = np.round(np.minimum(0.99, T.pxd.values + 0.01), 4)
    T['in_cap'] = pxr <= capr + 1e-9
    print(f'test half: {len(T)} entries (all variants), fresh snapshot (<=30 s) for {np.mean(T.snap >= 0) * 100:.1f}%, '
          f'ask moved above decision+1c within 1 s: {np.mean(~T.in_cap) * 100:.1f}%')
    out = []; bank = []
    for stake in (1.0, 5.0, 20.0):
        res = [fill(books[r.hour].loc[r.snap].px, books[r.hour].loc[r.snap].sz, r.px, min(0.99, r.pxd + 0.01), stake) if r.snap >= 0
               else fill([], [], r.px, min(0.99, r.pxd + 0.01), stake) for r in T.itertuples()]
        sp, sh, unk = map(np.array, zip(*res))
        X = T.assign(spent=sp, shares=sh, pnl_liq=sh * T.win - sp)
        X['pnl_cap'] = np.where(X.in_cap, X.pnl, np.nan)                     # infinite depth, but no trade if the ask ran away
        for v, d in X.groupby('variant'):
            f = d[d.spent > 0]; g = f.assign(r=f.pnl_liq / f.spent); mu, t, n = cl_t(g.r, g.start) if len(g) > 2 else (np.nan, np.nan, len(g))
            unit = np.round(f.px, 4) * (1 + 0.07 * (1 - np.round(f.px, 4)))
            out.append(dict(variant=v, stake=stake, signals=len(d), filled_pct=len(f) / len(d) * 100, full_pct=(d.spent >= stake - 1e-6).mean() * 100,
                            avg_fill_usd=f.spent.mean(), slippage_c=((f.spent / f.shares - unit) * 100).mean(),
                            roi_ideal=d.pnl.mean() * 100, roi_cap_only=d.pnl_cap.mean() * 100, roi_book=f.pnl_liq.sum() / f.spent.sum() * 100, t_book=t,
                            pnl_book=f.pnl_liq.sum()))
            if stake == 1.0:
                for lim in (None, 2.0):
                    Kt, bk = apply_stop(f.assign(pnl=f.pnl_liq), lim, None, 17.0)
                    bank.append(dict(variant=v, day_limit='-$2' if lim else 'none', trades=len(Kt), final=bk['final'], max_dd=bk['max_dd']))
    R = pd.DataFrame(out); R.to_parquet(f'{D}/liq_results.parquet')
    print(R.round(2).to_string(index=False))
    print()
    print('$17 deposit, $1 orders filled against the book, test half 05.09-25.09:')
    print(pd.DataFrame(bank).round(2).to_string(index=False))

if __name__ == '__main__':
    if sys.argv[1] == 'pull': pull_all(since=sys.argv[2] if len(sys.argv) > 2 else None)
    else: {'plan': plan, 'sim': sim}[sys.argv[1]]()
