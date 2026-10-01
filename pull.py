"""Pull Polymarket crypto Up/Down 15m data: Gamma metadata + pendulumflow v3 quotes/trades.
usage: python pull.py meta  2026-08-18T06 2026-09-25T11 btc,eth,sol,xrp [15m|5m]
       python pull.py hours 2026-08-18T06 2026-09-25T11 [workers]
"""
import sys, os, json, time, datetime as dt, requests, threading
import pandas as pd, numpy as np, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
import concurrent.futures as cf
import rng

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, 'data')
BASE = 'https://archive.pendulumflow.com/v3'
def url(h): return f"{BASE}/{h:%Y-%m-%d}/{h:%H}/{h:%Y-%m-%dT%H}.parquet"
def hk(h): return f"{h:%Y-%m-%dT%H}"
def hours(a, b):
    while a < b:
        yield a; a += dt.timedelta(hours=1)

def tok_hex(s): return format(int(s), '064x')

def meta(a, b, coins, tf='15m'):
    step = {'15m': 900, '5m': 300}[tf]
    starts = range(int(a.replace(tzinfo=dt.timezone.utc).timestamp()), int(b.replace(tzinfo=dt.timezone.utc).timestamp()), step)
    slugs = [f'{c}-updown-{tf}-{s}' for c in coins for s in starts]
    rows = []
    def fetch(chunk):
        for i in range(6):
            try:
                r = requests.get('https://gamma-api.polymarket.com/events', params=[('slug', s) for s in chunk] + [('limit', 500)], timeout=60)
                if r.status_code == 200: return r.json()
            except Exception: pass
            time.sleep(2 * (i + 1))
        raise IOError('gamma failed')
    chunks = [slugs[i:i + 50] for i in range(0, len(slugs), 50)]
    with cf.ThreadPoolExecutor(4) as ex:
        for evs in ex.map(fetch, chunks):
            for e in evs:
                for m in e.get('markets', []):
                    outs = json.loads(m['outcomes']); toks = json.loads(m['clobTokenIds'])
                    prices = json.loads(m.get('outcomePrices') or '[]')
                    em = e.get('eventMetadata') or {}
                    rows.append(dict(slug=e['slug'], coin=e['slug'].split('-')[0], start=int(e['slug'].rsplit('-', 1)[1]),
                        m=m['conditionId'][2:].lower(), up=tok_hex(toks[outs.index('Up')]), dn=tok_hex(toks[outs.index('Down')]),
                        p_up_final=float(prices[outs.index('Up')]) if prices else np.nan,
                        ptb=em.get('priceToBeat'), final=em.get('finalPrice'), vol=float(m.get('volumeClob') or 0),
                        closed=e.get('closedTime'), rsrc=e.get('resolutionSource'), fee=m.get('feeType')))
    df = pd.DataFrame(rows)
    df['start'] = pd.to_datetime(df.start, unit='s', utc=True)
    print('requested', len(slugs), 'got', len(df))
    return df

BBO = ['timestamp', 'market', 'asset_id', 'best_bid', 'best_ask']
TR = ['timestamp', 'market', 'asset_id', 'price', 'size', 'side', 'fee_rate_bps', 'transaction_hash']
REG = ['event_type', 'timestamp', 'market', 'slug', 'outcomes', 'assets_ids', 'winning_outcome', 'winning_asset_id']

def hour_job(h, ids):
    out = {k: f"{D}/{k}/{hk(h)}.parquet" for k in ('bbo', 'tr', 'reg')}
    if all(os.path.exists(p) for p in out.values()): return 'skip', 0
    u = url(h)
    sp, pf = rng.open_remote(u)
    et = rng.rg_stats(pf, 'event_type'); mk = rng.rg_stats(pf, 'market')
    bid = [bytes.fromhex(x) for x in ids]
    g_bbo = [i for i, (a, _) in enumerate(et) if a == 'best_bid_ask' and any(mk[i][0] <= x <= mk[i][1] for x in bid)]
    g_tr = [i for i, (a, _) in enumerate(et) if a == 'last_trade_price']
    g_reg = [i for i, (a, _) in enumerate(et) if a in ('new_market', 'market_resolved')]
    nb = 0
    def get(gs, cols, filt):
        nonlocal nb
        if not gs: return None
        cols = [c for c in cols if c in pf.schema_arrow.names]
        nb += rng.prefetch(sp, pf, gs, cols, gap=65536)
        t = pf.read_row_groups(gs, columns=cols)
        if filt and bid:
            t = t.filter(pc.is_in(t['market'], value_set=pa.array(bid, type=t.schema.field('market').type)))
        return t
    tb = get(g_bbo, BBO, True); tt = get(g_tr, TR, True); tr_ = get(g_reg, REG, False)
    for k, t in (('bbo', tb), ('tr', tt), ('reg', tr_)):
        if t is None: t = pa.table({})
        pq.write_table(t, out[k] + '.tmp', compression='zstd'); os.replace(out[k] + '.tmp', out[k])
    return tb.num_rows if tb is not None else 0, nb

if __name__ == '__main__':
    cmd = sys.argv[1]; a = dt.datetime.fromisoformat(sys.argv[2]); b = dt.datetime.fromisoformat(sys.argv[3])
    for k in ('bbo', 'tr', 'reg'): os.makedirs(f"{D}/{k}", exist_ok=True)
    if cmd == 'meta':
        tf = sys.argv[5] if len(sys.argv) > 5 else '15m'           # '5m' -> Chainlink TWAP every 5 min via priceToBeat/finalPrice
        df = meta(a, b, sys.argv[4].split(','), tf)
        df.to_parquet(f"{D}/meta_{tf}.parquet"); print(df.groupby('coin').size())
    else:
        workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
        mt = pd.read_parquet(f"{D}/meta_15m.parquet")
        items = []
        for h in hours(a, b):
            ts = pd.Timestamp(h, tz='UTC')
            items.append((h, mt[(mt.start >= ts) & (mt.start < ts + pd.Timedelta(hours=1))].m.tolist()))
        t0 = time.time(); tot = 0; errs = []
        with cf.ThreadPoolExecutor(workers) as ex:
            fut = {ex.submit(hour_job, h, ids): h for h, ids in items}
            for i, f in enumerate(cf.as_completed(fut), 1):
                try:
                    n, nb = f.result(); tot += nb
                except Exception as e:
                    errs.append((hk(fut[f]), repr(e)[:160]))
                if i % 12 == 0 or i == len(items):
                    el = time.time() - t0
                    print(f"{i}/{len(items)} {el:.0f}s {tot/1e9:.2f}GB {tot/1e6/max(el,1):.1f}MB/s errs={len(errs)}", flush=True)
        json.dump(errs, open(f"{D}/errors.json", 'w'), indent=1)
        print('errors', len(errs)); [print(' ', e) for e in errs[:20]]
