"""Polymarket 5m Up/Down: extend Gamma metadata and pull best bid/ask of the chosen coins from the v3 archive.
usage: python pull5m.py 2026-08-18T06 2026-09-25T13 btc,eth,sol [workers]  -> data/meta_5m.parquet, data/bbo5m/<hour>.parquet"""
import sys, os, time, datetime as dt
import pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
import concurrent.futures as cf
import rng, pull
from pull import url, hk, hours, D, BBO
OUT = f'{D}/bbo5m'

def extend_meta(until, coins):
    p = f'{D}/meta_5m.parquet'; old = pd.read_parquet(p)
    last = old.start.max().tz_convert(None).to_pydatetime() + dt.timedelta(minutes=5)
    new = pull.meta(last, until, coins, '5m') if last < until else old.iloc[:0]
    if len(new):
        pd.concat([old, new]).drop_duplicates('slug', keep='last').sort_values(['coin', 'start']).reset_index(drop=True).to_parquet(p)
    print('meta_5m: +', len(new), 'windows')

def job(h, ids):
    p = f'{OUT}/{hk(h)}.parquet'
    if os.path.exists(p): return 0
    sp, pf = rng.open_remote(url(h))
    et = rng.rg_stats(pf, 'event_type'); mk = rng.rg_stats(pf, 'market')
    bid = [bytes.fromhex(x) for x in ids]
    g = [i for i, (a, _) in enumerate(et) if a == 'best_bid_ask' and any(mk[i][0] <= x <= mk[i][1] for x in bid)]
    nb = 0
    if g:
        nb = rng.prefetch(sp, pf, g, BBO, gap=65536)
        t = pf.read_row_groups(g, columns=BBO)
        t = t.filter(pc.is_in(t['market'], value_set=pa.array(bid, type=t.schema.field('market').type)))
    else:
        t = pa.table({c: pa.array([], type=pf.schema_arrow.field(c).type) for c in BBO})
    pq.write_table(t, p + '.tmp', compression='zstd'); os.replace(p + '.tmp', p)
    return nb

if __name__ == '__main__':
    a, b = dt.datetime.fromisoformat(sys.argv[1]), dt.datetime.fromisoformat(sys.argv[2])
    coins = sys.argv[3].split(','); workers = int(sys.argv[4]) if len(sys.argv) > 4 else 6
    os.makedirs(OUT, exist_ok=True)
    extend_meta(b, coins)
    mt = pd.read_parquet(f'{D}/meta_5m.parquet'); mt = mt[mt.coin.isin(coins)]
    items = []
    for h in hours(a, b):
        ts = pd.Timestamp(h, tz='UTC'); items.append((h, mt[(mt.start >= ts) & (mt.start < ts + pd.Timedelta(hours=1))].m.tolist()))
    t0 = time.time(); tot = 0; errs = []
    with cf.ThreadPoolExecutor(workers) as ex:
        fut = {ex.submit(job, h, ids): h for h, ids in items}
        for i, f in enumerate(cf.as_completed(fut), 1):
            try: tot += f.result()
            except Exception as e: errs.append((hk(fut[f]), repr(e)[:150]))
            if i % 24 == 0 or i == len(items):
                el = time.time() - t0
                print(f'{i}/{len(items)} {el:.0f}s {tot / 1e9:.2f}GB {tot / 1e6 / max(el, 1):.1f}MB/s errs={len(errs)}', flush=True)
    print('errors', len(errs)); [print(' ', e) for e in errs[:20]]
