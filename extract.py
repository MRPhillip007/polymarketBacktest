"""Extract Polymarket crypto Up/Down 15m markets from the pendulumflow v3 archive.
phase 'reg': new_market + market_resolved for every hour  -> data/reg/<hour>.parquet
phase 'q'  : best_bid_ask + last_trade_price for live 15m markets of chosen coins -> data/q/<hour>.parquet
"""
import duckdb, datetime as dt, pandas as pd, os, sys, time, glob, re, threading
import concurrent.futures as cf

BASE = 'https://archive.pendulumflow.com/v3'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
_local = threading.local()

def url(h): return f"{BASE}/{h:%Y-%m-%d}/{h:%H}/{h:%Y-%m-%dT%H}.parquet"
def hkey(h): return f"{h:%Y-%m-%dT%H}"

def con():
    c = getattr(_local, 'c', None)
    if c is None:
        c = duckdb.connect()
        c.sql("SET TimeZone='UTC'; LOAD httpfs; SET threads=4;")
        c.sql("SET http_retries=8; SET http_timeout=120000;")
        _local.c = c
    return c

def hours(a, b):
    h = a
    while h < b:
        yield h
        h += dt.timedelta(hours=1)

def reg_hour(h):
    p = f"{OUT}/reg/{hkey(h)}.parquet"
    if os.path.exists(p): return 'skip'
    df = con().sql(f"""SELECT event_type, timestamp, lower(hex(market)) m, slug, question, outcomes,
        [lower(hex(a)) for a in assets_ids] aids, winning_outcome, lower(hex(winning_asset_id)) win_aid
        FROM read_parquet('{url(h)}', union_by_name=true)
        WHERE event_type IN ('new_market','market_resolved')""").df()
    df.to_parquet(p); return len(df)

def load_reg():
    fs = sorted(glob.glob(f"{OUT}/reg/*.parquet"))
    r = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    nm = r[r.event_type == 'new_market'].drop_duplicates('m').copy()
    x = nm.slug.str.extract(r'^(?P<coin>[a-z0-9]+)-updown-(?P<tf>\d+m)-(?P<start>\d{10})$')
    nm = pd.concat([nm, x], axis=1)
    nm = nm[nm.tf.notna()].copy()
    nm['start'] = pd.to_datetime(nm.start.astype('int64'), unit='s', utc=True)
    nm['up_aid'] = [a[list(o).index('Up')] for a, o in zip(nm.aids, nm.outcomes)]
    nm['dn_aid'] = [a[list(o).index('Down')] for a, o in zip(nm.aids, nm.outcomes)]
    rs = r[r.event_type == 'market_resolved'].sort_values('timestamp').drop_duplicates('m', keep='last')
    nm = nm.merge(rs[['m', 'winning_outcome', 'timestamp']].rename(columns={'timestamp': 'resolved_at'}), on='m', how='left')
    return nm[['m', 'slug', 'coin', 'tf', 'start', 'up_aid', 'dn_aid', 'winning_outcome', 'resolved_at']]

def q_hour(h, mk):
    p = f"{OUT}/q/{hkey(h)}.parquet"
    if os.path.exists(p): return 'skip'
    if not mk: return 'nomk'
    inl = ",".join(f"from_hex('{m}')" for m in mk)
    df = con().sql(f"""SELECT event_type, timestamp, timestamp_received, lower(hex(market)) m, lower(hex(asset_id)) a,
        best_bid::DOUBLE bid, best_ask::DOUBLE ask, price::DOUBLE price, size::DOUBLE size, side
        FROM read_parquet('{url(h)}', union_by_name=true)
        WHERE event_type IN ('best_bid_ask','last_trade_price') AND market IN ({inl})""").df()
    df['event_type'] = df.event_type.astype('category'); df['side'] = df.side.astype('category')
    df.to_parquet(p); return len(df)

def run(fn, items, workers=8, label=''):
    t = time.time(); done = 0; errs = []
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, *it): it for it in items}
        for f in cf.as_completed(futs):
            done += 1
            try: f.result()
            except Exception as e: errs.append((futs[f][0], repr(e)[:200]))
            if done % 24 == 0: print(f"{label} {done}/{len(items)} {time.time()-t:.0f}s errs={len(errs)}", flush=True)
    print(f"{label} done {done} in {time.time()-t:.0f}s, errors {len(errs)}", flush=True)
    for e in errs[:10]: print('  ERR', e)
    return errs

if __name__ == '__main__':
    phase, a, b = sys.argv[1], dt.datetime.fromisoformat(sys.argv[2]), dt.datetime.fromisoformat(sys.argv[3])
    coins = sys.argv[4].split(',') if len(sys.argv) > 4 else ['btc', 'eth']
    workers = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    os.makedirs(f"{OUT}/reg", exist_ok=True); os.makedirs(f"{OUT}/q", exist_ok=True)
    if phase == 'reg':
        run(reg_hour, [(h,) for h in hours(a, b)], workers, 'reg')
    else:
        nm = load_reg()
        nm = nm[(nm.tf == '15m') & nm.coin.isin(coins)]
        items = []
        for h in hours(a, b):
            ts = pd.Timestamp(h, tz='UTC')
            mk = nm[(nm.start >= ts) & (nm.start < ts + pd.Timedelta(hours=1))].m.tolist()
            items.append((h, mk))
        print('hours', len(items), 'with markets', sum(1 for _, m in items if m), 'markets', sum(len(m) for _, m in items))
        run(q_hour, items, workers, 'q')
