"""Binance spot 1s klines from data.binance.vision -> data/bn1s/<SYM>.parquet (t = open time ms UTC, tb = taker buy volume)."""
import requests, io, zipfile, pandas as pd, numpy as np, datetime as dt, os, sys
import concurrent.futures as cf
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'bn1s'); os.makedirs(D, exist_ok=True)
def day(sym, d):
    u = f"https://data.binance.vision/data/spot/daily/klines/{sym}/1s/{sym}-1s-{d:%Y-%m-%d}.zip"
    for i in range(5):
        r = requests.get(u, timeout=120)
        if r.status_code == 200: break
        if r.status_code == 404: return None
    z = zipfile.ZipFile(io.BytesIO(r.content))
    df = pd.read_csv(z.open(z.namelist()[0]), header=None, usecols=[0, 1, 2, 3, 4, 5, 8, 9], names=['t', 'o', 'h', 'l', 'c', 'v', 'n', 'tb'])   # tb = taker buy base volume
    if df.t.iloc[0] > 1e14: df['t'] = df.t // 1000   # microseconds -> ms
    return df
def sym(s, a, b):
    days = [a + dt.timedelta(days=i) for i in range((b - a).days + 1)]
    with cf.ThreadPoolExecutor(3) as ex:
        parts = [p for p in ex.map(lambda d: day(s, d), days) if p is not None]
    df = pd.concat(parts, ignore_index=True).sort_values('t').drop_duplicates('t')
    df.to_parquet(f"{D}/{s}.parquet"); print(s, len(df), pd.to_datetime(df.t.iloc[[0, -1]], unit='ms').tolist(), flush=True)
if __name__ == '__main__':
    a, b = dt.date.fromisoformat(sys.argv[1]), dt.date.fromisoformat(sys.argv[2])
    for s in sys.argv[3].split(','): sym(s, a, b)
