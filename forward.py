"""Forward test of the favourite strategy with rules FROZEN on 2026-09-25 (no re-optimisation).
  python forward.py          -> update data (Gamma meta, archive hours, Binance 1s, incidents), rebuild panels, report
  python forward.py report   -> report only
Forward period starts 2026-09-25 00:00 UTC: the backtest never evaluated those windows (no Binance data at the time)."""
import sys, os, json, time, datetime as dt, subprocess, glob
import numpy as np, pandas as pd, requests

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, 'data')
FWD_START = pd.Timestamp('2026-09-25T00:00Z')
COINS_ = ['btc', 'eth', 'sol', 'xrp']
SYMS = {'btc': 'BTCUSDT', 'eth': 'ETHUSDT', 'sol': 'SOLUSDT', 'xrp': 'XRPUSDT'}

# ---- rules frozen on 2026-09-25: do not edit, otherwise it is no longer a forward test ----
FROZEN = dict(price=(0.80, 0.98), veto=0.0, streak=300, spread=2.0)
VARIANTS = {'A: first 6->3 min, 12-20 UTC': dict(win=(360, 180), hours=(12, 20)),
            'B: first 8->3 min, all hours': dict(win=(480, 180), hours=None)}
BACKTEST = {'A: first 6->3 min, 12-20 UTC': dict(roi=2.0, win=95.1, per_day=29),        # test half 05.09-25.09
            'B: first 8->3 min, all hours': dict(roi=1.0, win=92.5, per_day=120)}
DAY_LIMIT = 2.0                                                                          # -$2 per UTC day (risk variant)

def latest_archive_hour():
    import re
    for i in range(5):
        txt = requests.get('https://archive.pendulumflow.com/v3/SHA256SUMS.txt', timeout=60).text
        hs = sorted(set(re.findall(r'(\d{4}-\d{2}-\d{2}T\d{2})\.parquet', txt)))
        if hs: return dt.datetime.strptime(hs[-1], '%Y-%m-%dT%H')
        time.sleep(5)
    raise IOError('could not read the archive checksum list')

def update_meta(until):
    import pull
    old = pd.read_parquet(f'{D}/meta_15m.parquet')
    last = old.start.max().tz_convert(None).to_pydatetime()
    new = pull.meta(last + dt.timedelta(minutes=15), until, COINS_)
    if len(new):
        m = pd.concat([old, new]).drop_duplicates('slug', keep='last').sort_values(['coin', 'start']).reset_index(drop=True)
        m.to_parquet(f'{D}/meta_15m.parquet')
    print(f'meta: +{len(new)} windows (up to {until})')

def update_archive(until):
    import pull
    mt = pd.read_parquet(f'{D}/meta_15m.parquet')
    have = {os.path.basename(f)[:13] for f in glob.glob(f'{D}/bbo/*.parquet')}
    h = FWD_START.tz_convert(None).to_pydatetime(); n = 0
    while h <= until:
        if pull.hk(h) not in have:
            ts = pd.Timestamp(h, tz='UTC')
            ids = mt[(mt.start >= ts) & (mt.start < ts + pd.Timedelta(hours=1))].m.tolist()
            pull.hour_job(h, ids); n += 1
        h += dt.timedelta(hours=1)
    print(f'archive: {n} new hours')

def rest_1s(sym, start_ms, end_ms):
    """Binance spot 1s klines via the market-data-only endpoint (api.binance.com is geo-blocked here)"""
    out = []; t = start_ms
    while t < end_ms:
        for i in range(5):
            try:
                r = requests.get('https://data-api.binance.vision/api/v3/klines', timeout=30,
                                 params=dict(symbol=sym, interval='1s', startTime=t, endTime=end_ms - 1, limit=1000))
                if r.status_code == 200: break
            except Exception: pass
            time.sleep(2 * (i + 1))
        k = r.json()
        if not k: break
        out.extend(k); t = k[-1][0] + 1000
    df = pd.DataFrame(out).iloc[:, [0, 1, 2, 3, 4, 5, 8, 9]]
    df.columns = ['t', 'o', 'h', 'l', 'c', 'v', 'n', 'tb']
    return df.astype({'t': 'int64', 'o': float, 'h': float, 'l': float, 'c': float, 'v': float, 'n': 'int64', 'tb': float})

def update_binance(until):
    import binance1s
    end_ms = int(pd.Timestamp(until + dt.timedelta(hours=1, minutes=20), tz='UTC').value // 10**6)
    for coin, sym in SYMS.items():
        p = f'{D}/bn1s/{sym}.parquet'; b = pd.read_parquet(p)
        parts = [b]; t = int(b.t.max()) + 1000
        day = pd.Timestamp(t, unit='ms').normalize()
        while t < end_ms:                       # full days from the daily dumps when published, else REST
            dd = binance1s.day(sym, day.date()) if (day + pd.Timedelta(days=1)).value // 10**6 <= end_ms else None
            if dd is not None:
                parts.append(dd); t = int(dd.t.max()) + 1000
            else:
                r = rest_1s(sym, t, min(end_ms, int((day + pd.Timedelta(days=1)).value // 10**6)))
                if len(r) == 0: break
                parts.append(r); t = int(r.t.max()) + 1000
            day = pd.Timestamp(t, unit='ms').normalize()
        b = pd.concat(parts).drop_duplicates('t').sort_values('t')
        full = pd.DataFrame({'t': np.arange(b.t.min(), b.t.max() + 1000, 1000, dtype='int64')})     # contiguous 1s grid
        gaps = len(full) - len(b)
        b = full.merge(b, on='t', how='left'); b['c'] = b.c.ffill()
        for c_ in ('o', 'h', 'l'): b[c_] = b[c_].fillna(b.c)
        b[['v', 'tb']] = b[['v', 'tb']].fillna(0.0); b['n'] = b.n.fillna(0).astype('int64')
        b.to_parquet(p)
        print(f'binance {sym}: up to {pd.Timestamp(b.t.max(), unit="ms")}, filled gaps {gaps}')

def update():
    until = latest_archive_hour()
    print('latest archive hour:', until)
    open(f'{D}/INCIDENTS.json', 'wb').write(requests.get('https://archive.pendulumflow.com/INCIDENTS.json', timeout=60).content)
    update_meta(until + dt.timedelta(hours=1)); update_archive(until); update_binance(until)     # meta: windows starting inside the last hour too
    subprocess.run([sys.executable, os.path.join(HERE, 'panel.py'), ','.join(COINS_)], check=True, cwd=HERE)

def report():
    from alib import COINS, cl_t
    from favlib import features, conditions, first_entry, K
    from a_daystop import apply_stop
    FS = {c: features(c) for c in COINS}
    fwd0 = FWD_START.value // 10**6
    last_ok = max(int(F['start'][np.isfinite(F['P']['fair'][:, 0])].max()) for F in FS.values())
    print(f'\nFORWARD TEST  windows {FWD_START:%Y-%m-%d %H:%M} -> {pd.Timestamp(last_ok, unit="ms"):%Y-%m-%d %H:%M} UTC '
          f'({(last_ok - fwd0) / 3.6e6 + 0.25:.1f} h), rules frozen 2026-09-25, $1 per trade')
    rows = []; log = []
    for name, v in VARIANTS.items():
        cfg = dict(FROZEN, hours=v['hours'])
        T = pd.concat([first_entry(F, conditions(F, cfg, use_hours=v['hours'] is not None), K - v['win'][0], K - v['win'][1]) for F in FS.values()])
        T = T[(T.start >= fwd0) & T.win.isin([0, 1])]
        for lim in (None, DAY_LIMIT):
            X = apply_stop(T, lim)[0] if len(T) else T
            mu, t, n = cl_t(X.pnl, X.start) if len(X) > 2 else (X.pnl.mean() if len(X) else np.nan, np.nan, len(X))
            bt = BACKTEST[name]
            r = dict(variant=name, day_limit=f'-${lim:g}' if lim else 'none', trades=n, wins=int(X.win.sum()), losses=int((X.win == 0).sum()),
                     win_rate=X.win.mean() * 100 if n else np.nan, roi_per_trade=mu * 100 if n else np.nan, t=t, pnl_usd=X.pnl.sum(),
                     deposit_17_to=17 + X.pnl.sum(), backtest_roi=bt['roi'], backtest_win=bt['win'])
            rows.append(r); log.append(dict(run_utc=pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M'), data_until=str(pd.Timestamp(last_ok, unit='ms')), **r))
        if len(T):
            print(f'\n{name}: forward trades')
            show = T.sort_values('start').assign(time=lambda d: pd.to_datetime(d.start, unit='ms').dt.strftime('%m-%d %H:%M'),
                                                 min_before=lambda d: ((K - d.k) / 60).round(1))
            print(show[['time', 'coin', 'side', 'min_before', 'px', 'win', 'pnl']].round(3).to_string(index=False))
    print('\n' + pd.DataFrame(rows).round(2).to_string(index=False))
    lp = f'{D}/forward_log.csv'
    pd.DataFrame(log).to_csv(lp, mode='a', header=not os.path.exists(lp), index=False)
    print(f'\nappended to {lp}')

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'report': report()
    else:
        update(); subprocess.run([sys.executable, os.path.abspath(__file__), 'report'], check=True, cwd=HERE)
