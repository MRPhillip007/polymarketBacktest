"""Buy the favourite TAU seconds before the end of every 15m window (market order, taker), hold to resolution.
Fixed stake per trade, bankroll simulated from a starting deposit.
usage: python a_fav5m.py [tau_sec=300] [stake=3] [deposit=17]"""
import sys, numpy as np, pandas as pd
from alib import *
from a_incidents import in_inc
pd.set_option('display.width', 250); pd.set_option('display.max_columns', None)
TAU = int(sys.argv[1]) if len(sys.argv) > 1 else 300
STAKE = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
DEP = float(sys.argv[3]) if len(sys.argv) > 3 else 17.0
DELAY = 1                      # the order reaches the book 1 s after the decision and pays the favourite's ask prevailing then
ORDER = ['btc', 'eth', 'sol', 'xrp']   # priority when the balance cannot cover every coin in the same window

def trades(coin):
    P = load(coin); y = P['y']; st = P['start']; k = 900 - TAU
    b, a = P['bid'][:, k].astype(float), P['ask'][:, k].astype(float)
    b1, a1 = P['bid'][:, k + DELAY].astype(float), P['ask'][:, k + DELAY].astype(float)
    mid = (a + b) / 2
    up = mid > 0.5
    px = np.where(up, a1, 1 - b1)                     # favourite's ask at arrival (Down ask = 1 - Up bid)
    ok = np.isfinite(mid) & np.isfinite(px) & (a - b <= 0.10) & (mid != 0.5) & (px > 0) & (px < 1)
    win = np.where(up, y, 1 - y)
    sh = STAKE / (px * (1 + 0.07 * (1 - px)))          # stake covers price + taker fee 0.07*p*(1-p) per share
    df = pd.DataFrame({'coin': coin, 'start': st, 'side': np.where(up, 'Up', 'Down'), 'px': px, 'win': win,
                       'pnl': sh * win - STAKE, 'edge_c': (win - px - fee(px)) * 100, 'inc': in_inc(st)})
    return df[ok]

def stats(T):
    mu, t, n = cl_t(T.pnl, T.start)
    days = (T.start.max() - T.start.min()) / 86_400_000 + 1 / 96
    return pd.Series({'trades': n, 'win_rate_%': T.win.mean() * 100, 'avg_price': T.px.mean(),
                      'breakeven_win_%': (T.px * (1 + 0.07 * (1 - T.px))).mean() * 100,
                      'edge_c_per_contract': T.edge_c.mean(), 'avg_pnl_$': mu, 'roi_per_trade_%': mu / STAKE * 100, 't': t,
                      'avg_win_$': T.pnl[T.win == 1].mean(), 'avg_loss_$': T.pnl[T.win == 0].mean(),
                      'pnl_all_trades_$': T.pnl.sum(), 'pnl_per_day_$': T.pnl.sum() / days})

def bankroll(T):
    """one deposit; each window's positions resolve (~7 min) before the next entry 15 min later"""
    bal = peak = DEP; mdd = 0; low = DEP; n_tr = 0; skipped = 0; bust = None
    pr = {c: i for i, c in enumerate(ORDER)}
    for s, g in T.sort_values('start').groupby('start', sort=True):
        g = g.iloc[np.argsort(g.coin.map(pr).values)]
        n = int(min(len(g), bal // STAKE)); skipped += len(g) - n
        if n == 0:
            if bust is None: bust = pd.to_datetime(s, unit='ms')
            continue
        bal += g.pnl.iloc[:n].sum(); n_tr += n
        peak = max(peak, bal); mdd = max(mdd, peak - bal); low = min(low, bal)
    return pd.Series({'start_$': DEP, 'final_$': bal, 'profit_$': bal - DEP, 'return_%': (bal / DEP - 1) * 100, 'min_balance_$': low,
                      'max_drawdown_$': mdd, 'trades_taken': n_tr, 'skipped_no_cash': skipped, 'bust_at': bust})

if __name__ == '__main__':
    A = pd.concat([trades(c) for c in COINS], ignore_index=True)
    A['week'] = pd.to_datetime(A.start, unit='ms').dt.to_period('W-SUN').astype(str).str[:10]
    N = A[~A.inc]
    print(f'Buy favourite {TAU}s before close, stake ${STAKE}, deposit ${DEP}. Period',
          pd.to_datetime(A.start.min(), unit='ms'), '->', pd.to_datetime(A.start.max(), unit='ms'))
    print(f'windows excluded as venue incidents: {A.inc.sum()} of {len(A)}\n')
    print('== per trade (no incidents) ==')
    per = pd.concat({c: stats(N[N.coin == c]) for c in COINS} | {'all 4': stats(N), 'all 4 incl. incidents': stats(A)}, axis=1)
    print(per.round(3).to_string())
    print('\n== bankroll: separate $%g deposit per coin, and one deposit for all 4 coins ==' % DEP)
    bk = pd.concat({c: bankroll(N[N.coin == c]) for c in COINS} | {'all 4 (one deposit)': bankroll(N)}, axis=1)
    print(bk.to_string())
    print('\n== by favourite price (all 4, no incidents) ==')
    N = N.assign(bucket=pd.cut(N.px, [0.5, .6, .7, .8, .9, .95, .98, .99, 1.0], right=False))
    print(N.groupby('bucket', observed=True).apply(lambda d: pd.Series({'trades': len(d), 'share_%': len(d) / len(N) * 100,
          'win_%': d.win.mean() * 100, 'avg_px': d.px.mean(), 'edge_c': d.edge_c.mean(), 'avg_pnl_$': d.pnl.mean(),
          'sum_pnl_$': d.pnl.sum()})).round(3).to_string())
    print('\n== by week (all 4, no incidents) ==')
    print(N.groupby('week').apply(lambda d: pd.Series({'trades': len(d), 'win_%': d.win.mean() * 100, 'edge_c': d.edge_c.mean(),
          'avg_pnl_$': d.pnl.mean(), 'sum_pnl_$': d.pnl.sum()})).round(3).to_string())
    A.to_parquet(f'{D}/fav{TAU}_trades.parquet')
