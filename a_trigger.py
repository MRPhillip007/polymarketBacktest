"""Entry at the FIRST second the conditions hold (inside a time interval) vs a fixed entry 8 min before close.
Interval chosen on the train half only, evaluated on the test half.  out: data/trigger_trades.parquet"""
import numpy as np, pandas as pd
from alib import *
from favlib import *
pd.set_option('display.width', 250); pd.set_option('display.max_columns', None)

if __name__ == '__main__':
    FS = {c: features(c) for c in COINS}
    rows = []; best = None
    for use_hours in (True, False):
        conds = {c: conditions(FS[c], use_hours=use_hours) for c in COINS}
        T = pd.concat([fixed_entry(FS[c], conds[c], 900 - 480) for c in COINS])
        rows.append(summary(T, f'fixed 8 min  hours={"12-20" if use_hours else "all"}'))
        for e in (10, 8, 6, 5, 4):
            for l in (3, 2, 1):
                T = pd.concat([first_entry(FS[c], conds[c], 900 - 60 * e, 900 - 60 * l) for c in COINS])
                r = summary(T, f'first {e}->{l} min  hours={"12-20" if use_hours else "all"}'); rows.append(r)
                if best is None or r['train_roi%'] > best[0]['train_roi%']: best = (r, T)
    R = pd.DataFrame(rows)
    print(R.round(2).to_string(index=False))
    print('\nbest on TRAIN:', best[0]['variant'], '-> test ROI %.2f%% (t %.2f), %.1f trades/day'
          % (best[0]['test_roi%'], best[0]['test_t'], best[0]['test_per_day']))
    best[1].to_parquet(f'{D}/trigger_trades.parquet')
    # when does the trigger fire?
    T = best[1]; print('entry time (min before close) quantiles:', ((900 - T.k) / 60).quantile([.1, .25, .5, .75, .9]).round(1).to_dict())
