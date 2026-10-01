"""Do the TWAP model v2 and its blend with the market improve the favourite strategies? (veto = model not against)
Model v2 was fitted on the first half, so only the TEST half (05.09-25.09) is out-of-sample for 'new' and 'blend'."""
import numpy as np, pandas as pd
from alib import *
from favlib import features, conditions, first_entry, summary
pd.set_option('display.width', 250); pd.set_option('display.max_columns', None)
RULES = {'A15 (15m, 6->3 min, 12-20 UTC)': ('15m', ['btc', 'eth', 'sol', 'xrp'], 300, 360, 180, (12, 20)),
         'B15 (15m, 8->3 min, all hours)': ('15m', ['btc', 'eth', 'sol', 'xrp'], 300, 480, 180, None),
         'A5 (5m, 2:00->1:00, 12-20 UTC)': ('5m', ['btc', 'eth', 'sol'], 100, 120, 60, (12, 20)),
         'B5 (5m, 2:40->1:00, all hours)': ('5m', ['btc', 'eth', 'sol'], 100, 160, 60, None)}
rows = []
for name, (tf, coins, streak, ef, et, hrs) in RULES.items():
    for model in ('old', 'new', 'blend'):
        for veto in (0.0, 1.0):
            Ts = []
            for c in coins:
                F = features(c, streak, tf, model)
                cfg = dict(price=(0.80, 0.98), veto=veto, spread=2.0, streak=streak, hours=hrs)
                Ts.append(first_entry(F, conditions(F, cfg, use_hours=hrs is not None), F['K'] - ef, F['K'] - et))
            T = pd.concat(Ts); T = T[T.win.isin([0, 1])]
            rows.append(summary(T, f'{name} | model={model} veto>{veto:g}c'))
R = pd.DataFrame(rows)
print(R[['variant', 'train_n', 'train_roi%', 'train_t', 'test_n', 'test_win%', 'test_roi%', 'test_t', 'test_per_day']].round(2).to_string(index=False))
