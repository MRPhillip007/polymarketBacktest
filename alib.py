import numpy as np, pandas as pd, os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
COINS = ['btc', 'eth', 'sol', 'xrp']
def fee(p): return 0.07 * p * (1 - p)          # taker fee per share (Polymarket crypto: C*0.07*p*(1-p))
def load(coin, tf='15m'):
    z = np.load(f'{D}/panel_{coin}.npz' if tf == '15m' else f'{D}/panel{tf}_{coin}.npz', allow_pickle=True)
    return {k: z[k] for k in z.files}
def cl_t(x, g):
    """mean, cluster-robust t (clusters g)"""
    x = np.asarray(x, float); g = np.asarray(g)
    n = len(x)
    if n < 3: return np.nan, np.nan, n
    mu = x.mean(); s = pd.Series(x - mu).groupby(g).sum().values
    se = np.sqrt((s ** 2).sum()) / n
    return mu, mu / se if se > 0 else np.nan, n

def logit_fit(X, y, groups, iters=50):
    """logistic regression (Newton) with cluster-robust (sandwich) SEs. X must include a constant column."""
    X = np.asarray(X, float); y = np.asarray(y, float); b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ b)); W = p * (1 - p)
        H = X.T @ (X * W[:, None]); g = X.T @ (y - p)
        step = np.linalg.solve(H + 1e-9 * np.eye(len(b)), g); b += step
        if np.abs(step).max() < 1e-8: break
    p = 1 / (1 + np.exp(-X @ b)); H = X.T @ (X * (p * (1 - p))[:, None]); Hi = np.linalg.inv(H)
    sc = pd.DataFrame(X * (y - p)[:, None]).groupby(np.asarray(groups)).sum().values
    V = Hi @ (sc.T @ sc) @ Hi
    return b, b / np.sqrt(np.diag(V))
