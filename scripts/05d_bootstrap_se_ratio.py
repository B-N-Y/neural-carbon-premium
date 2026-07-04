#!/usr/bin/env python3
"""
5d_bootstrap_se_ratio.py -- Block-bootstrap vs Newey-West SE ratio for the
generated-regressor (neural-residual) Fama--MacBeth CO2 coefficients.

Because the neural residual is a generated regressor, we verify that the
Newey--West standard errors of the CO2 coefficient are not downward-biased:
for each CO2 measure we compare the NW SE of the monthly gamma series with a
moving-block-bootstrap SE (block length 6, B=1000). A ratio near/above 1.0
indicates NW SEs are not understated.

Reuses the identical panel preprocessing / feature set of
2_neural_cross_sectional.py.

Output: results/tables/bootstrap_vs_nw.csv  (se_ratio column feeds
        BootstrapSEratio_min / _max macros in 8_generate_latex_tables.py)
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL_FILE = os.path.join(BASE, 'data_clean', 'final_monthly_panel_clean.csv')
NEURAL_PRED_FILE = os.path.join(BASE, 'data_clean', 'neural_predicted_returns.csv')
OUT_CSV = os.path.join(BASE, 'results', 'tables', 'bootstrap_vs_nw.csv')
OUT_CSV_RESULTS = os.path.join(BASE, 'results', 'bootstrap_vs_nw.csv')
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
os.makedirs(os.path.dirname(OUT_CSV_RESULTS), exist_ok=True)

BOLTON_CHARS = ['SIZE', 'BM', 'ROE_PCT', 'MOM', 'VOLAT', 'INVEST_A',
                'LEVERAGE', 'HHI', 'IO', 'LOG_PPE', 'SALESGR']
CO2_MEASURES = [('LOG_CO2_TOTAL', 'Total CO2'), ('LOG_SCOPE1', 'Scope 1'),
                ('LOG_SCOPE2', 'Scope 2'), ('CARBON_INTENSITY', 'CO2 Intensity'),
                ('DELTA_CO2', 'Emission growth')]
MAX_LAG = 6      # Newey-West lags
BLOCK = 6        # moving-block-bootstrap block length (months)
B = 1000
SEED = 42


def nw_se(gammas, max_lag=MAX_LAG):
    """Newey-West standard error of the mean of a coefficient time series."""
    g = np.asarray(gammas, float)
    T = len(g)
    mu = g.mean()
    dm = g - mu
    v = np.sum(dm ** 2) / T
    for lag in range(1, max_lag + 1):
        w = 1 - lag / (max_lag + 1)
        v += 2 * w * np.sum(dm[lag:] * dm[:-lag]) / T
    return np.sqrt(v / T)


def mbb_se(gammas, block=BLOCK, B=B, seed=SEED):
    """Moving-block-bootstrap SE of the FMB mean (resampling calendar months)."""
    g = np.asarray(gammas, float)
    T = len(g)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(T / block))
    starts_max = T - block
    means = np.empty(B)
    for b in range(B):
        starts = rng.integers(0, starts_max + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:T]
        means[b] = g[idx].mean()
    return means.std(ddof=1)


def monthly_gammas(df, dep, indep):
    """Monthly cross-sectional OLS; return the series of the first regressor's coef."""
    out = []
    for m in sorted(df['YearMonth'].unique()):
        c = df[df['YearMonth'] == m][[dep] + indep].dropna()
        if len(c) < 30:
            continue
        X = sm.add_constant(c[indep].values)
        try:
            res = sm.OLS(c[dep].values, X).fit()
            out.append(res.params[1])   # first indep = CO2 measure
        except Exception:
            continue
    return np.array(out)


def main():
    df = pd.read_csv(PANEL_FILE)
    df['RET_PCT'] = df['MonthlyReturn'] * 100
    df['INVEST_A'] = df['INVEST_A'].abs()
    for v, p in [('INVEST_A', 0.025), ('BM', 0.025), ('LEVERAGE', 0.025),
                 ('MOM', 0.005), ('VOLAT', 0.005)]:
        lo, hi = df[v].quantile(p), df[v].quantile(1 - p)
        df[v] = df[v].clip(lo, hi)
    lo, hi = df['ROE'].quantile(0.025), df['ROE'].quantile(0.975)
    df['ROE_PCT'] = df['ROE'].clip(lo, hi) * 100
    npred = pd.read_csv(NEURAL_PRED_FILE)
    df = pd.merge(df, npred, on=['Ticker', 'YearMonth'], how='left')
    df['NEURAL_RESID_PCT'] = (df['MonthlyReturn'] - df['NEURAL_PRED']) * 100

    base = df.dropna(subset=['NEURAL_PRED', 'LOG_CO2_TOTAL'] + BOLTON_CHARS).copy()

    rows = []
    for co2, label in CO2_MEASURES:
        if co2 not in base.columns:
            continue
        s = base.dropna(subset=[co2]).copy()
        g = monthly_gammas(s, 'NEURAL_RESID_PCT', [co2] + BOLTON_CHARS)
        if len(g) < 10:
            continue
        nw = nw_se(g)
        bb = mbb_se(g)
        ratio = bb / nw if nw > 1e-15 else np.nan
        rows.append({'Variable': label, 'gamma_mean': g.mean(),
                     'NW_SE': nw, 'Block_SE': bb, 'se_ratio': ratio, 'T': len(g)})
        print(f"  {label:16s}  NW_SE={nw:.4f}  Block_SE={bb:.4f}  ratio={ratio:.2f}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    out.to_csv(OUT_CSV_RESULTS, index=False)
    print(f"\nse_ratio range: {out['se_ratio'].min():.2f} to {out['se_ratio'].max():.2f}")
    print(f"wrote {OUT_CSV}")


if __name__ == '__main__':
    main()
