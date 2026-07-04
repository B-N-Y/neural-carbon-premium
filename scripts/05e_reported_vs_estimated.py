#!/usr/bin/env python3
"""
5e_reported_vs_estimated.py -- Aswani (2024) robustness: is the carbon premium
an artifact of vendor-ESTIMATED emissions, or is it present in SELF-REPORTED data?

Aswani, Raghunandan, and Rajgopal (2024) argue that the carbon premium is a
measurement artifact: data vendors back out unreported emissions from firm
characteristics, so a "premium" on estimated emissions can be mechanical. We
test this directly by estimating the Bolton (M1) premium (i) on the main
self-reported measure (LOG_CO2_TOTAL, self-reported for ~99% of the CO2 sample)
and (ii) on the vendor-estimated measure (LOG_EST_CO2), under identical Bolton
controls and year-month fixed effects.

Reuses the exact panel preprocessing of 04_neural_cross_sectional.py.

Output: results/tables/reported_vs_estimated.csv  (feeds Aswani* macros in
        8_generate_latex_tables.py -> manuscript_values.tex)
"""
import os
import numpy as np
import pandas as pd
from linearmodels import PanelOLS

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL_FILE = os.path.join(BASE, 'data_clean', 'final_monthly_panel_clean.csv')
OUT_CSV = os.path.join(BASE, 'results', 'tables', 'reported_vs_estimated.csv')
OUT_CSV_RES = os.path.join(BASE, 'results', 'reported_vs_estimated.csv')
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
os.makedirs(os.path.dirname(OUT_CSV_RES), exist_ok=True)

BOLTON = ['SIZE', 'BM', 'ROE_PCT', 'MOM', 'VOLAT', 'INVEST_A',
          'LEVERAGE', 'HHI', 'IO', 'LOG_PPE', 'SALESGR']


def prep():
    df = pd.read_csv(PANEL_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    df['RET_PCT'] = df['MonthlyReturn'] * 100
    df['TimeIdx'] = df['Date'].astype(np.int64) // 10**9
    df['INVEST_A'] = df['INVEST_A'].abs()
    for v, p in [('INVEST_A', 0.025), ('BM', 0.025), ('LEVERAGE', 0.025),
                 ('MOM', 0.005), ('VOLAT', 0.005)]:
        lo, hi = df[v].quantile(p), df[v].quantile(1 - p)
        df[v] = df[v].clip(lo, hi)
    lo, hi = df['ROE'].quantile(0.025), df['ROE'].quantile(0.975)
    df['ROE_PCT'] = df['ROE'].clip(lo, hi) * 100
    return df


def m1(df, measure):
    reg = df.dropna(subset=[measure, 'RET_PCT'] + BOLTON)
    reg = reg[['Ticker', 'TimeIdx', measure, 'RET_PCT'] + BOLTON].set_index(['Ticker', 'TimeIdx'])
    res = PanelOLS(reg['RET_PCT'], reg[[measure] + BOLTON],
                   time_effects=True, check_rank=False).fit(
        cov_type='clustered', cluster_entity=True, cluster_time=True)
    return float(res.params[measure]), float(res.tstats[measure]), int(res.nobs)


def main():
    df = prep()
    # self-reported share of the main CO2 measure
    co2rows = df[df['LOG_CO2_TOTAL'].notna()]
    n_rep = int((co2rows['D_REPORTED'] == 1).sum())
    n_tot = int(len(co2rows))
    share = 100.0 * n_rep / n_tot

    g_rep, t_rep, n_repobs = m1(df, 'LOG_CO2_TOTAL')
    g_est, t_est, n_estobs = m1(df, 'LOG_EST_CO2')

    rows = [
        {'measure': 'Reported (LOG_CO2_TOTAL)', 'gamma': g_rep, 't': t_rep, 'N': n_repobs},
        {'measure': 'Estimated (LOG_EST_CO2)', 'gamma': g_est, 't': t_est, 'N': n_estobs},
        {'measure': 'reported_share_pct', 'gamma': share, 't': np.nan, 'N': n_tot},
    ]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    out.to_csv(OUT_CSV_RES, index=False)
    print(f"self-reported share of CO2 sample: {share:.1f}% ({n_rep:,}/{n_tot:,})")
    print(f"Reported  M1: gamma={g_rep:+.4f} t={t_rep:+.2f} N={n_repobs:,}")
    print(f"Estimated M1: gamma={g_est:+.4f} t={t_est:+.2f} N={n_estobs:,}")
    print(f"wrote {OUT_CSV}")


if __name__ == '__main__':
    main()
