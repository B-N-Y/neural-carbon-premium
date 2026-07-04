#!/usr/bin/env python3
"""
05c_industry_fe_robustness.py  --  Clean within-industry-FE robustness for Table 14.

Following Bolton & Kacperczyk (2023), all variables are demeaned within each
industry x month cell before the Fama--MacBeth procedure, and the CO2 coefficient
in the NEURAL RESIDUAL is compared with vs. without industry demeaning.

This reuses the EXACT fama_macbeth / nw_tstat machinery and panel preprocessing of
scripts/2_neural_cross_sectional.py, so the "No Ind. FE" column reproduces the
neural-residual FMB reported in the main text (validation check).

Replaces the stale results/industry_fe_bolton.csv that had undefined
CREDIBLE / GREENWASH / BROWN interaction rows from an archived greenwashing audit.

Output: results/industry_fe_bolton.csv  (consumed by 8_generate_latex_tables.py -> tab_industry_fe.tex)
"""
import os
import importlib.util
import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL_FILE = os.path.join(BASE, 'data_clean', 'final_monthly_panel_clean.csv')
NEURAL_PRED_FILE = os.path.join(BASE, 'data_clean', 'neural_predicted_returns.csv')
# The table generator (8_generate_latex_tables.py) reads CSVs from results/tables/.
OUT_CSV = os.path.join(BASE, 'results', 'tables', 'industry_fe_bolton.csv')
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
OUT_CSV_RESULTS = os.path.join(BASE, 'results', 'industry_fe_bolton.csv')

# --- reuse the identical fama_macbeth / nw_tstat from the main script ---
spec = importlib.util.spec_from_file_location(
    'ncs', os.path.join(BASE, 'scripts', '04_neural_cross_sectional.py'))
ncs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ncs)
fama_macbeth = ncs.fama_macbeth

BOLTON_CHARS = ['SIZE', 'BM', 'ROE_PCT', 'MOM', 'VOLAT', 'INVEST_A',
                'LEVERAGE', 'HHI', 'IO', 'LOG_PPE', 'SALESGR']
CO2_MEASURES = [
    ('LOG_CO2_TOTAL',   r'Total CO$_2$'),
    ('LOG_SCOPE1',      r'Scope 1'),
    ('LOG_SCOPE2',      r'Scope 2'),
    ('LOG_SCOPE3',      r'Scope 3'),
    ('CARBON_INTENSITY', r'CO$_2$ Intensity'),
    ('DELTA_CO2',       r'$\Delta$CO$_2$'),
]


def prep_panel():
    """Identical preprocessing to 2_neural_cross_sectional.main()."""
    df = pd.read_csv(PANEL_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
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
    return df


def demean_within_industry_month(df, cols):
    """Bolton (2023): subtract the industry x month mean from each variable."""
    d = df.copy()
    g = d.groupby(['Industry', 'YearMonth'])
    for c in cols:
        d[c] = d[c] - g[c].transform('mean')
    return d


def coef_t_p(res, key):
    if res is None or key not in res:
        return np.nan, np.nan, np.nan
    coef, t = res[key]['coef'], res[key]['t']
    T = res['_T']
    p = 2 * stats.t.sf(abs(t), max(T - 1, 1))
    return coef, t, p


def main():
    df = prep_panel()
    base = df.dropna(subset=['MonthlyReturn', 'NEURAL_PRED', 'LOG_CO2_TOTAL', 'Industry'] + BOLTON_CHARS).copy()
    print(f"resid sample: {len(base):,} firm-months, {base['YearMonth'].nunique()} months, "
          f"{base['Industry'].nunique()} industries")

    rows = []
    for co2, label in CO2_MEASURES:
        s = base.dropna(subset=[co2]).copy()
        indep = [co2] + BOLTON_CHARS

        # No industry FE
        r_no = fama_macbeth(s, 'NEURAL_RESID_PCT', indep)
        c_no, t_no, p_no = coef_t_p(r_no, co2)

        # With industry FE = demean all model variables within industry x month
        sd = demean_within_industry_month(s, ['NEURAL_RESID_PCT'] + indep)
        r_fe = fama_macbeth(sd, 'NEURAL_RESID_PCT', indep)
        c_fe, t_fe, p_fe = coef_t_p(r_fe, co2)

        rows.append({
            'Variable': label,
            'NoIndFE_coef': c_no, 'NoIndFE_t': t_no, 'NoIndFE_p': p_no,
            'IndFE_coef': c_fe, 'IndFE_t': t_fe, 'IndFE_p': p_fe,
            'Delta_T': t_fe - t_no,
        })
        print(f"  {label:16s}  no-FE: gamma={c_no:+.4f} t={t_no:+.2f}   "
              f"Ind-FE: gamma={c_fe:+.4f} t={t_fe:+.2f}   dT={t_fe - t_no:+.2f}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    out.to_csv(OUT_CSV_RESULTS, index=False)
    print(f"\nwrote {OUT_CSV}")


if __name__ == '__main__':
    main()
