"""
06_scope3_robustness.py — Scope 3 Robustness Analysis for Carbon Premium Absorption
=====================================================================================
Bolton & Kacperczyk (2021) analyze Scope 1, 2, and 3 emissions separately and find
that Scope 3 (supply-chain) emissions often command a significant premium.  Our main
analysis excludes Scope 3 due to data quality concerns (vendor-estimated dominance
in LSEG).  This script provides an appendix-level robustness check:

  1. COVERAGE DIAGNOSTICS: Scope 3 vs Scope 1/2 overlap and self-reported share
  2. BOLTON BASELINE (M1): Pooled OLS for Scope 3, same specification as main text
  3. NEURAL ABSORPTION (M7): Bolton + NEURAL_PRED for Scope 3
  4. ABSORPTION COMPARISON: Side-by-side Scope 1, Scope 2, Scope 3, Total CO2
  5. FAMA-MACBETH: NW t-statistics for Scope 3 (robustness method)
  6. NEURAL RESIDUAL FMB: Does Scope 3 carry information orthogonal to LSTM?
  7. LATEX TABLE: Generates tab_scope3_robustness.tex for the manuscript appendix

INPUT:  data_clean/final_monthly_panel_clean.csv
        data_clean/neural_predicted_returns.csv
        data_clean/ica_betas_monthly.csv
OUTPUT: results/tables/tab_scope3_robustness.tex
        results/tables/scope3_robustness.csv
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
import os, sys, warnings
warnings.filterwarnings('ignore')

# ============================================================
# PATHS (same convention as 05_absorption_robustness.py)
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(PAPER_DIR)

PANEL_FILE = os.path.join(PAPER_DIR, 'data_clean', 'final_monthly_panel_clean.csv')
NEURAL_PRED_FILE = os.path.join(PAPER_DIR, 'data_clean', 'neural_predicted_returns.csv')
ICA_BETAS_FILE = os.path.join(PAPER_DIR, 'data_clean', 'ica_betas_monthly.csv')
OUTPUT_DIR = os.path.join(PAPER_DIR, 'results', 'tables')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HELPERS (identical to 05_absorption_robustness.py)
# ============================================================
def run_panel_ols(df, dep_var, indep_vars):
    """Bolton-style PanelOLS with time FE and double-clustered SE."""
    all_vars = [dep_var] + indep_vars
    need_cols = ['Ticker', 'TimeIdx']
    if 'Year' in df.columns:
        need_cols.append('Year')
    reg = df[need_cols + all_vars].dropna(subset=all_vars)
    if len(reg) < 200:
        return None
    reg = reg.set_index(['Ticker', 'TimeIdx'])
    y = reg[dep_var].astype(float)
    X = reg[indep_vars].astype(float)
    try:
        mod = PanelOLS(y, X, time_effects=True, entity_effects=False, check_rank=False)
        res = mod.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
        co2_var = indep_vars[0]
        return {
            'coef': res.params[co2_var],
            't_stat': res.tstats[co2_var],
            'pval': res.pvalues[co2_var],
            'se': res.std_errors[co2_var],
            'ci_lo': res.params[co2_var] - 1.96 * res.std_errors[co2_var],
            'ci_hi': res.params[co2_var] + 1.96 * res.std_errors[co2_var],
            'r2': res.rsquared,
            'n_obs': int(res.nobs),
            'n_firms': int(reg.index.get_level_values(0).nunique()),
        }
    except Exception as e:
        print(f"     PanelOLS error: {e}")
        return None


def nw_tstat(gammas, max_lag=6):
    """Newey-West (1987) t-statistic for mean of time-series coefficients."""
    T = len(gammas)
    if T < 10:
        return np.nan
    mu = gammas.mean()
    dm = gammas - mu
    v = np.sum(dm**2) / T
    for lag in range(1, max_lag + 1):
        w = 1 - lag / (max_lag + 1)
        v += 2 * w * np.sum(dm[lag:] * dm[:-lag]) / T
    se = np.sqrt(v / T)
    return mu / se if se > 1e-15 else np.nan


def run_fama_macbeth(df, dep_var, indep_vars, min_obs=30):
    """Fama-MacBeth cross-sectional regression with NW standard errors."""
    months = sorted(df['YearMonth'].unique())
    gammas = {v: [] for v in indep_vars}
    r2s, ns = [], []
    for m in months:
        cross = df[df['YearMonth'] == m][[dep_var] + indep_vars].dropna()
        if len(cross) < min_obs:
            continue
        y = cross[dep_var].values
        X = sm.add_constant(cross[indep_vars].values)
        try:
            res = sm.OLS(y, X).fit()
            for i, v in enumerate(indep_vars):
                gammas[v].append(res.params[i + 1])
            r2s.append(res.rsquared)
            ns.append(len(cross))
        except:
            continue
    if not r2s:
        return None
    result = {}
    for v in indep_vars:
        g = np.array(gammas[v])
        result[v] = {'coef': g.mean(), 't': nw_tstat(g)}
    result['_r2'] = np.mean(r2s)
    result['_T'] = len(r2s)
    result['_avg_n'] = int(np.mean(ns))
    return result


def sig(t):
    at = abs(t)
    if at > 2.576: return '***'
    if at > 1.960: return '**'
    if at > 1.645: return '*'
    return ''


# ============================================================
# LOAD DATA (same procedure as 05_absorption_robustness.py)
# ============================================================
print("=" * 80)
print("SCOPE 3 ROBUSTNESS ANALYSIS")
print("=" * 80)

print("\n Loading panel...")
df = pd.read_csv(PANEL_FILE, low_memory=False)
df['Date'] = pd.to_datetime(df['Date'])
df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)
if 'Year' not in df.columns:
    df['Year'] = df['Date'].dt.year
df['TimeIdx'] = df['Date'].astype(np.int64) // 10**9

# Standardize — match script 2 / script 5
df['RET_PCT'] = df['MonthlyReturn'] * 100

lo, hi = df['ROE'].quantile(0.025), df['ROE'].quantile(0.975)
df['ROE_PCT'] = df['ROE'].clip(lo, hi) * 100

df['INVEST_A'] = df['INVEST_A'].abs()
for v, p in [('INVEST_A', 0.025), ('BM', 0.025), ('LEVERAGE', 0.025),
             ('MOM', 0.005), ('VOLAT', 0.005)]:
    lo, hi = df[v].quantile(p), df[v].quantile(1-p)
    df[v] = df[v].clip(lo, hi)

# Merge ICA betas
ica = pd.read_csv(ICA_BETAS_FILE)
df = pd.merge(df, ica, on=['Ticker', 'YearMonth'], how='left')

# Merge neural predictions
npred = pd.read_csv(NEURAL_PRED_FILE)
df = pd.merge(df, npred, on=['Ticker', 'YearMonth'], how='left')
df['NEURAL_PRED_PCT'] = df['NEURAL_PRED'] * 100

# Variables (identical to script 5)
bolton_chars = ['SIZE', 'BM', 'ROE_PCT', 'MOM', 'VOLAT', 'INVEST_A',
                'LEVERAGE', 'HHI', 'IO', 'LOG_PPE', 'SALESGR']
ica_vars = ['ICA_LF1', 'ICA_LF2', 'ICA_LF3', 'ICA_LF4', 'ICA_LF5']

# CO2 measures for comparative analysis
co2_measures = [
    ('LOG_CO2_TOTAL', 'Total CO$_2$'),
    ('LOG_SCOPE1',    'Scope 1'),
    ('LOG_SCOPE2',    'Scope 2'),
    ('LOG_SCOPE3',    'Scope 3'),
]

print(f"  Panel: {len(df):,} rows, {df['Ticker'].nunique()} tickers")

# Check columns
for c in bolton_chars + ['RET_PCT', 'NEURAL_PRED_PCT', 'LOG_SCOPE3']:
    if c not in df.columns:
        print(f"   Missing column: {c}")
        sys.exit(1)


# ============================================================
# TEST 1: COVERAGE DIAGNOSTICS
# ============================================================
print(f"\n{'='*80}")
print("  TEST 1: SCOPE 3 COVERAGE DIAGNOSTICS")
print(f"{'='*80}\n")

for col, label in co2_measures:
    n_total = df[col].notna().sum()
    n_tickers = df[df[col].notna()]['Ticker'].nunique()
    n_months = df[df[col].notna()]['YearMonth'].nunique()
    print(f"  {label:15s}: {n_total:>8,} obs, {n_tickers:>5} tickers, {n_months:>4} months")

# Scope 3 overlap with neural predictions
s3_np = df[(df['LOG_SCOPE3'].notna()) & (df['NEURAL_PRED'].notna())]
print(f"\n  Scope 3 + NEURAL_PRED overlap: {len(s3_np):,} obs, {s3_np['Ticker'].nunique()} tickers")

# Self-reported vs estimated (if D_REPORTED column exists)
if 'D_REPORTED' in df.columns:
    s3_reported = df[(df['LOG_SCOPE3'].notna()) & (df['D_REPORTED'] == 1)]
    s3_estimated = df[(df['LOG_SCOPE3'].notna()) & (df['D_REPORTED'] == 0)]
    s3_total = df['LOG_SCOPE3'].notna().sum()
    pct_reported = len(s3_reported) / s3_total * 100 if s3_total > 0 else 0
    print(f"\n  Scope 3 self-reported: {len(s3_reported):,} ({pct_reported:.1f}%)")
    print(f"  Scope 3 estimated:    {len(s3_estimated):,} ({100-pct_reported:.1f}%)")
    print(f"   {'HIGH' if pct_reported < 50 else 'LOW'} vendor-estimation share "
          f"— {'justifies limitation caveat' if pct_reported < 50 else 'acceptable for analysis'}")
else:
    print("\n  D_REPORTED column not found — cannot distinguish reported vs estimated")

# Scope 3 / Scope 1 ratio
s1_s3 = df[(df['LOG_SCOPE1'].notna()) & (df['LOG_SCOPE3'].notna())]
pct_overlap = len(s1_s3) / df['LOG_SCOPE1'].notna().sum() * 100
print(f"\n  Scope 3 coverage relative to Scope 1: {pct_overlap:.1f}%")

# Date range for Scope 3
s3_dates = df[df['LOG_SCOPE3'].notna()]['Date']
print(f"  Scope 3 date range: {s3_dates.min().strftime('%Y-%m')} to {s3_dates.max().strftime('%Y-%m')}")


# ============================================================
# TEST 2: BOLTON BASELINE (M1) & NEURAL ABSORPTION (M7) — ALL SCOPES
# ============================================================
print(f"\n{'='*80}")
print("  TEST 2: ABSORPTION COMPARISON ACROSS SCOPES (M1 vs M7)")
print(f"{'='*80}\n")

print(f"  {'CO₂ Measure':18s} | {'γ(M1)':>8s} {'t(M1)':>7s} {'N(M1)':>8s} | "
      f"{'γ(M7)':>8s} {'t(M7)':>7s} {'N(M7)':>8s} | {'Absorb%':>8s}")
print(f"  {'-'*18}-+-{'-'*26}-+-{'-'*26}-+-{'-'*8}")

all_results = []

for co2_var, co2_label in co2_measures:
    if co2_var not in df.columns:
        print(f"  {co2_label:18s} |  Column not found")
        continue

    # M1: Bolton baseline
    r_m1 = run_panel_ols(df, 'RET_PCT', [co2_var] + bolton_chars)
    # M7: Bolton + NEURAL_PRED
    r_m7 = run_panel_ols(df, 'RET_PCT', [co2_var] + bolton_chars + ['NEURAL_PRED_PCT'])

    if r_m1 and r_m7:
        abs_pct = (1 - r_m7['coef'] / r_m1['coef']) * 100 if abs(r_m1['coef']) > 1e-10 else 0
        print(f"  {co2_label:18s} | {r_m1['coef']:>+8.4f} {r_m1['t_stat']:>+6.2f}{sig(r_m1['t_stat']):1s} "
              f"{r_m1['n_obs']:>8,} | {r_m7['coef']:>+8.4f} {r_m7['t_stat']:>+6.2f}{sig(r_m7['t_stat']):1s} "
              f"{r_m7['n_obs']:>8,} | {abs_pct:>+7.1f}%")

        all_results.append({
            'CO2_var': co2_var, 'CO2_label': co2_label,
            'M1_coef': r_m1['coef'], 'M1_t': r_m1['t_stat'], 'M1_n': r_m1['n_obs'],
            'M1_r2': r_m1['r2'], 'M1_firms': r_m1['n_firms'],
            'M7_coef': r_m7['coef'], 'M7_t': r_m7['t_stat'], 'M7_n': r_m7['n_obs'],
            'M7_r2': r_m7['r2'], 'M7_firms': r_m7['n_firms'],
            'absorption_pct': abs_pct,
        })
    elif r_m1:
        print(f"  {co2_label:18s} | {r_m1['coef']:>+8.4f} {r_m1['t_stat']:>+6.2f}{sig(r_m1['t_stat']):1s} "
              f"{r_m1['n_obs']:>8,} | {'M7 failed':>26s} |      —")
        all_results.append({
            'CO2_var': co2_var, 'CO2_label': co2_label,
            'M1_coef': r_m1['coef'], 'M1_t': r_m1['t_stat'], 'M1_n': r_m1['n_obs'],
            'M1_r2': r_m1['r2'], 'M1_firms': r_m1['n_firms'],
            'M7_coef': np.nan, 'M7_t': np.nan, 'M7_n': np.nan,
            'M7_r2': np.nan, 'M7_firms': np.nan,
            'absorption_pct': np.nan,
        })
    else:
        print(f"  {co2_label:18s} | {'Insufficient data':>26s} | {'—':>26s} |      —")


# ============================================================
# TEST 3: MATCHED-SAMPLE M1* FOR SCOPE 3
# ============================================================
print(f"\n{'='*80}")
print("  TEST 3: MATCHED-SAMPLE M1* vs M7 FOR SCOPE 3")
print("  Eliminates sample composition as a confound (same N for M1* and M7)")
print(f"{'='*80}\n")

co2_s3 = 'LOG_SCOPE3'
m7_cols_s3 = [co2_s3] + bolton_chars + ['NEURAL_PRED_PCT']
m7_sample_s3 = df.dropna(subset=['RET_PCT'] + m7_cols_s3).copy()
m7_n_s3 = len(m7_sample_s3)
m7_firms_s3 = m7_sample_s3['Ticker'].nunique()
print(f"  Scope 3 matched sample: {m7_n_s3:,} obs, {m7_firms_s3} firms")

r_m1star_s3 = run_panel_ols(m7_sample_s3, 'RET_PCT', [co2_s3] + bolton_chars)
r_m7_s3 = run_panel_ols(m7_sample_s3, 'RET_PCT', [co2_s3] + bolton_chars + ['NEURAL_PRED_PCT'])

if r_m1star_s3 and r_m7_s3:
    abs_matched = (1 - r_m7_s3['coef'] / r_m1star_s3['coef']) * 100 \
        if abs(r_m1star_s3['coef']) > 1e-10 else 0
    print(f"  M1* (Bolton only, matched):    γ={r_m1star_s3['coef']:+.4f}  "
          f"t={r_m1star_s3['t_stat']:+.2f}{sig(r_m1star_s3['t_stat'])}  N={r_m1star_s3['n_obs']:,}")
    print(f"  M7  (Bolton + NEURAL_PRED):    γ={r_m7_s3['coef']:+.4f}  "
          f"t={r_m7_s3['t_stat']:+.2f}{sig(r_m7_s3['t_stat'])}  N={r_m7_s3['n_obs']:,}")
    print(f"  Matched-sample absorption:     {abs_matched:+.1f}%")
else:
    print("   Matched-sample test failed (insufficient overlap)")
    abs_matched = np.nan


# ============================================================
# TEST 4: FAMA-MACBETH FOR SCOPE 3 (SECONDARY METHOD)
# ============================================================
print(f"\n{'='*80}")
print("  TEST 4: FAMA-MACBETH CROSS-SECTIONAL REGRESSIONS — SCOPE 3")
print(f"{'='*80}\n")

# FMB on raw returns
fm_s3_raw = run_fama_macbeth(df, 'RET_PCT', [co2_s3] + bolton_chars)
if fm_s3_raw:
    s3_r = fm_s3_raw[co2_s3]
    print(f"  FMB (raw returns):    γ={s3_r['coef']:+.4f}  "
          f"t(NW)={s3_r['t']:+.2f}{sig(s3_r['t'])}  "
          f"R²={fm_s3_raw['_r2']:.4f}  T={fm_s3_raw['_T']}  "
          f"avg N/mo={fm_s3_raw['_avg_n']}")

# FMB with NEURAL_PRED as control
fm_s3_neural = run_fama_macbeth(df, 'RET_PCT', [co2_s3] + bolton_chars + ['NEURAL_PRED_PCT'])
if fm_s3_neural:
    s3_rn = fm_s3_neural[co2_s3]
    print(f"  FMB (+ NEURAL_PRED):  γ={s3_rn['coef']:+.4f}  "
          f"t(NW)={s3_rn['t']:+.2f}{sig(s3_rn['t'])}  "
          f"R²={fm_s3_neural['_r2']:.4f}  T={fm_s3_neural['_T']}  "
          f"avg N/mo={fm_s3_neural['_avg_n']}")

if fm_s3_raw and fm_s3_neural:
    fmb_abs = (1 - fm_s3_neural[co2_s3]['coef'] / fm_s3_raw[co2_s3]['coef']) * 100 \
        if abs(fm_s3_raw[co2_s3]['coef']) > 1e-10 else 0
    print(f"  FMB absorption:       {fmb_abs:+.1f}%")


# ============================================================
# TEST 5: NEURAL RESIDUAL FMB — SCOPE 3
# ============================================================
print(f"\n{'='*80}")
print("  TEST 5: NEURAL RESIDUAL FMB — Does Scope 3 carry orthogonal information?")
print(f"{'='*80}\n")

# Compute neural residual
df['NEURAL_RESID'] = df['RET_PCT'] - df['NEURAL_PRED_PCT']

# FMB: neural residual ~ Scope 3 + Bolton controls
fm_s3_resid = run_fama_macbeth(df, 'NEURAL_RESID', [co2_s3] + bolton_chars)
if fm_s3_resid:
    s3_res = fm_s3_resid[co2_s3]
    print(f"  FMB (neural residual): γ={s3_res['coef']:+.4f}  "
          f"t(NW)={s3_res['t']:+.2f}{sig(s3_res['t'])}  "
          f"R²={fm_s3_resid['_r2']:.4f}  T={fm_s3_resid['_T']}")
    if abs(s3_res['t']) > 1.96:
        print(f"  → Scope 3 carries pricing information ORTHOGONAL to LSTM characteristics")
    else:
        print(f"  → Scope 3 does NOT carry significant orthogonal information")

# For comparison: also run Total CO2 and Scope 1 in neural residuals
for co2_var, co2_label in [('LOG_CO2_TOTAL', 'Total CO2'), ('LOG_SCOPE1', 'Scope 1')]:
    fm_resid = run_fama_macbeth(df, 'NEURAL_RESID', [co2_var] + bolton_chars)
    if fm_resid:
        r = fm_resid[co2_var]
        print(f"  FMB residual ({co2_label:10s}): γ={r['coef']:+.4f}  "
              f"t(NW)={r['t']:+.2f}{sig(r['t'])}")


# ============================================================
# TEST 6: ICA LATENT FACTOR CONTROL (M3) — SCOPE 3
# ============================================================
print(f"\n{'='*80}")
print("  TEST 6: ICA LATENT FACTOR CONTROL (M3) — SCOPE 3")
print(f"{'='*80}\n")

r_m3_s3 = run_panel_ols(df, 'RET_PCT', [co2_s3] + bolton_chars + ica_vars)
if r_m3_s3:
    print(f"  M3 (Bolton + ICA LF): γ={r_m3_s3['coef']:+.4f}  "
          f"t={r_m3_s3['t_stat']:+.2f}{sig(r_m3_s3['t_stat'])}  "
          f"R²={r_m3_s3['r2']:.4f}  N={r_m3_s3['n_obs']:,}")


# ============================================================
# GENERATE LATEX TABLE (tab_scope3_robustness.tex)
# ============================================================
print(f"\n{'='*80}")
print("  GENERATING LATEX TABLE: tab_scope3_robustness.tex")
print(f"{'='*80}\n")


def fmt_coef(c, decimals=4):
    """Format coefficient with significance stars embedded."""
    return f"{c:.{decimals}f}"


def fmt_t(t_val):
    """Format t-stat with significance stars."""
    if np.isnan(t_val):
        return '--'
    stars = sig(t_val)
    return f"{t_val:.2f}^{{{stars}}}" if stars else f"{t_val:.2f}"


tex = []
tex.append(r'\begin{table}[H]')
tex.append(r'\centering')
tex.append(r'\caption{Scope 3 Robustness: Carbon Premium Absorption Across Emission Scopes}')
tex.append(r'\label{tab:scope3_robustness}')
tex.append(r'\small')
tex.append(r'\begin{tabular}{lrrrrr}')
tex.append(r'\toprule')
tex.append(r' & \multicolumn{2}{c}{M1: Bolton} & \multicolumn{2}{c}{M7: Bolton + $\hat{R}^{NN}$} & \\')
tex.append(r'\cmidrule(lr){2-3} \cmidrule(lr){4-5}')
tex.append(r'CO$_2$ Measure & $\gamma$ & $t$ & $\gamma$ & $t$ & $N$ \\')
tex.append(r'\midrule')

for row in all_results:
    label = row['CO2_label']
    m1_c = fmt_coef(row['M1_coef'])
    m1_t = fmt_t(row['M1_t'])
    if not np.isnan(row['M7_coef']):
        m7_c = fmt_coef(row['M7_coef'])
        m7_t = fmt_t(row['M7_t'])
        n_str = f"{row['M7_n']:,}"
    else:
        m7_c, m7_t = '--', '--'
        n_str = f"{row['M1_n']:,}"
    tex.append(f'  {label:20s} & ${m1_c}$ & ${m1_t}$ & ${m7_c}$ & ${m7_t}$ & {n_str} \\\\')

tex.append(r'\bottomrule')
tex.append(r'\end{tabular}')
tex.append(r'\begin{tablenotes}')
tex.append(r'\small')
tex.append(r'\item PanelOLS with year-month FE, double-clustered SE (firm $\times$ year). '
           r'Dependent variable: monthly return $\times$ 100. '
           r'M1: Bolton controls (SIZE, BM, LEV, ROE, INVEST, SALESGR, log(PPE), MOM, VOLAT, HHI, IO). '
           r'M7: Bolton controls + $\hat{R}^{NN}$ (LSTM neural predicted return). '
           r''
           r'Scope 3 emissions in LSEG are predominantly vendor-estimated rather than self-reported; '
           r'results should be interpreted with this measurement caveat. '
           r'$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.')
tex.append(r'\end{tablenotes}')
tex.append(r'\end{table}')

tex_path = os.path.join(OUTPUT_DIR, 'tab_scope3_robustness.tex')
with open(tex_path, 'w') as f:
    f.write('\n'.join(tex))
print(f"   Saved: {tex_path}")


# ============================================================
# SAVE CSV
# ============================================================
csv_path = os.path.join(OUTPUT_DIR, 'scope3_robustness.csv')
pd.DataFrame(all_results).to_csv(csv_path, index=False)
print(f"   Saved: {csv_path}")


# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*80}")
print("  SCOPE 3 ROBUSTNESS — SUMMARY")
print(f"{'='*80}")

s3_row = [r for r in all_results if r['CO2_var'] == 'LOG_SCOPE3']
if s3_row:
    r = s3_row[0]
    print(f"""
  Scope 3 Coverage:
    Total Scope 3 obs:          {df['LOG_SCOPE3'].notna().sum():,}
    Scope 3 + NEURAL_PRED:      {len(s3_np):,} obs, {s3_np['Ticker'].nunique()} tickers

  Scope 3 Bolton Baseline (M1):
    γ = {r['M1_coef']:+.4f},  t = {r['M1_t']:+.2f}{sig(r['M1_t'])}

  Scope 3 Neural Absorption (M7):
    γ = {r['M7_coef']:+.4f},  t = {r['M7_t']:+.2f}{sig(r['M7_t'])}
    Absorption: {r['absorption_pct']:+.1f}%""")

    if abs(r['M1_t']) > 1.96 and abs(r['M7_t']) < 1.96:
        print("\n   RESULT: Scope 3 premium is ABSORBED by neural model (consistent with main text)")
    elif abs(r['M1_t']) > 1.96 and abs(r['M7_t']) > 1.96:
        print("\n   RESULT: Scope 3 premium SURVIVES neural control (differs from main text)")
    elif abs(r['M1_t']) < 1.96:
        print("\n   RESULT: No significant Scope 3 premium in baseline (nothing to absorb)")
    else:
        print("\n   RESULT: Mixed — examine coefficients carefully")

print(f"\n Scope 3 robustness analysis complete!")
print(f" Results saved to: {OUTPUT_DIR}")
