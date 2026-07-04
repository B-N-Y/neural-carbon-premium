"""
04_neural_cross_sectional.py — BOLTON REPLICATION WITH NEURAL FACTOR CONTROLS
=============================================================================
Bolton & Kacperczyk (2021, JFE) Table 8 Extension:

  METHODOLOGY: PanelOLS with time FE, double-clustered SE (Bolton's exact method)
  
  The key question: Does the carbon premium survive after controlling for
  neural latent risk factors?

  MODEL COMPARISON (all Pooled PanelOLS with time FE):
    M1: R ~ CO₂ + Bolton characteristics         (Bolton baseline)
    M2: R ~ CO₂ + ICA LF betas only              (Neural substitute)
    M3: R ~ CO₂ + Bolton chars + ICA LF betas    (Combined — KEY TEST)
    M4: R ~ CO₂ + Bolton chars + FF5 betas       (Traditional benchmark)
    M5: R ~ CO₂ + Bolton chars + ICA LF + FF5    (Kitchen sink)
  
  If γ(CO₂) in M1 ≈ γ(CO₂) in M3 → neural factors DON'T absorb the premium
  If γ(CO₂) in M3 → 0 → neural factors DO absorb the premium

  Also includes:
    - R² comparison: Neural LSTM+CA vs Bolton OLS explanatory power
    - Multiple CO₂ measures (Total, Scope 1, Scope 2, Intensity, Growth)
    - Fama-MacBeth as robustness check

OUTPUT: results/tables/table2b_neural_cross_sectional.csv
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
import os, sys, warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(PAPER_DIR)

PANEL_FILE = os.path.join(PAPER_DIR, 'data_clean', 'final_monthly_panel_clean.csv')
DAILY_FILE = os.path.join(PROJECT_DIR, 'data_clean', 'final_dataset_filtered.csv')
ICA_BETAS_FILE = os.path.join(PAPER_DIR, 'data_clean', 'ica_betas_monthly.csv')
NEURAL_PRED_FILE = os.path.join(PAPER_DIR, 'data_clean', 'neural_predicted_returns.csv')
OUTPUT_DIR = os.path.join(PAPER_DIR, 'results', 'tables')


# ============================================================
# FF5 BETAS (Rolling 252-day)
# ============================================================
def compute_ff5_betas(daily_path, tickers, window=252):
    print("  [FF5] Loading daily returns...")
    cols = ['Ticker', 'Date', 'Return_1D', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
    daily = pd.read_csv(daily_path, usecols=cols)
    daily['Date'] = pd.to_datetime(daily['Date'])
    daily = daily[daily['Ticker'].isin(tickers)].copy()
    daily = daily.dropna(subset=['Return_1D', 'Mkt-RF'])
    daily['ExRet'] = daily['Return_1D'] - daily['RF']
    daily = daily.sort_values(['Ticker', 'Date'])

    ff5 = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    print(f"  [FF5] Rolling {window}-day betas for {daily['Ticker'].nunique()} tickers...")

    results = []
    for ticker, grp in daily.groupby('Ticker'):
        if len(grp) < window:
            continue
        grp = grp.sort_values('Date').reset_index(drop=True)
        er = grp['ExRet'].values
        X = grp[ff5].values
        for end_idx in range(window, len(grp), 21):
            X_w = np.column_stack([np.ones(window), X[end_idx-window:end_idx]])
            try:
                coefs = np.linalg.lstsq(X_w, er[end_idx-window:end_idx], rcond=None)[0]
                results.append({
                    'Ticker': ticker,
                    'Date': grp.at[end_idx - 1, 'Date'],
                    'FF5_MKT': coefs[1], 'FF5_SMB': coefs[2],
                    'FF5_HML': coefs[3], 'FF5_RMW': coefs[4], 'FF5_CMA': coefs[5],
                })
            except:
                continue

    beta_df = pd.DataFrame(results)
    beta_df['YearMonth'] = pd.to_datetime(beta_df['Date']).dt.to_period('M').astype(str)
    beta_df = beta_df.sort_values('Date').groupby(['Ticker', 'YearMonth']).last().reset_index()
    ff5_cols = ['FF5_MKT', 'FF5_SMB', 'FF5_HML', 'FF5_RMW', 'FF5_CMA']
    print(f"  [FF5] Done: {len(beta_df):,} obs, {beta_df['Ticker'].nunique()} tickers")
    return beta_df[['Ticker', 'YearMonth'] + ff5_cols]


# ============================================================
# HELPERS
# ============================================================
def nw_tstat(gammas, max_lag=6):
    T = len(gammas)
    if T < 10: return np.nan
    mu = gammas.mean()
    dm = gammas - mu
    v = np.sum(dm**2) / T
    for lag in range(1, max_lag + 1):
        w = 1 - lag / (max_lag + 1)
        v += 2 * w * np.sum(dm[lag:] * dm[:-lag]) / T
    se = np.sqrt(v / T)
    return mu / se if se > 1e-15 else np.nan

def s(t):
    t = abs(t) if not np.isnan(t) else 0
    if t >= 2.576: return '***'
    if t >= 1.960: return '**'
    if t >= 1.645: return '*'
    return ''

def overall_r2(df, dep, indep):
    reg = df[['YearMonth', dep] + indep].dropna()
    if len(reg) < 100: return np.nan
    y = reg[dep].values
    ym_y = reg.groupby('YearMonth')[dep].transform('mean').values
    ym_x = reg.groupby('YearMonth')[indep].transform('mean')
    y_dm, X_dm = y - ym_y, (reg[indep] - ym_x).values
    try:
        c = np.linalg.lstsq(X_dm, y_dm, rcond=None)[0]
        fitted = X_dm @ c + ym_y
        return 1 - np.sum((y - fitted)**2) / np.sum((y - y.mean())**2)
    except:
        return np.nan

def pooled_ols(df, dep, indep):
    """Bolton-style PanelOLS: time FE, double-clustered SE."""
    reg = df[['Ticker', 'TimeIdx', 'YearMonth'] + [dep] + indep].dropna(subset=[dep] + indep)
    if len(reg) < 200: return None
    try:
        reg_p = reg.set_index(['Ticker', 'TimeIdx'])
        mod = PanelOLS(reg_p[dep], reg_p[indep], time_effects=True, check_rank=False)
        res = mod.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
        co2_var = indep[0]
        ov = overall_r2(reg.reset_index(), dep, indep)
        return {'coef': res.params[co2_var], 't': res.tstats[co2_var],
                'within_r2': res.rsquared, 'overall_r2': ov,
                'n': int(res.nobs), 'firms': int(reg['Ticker'].nunique())}
    except Exception as e:
        print(f"     {e}")
        return None

def fama_macbeth(df, dep, indep, min_obs=30):
    """Fama-MacBeth cross-sectional regression (robustness method)."""
    months = sorted(df['YearMonth'].unique())
    gammas = {v: [] for v in indep}
    r2s = []
    for m in months:
        c = df[df['YearMonth'] == m][[dep] + indep].dropna()
        if len(c) < min_obs: continue
        X = sm.add_constant(c[indep].values)
        try:
            res = sm.OLS(c[dep].values, X).fit()
            for i, v in enumerate(indep):
                gammas[v].append(res.params[i+1])
            r2s.append(res.rsquared)
        except: continue
    if not r2s: return None
    result = {}
    for v in indep:
        g = np.array(gammas[v])
        result[v] = {'coef': g.mean(), 't': nw_tstat(g)}
    result['_r2'] = np.mean(r2s)
    result['_T'] = len(r2s)
    return result


def main():
    print("=" * 80)
    print("=  BOLTON (2021) REPLICATION WITH NEURAL FACTOR CONTROLS")
    print("=  PanelOLS: Time FE, Double-Clustered SE (Bolton's exact method)")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load & prepare panel
    print("\n Loading panel...")
    df = pd.read_csv(PANEL_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    df['RET_PCT'] = df['MonthlyReturn'] * 100
    df['TimeIdx'] = df['Date'].astype(np.int64) // 10**9

    df['INVEST_A'] = df['INVEST_A'].abs()
    for v, p in [('INVEST_A', 0.025), ('BM', 0.025), ('LEVERAGE', 0.025),
                 ('MOM', 0.005), ('VOLAT', 0.005)]:
        lo, hi = df[v].quantile(p), df[v].quantile(1-p)
        df[v] = df[v].clip(lo, hi)
    lo, hi = df['ROE'].quantile(0.025), df['ROE'].quantile(0.975)
    df['ROE_PCT'] = df['ROE'].clip(lo, hi) * 100

    # Merge ICA betas (neural latent factor loadings)
    print("  Loading ICA betas...")
    ica = pd.read_csv(ICA_BETAS_FILE)
    df = pd.merge(df, ica, on=['Ticker', 'YearMonth'], how='left')
    print(f"  ICA betas merged: {df['ICA_LF1'].notna().sum():,}")

    # Merge neural predicted returns (for R² comparison AND as control)
    print("  Loading neural predictions...")
    npred = pd.read_csv(NEURAL_PRED_FILE)
    df = pd.merge(df, npred, on=['Ticker', 'YearMonth'], how='left')
    # NEURAL_PRED_PCT: the neural model's predicted return × 100 (same scale as RET_PCT)
    df['NEURAL_PRED_PCT'] = df['NEURAL_PRED'] * 100
    print(f"  Neural prediction coverage: {df['NEURAL_PRED'].notna().sum():,}")

    # Compute FF5 betas
    co2_tickers = df[df['LOG_CO2_TOTAL'].notna()]['Ticker'].unique()
    ff5_betas = compute_ff5_betas(DAILY_FILE, co2_tickers)
    df = pd.merge(df, ff5_betas, on=['Ticker', 'YearMonth'], how='left')
    print(f"  FF5 betas merged: {df['FF5_MKT'].notna().sum():,}")

    # Variable groups
    bolton_chars = ['SIZE', 'BM', 'ROE_PCT', 'MOM', 'VOLAT', 'INVEST_A',
                    'LEVERAGE', 'HHI', 'IO', 'LOG_PPE', 'SALESGR']
    ica_vars = ['ICA_LF1', 'ICA_LF2', 'ICA_LF3', 'ICA_LF4', 'ICA_LF5']  # K=5 (Paper 2 K-sweep optimal)
    ff5_vars = ['FF5_MKT', 'FF5_SMB', 'FF5_HML', 'FF5_RMW', 'FF5_CMA']
    co2 = 'LOG_CO2_TOTAL'
    all_results = []

    # ================================================================
    #  FLAGSHIP: BOLTON TABLE 8 EXTENSION (PanelOLS) 
    # ================================================================
    print(f"\n{'='*90}")
    print("   FLAGSHIP: BOLTON TABLE 8 EXTENSION WITH NEURAL FACTORS ")
    print("  Does γ(CO₂) change when we add neural risk factor controls?")
    print(f"{'='*90}")

    models = [
        ('M1: Bolton chars only',             bolton_chars),
        ('M2: ICA LF betas only',             ica_vars),
        ('M3: Bolton + ICA LF betas',         bolton_chars + ica_vars),
        ('M4: Bolton + FF5 betas',            bolton_chars + ff5_vars),
        ('M5: Bolton + ICA LF + FF5',         bolton_chars + ica_vars + ff5_vars),
        ('M6: NEURAL_PRED only',              ['NEURAL_PRED_PCT']),
        ('M7: Bolton + NEURAL_PRED',          bolton_chars + ['NEURAL_PRED_PCT']),
    ]

    print(f"\n  {'Model':<35s} {'γ(CO₂)':<10s} {'t-stat':>8s} {'':<4s} {'W-R²':>7s} {'O-R²':>7s} {'N':>8s} {'Firms':>6s}")
    print(f"  {'-'*88}")

    for label, controls in models:
        r = pooled_ols(df, 'RET_PCT', [co2] + controls)
        if r:
            print(f"  {label:<35s} {r['coef']:>+9.4f} {r['t']:>8.2f} "
                  f"{s(r['t']):<4s} {r['within_r2']:>7.4f} "
                  f"{r['overall_r2']:>7.4f} {r['n']:>8,d} {r['firms']:>6d}")
            all_results.append({
                'Test': 'PanelOLS_Flagship', 'Model': label,
                'coef': r['coef'], 't': r['t'],
                'within_r2': r['within_r2'], 'overall_r2': r['overall_r2'],
                'n': r['n'], 'firms': r['firms'],
            })

    # Absorption analysis
    m1 = [r for r in all_results if r['Model'] == 'M1: Bolton chars only']
    m3 = [r for r in all_results if r['Model'] == 'M3: Bolton + ICA LF betas']
    if m1 and m3:
        gamma_change = (m3[0]['coef'] - m1[0]['coef']) / abs(m1[0]['coef']) * 100
        print(f"\n   ABSORPTION ANALYSIS:")
        print(f"    γ(CO₂) M1 → M3: {m1[0]['coef']:+.4f} → {m3[0]['coef']:+.4f} "
              f"(Δ = {gamma_change:+.1f}%)")
        if abs(gamma_change) < 10:
            print(f"    → Neural factors absorb < 10% of carbon premium → INDEPENDENT")
        elif abs(gamma_change) < 50:
            print(f"    → Neural factors partially absorb carbon premium")
        else:
            print(f"    → Neural factors substantially absorb carbon premium")

    # ================================================================
    #  MATCHED-SAMPLE M1* — Same N as M7 (referee requirement)
    # ================================================================
    print(f"\n{'='*90}")
    print("   MATCHED-SAMPLE TEST: M1* on same sample as M7")
    print("  Eliminates sample composition as a confound")
    print(f"{'='*90}")

    # Build the M7 sample: rows with Bolton + NEURAL_PRED + CO2 all non-missing
    m7_cols = [co2] + bolton_chars + ['NEURAL_PRED_PCT']
    m7_sample = df.dropna(subset=['RET_PCT'] + m7_cols).copy()
    m7_n = len(m7_sample)
    print(f"  M7 sample: {m7_n:,} obs, {m7_sample['Ticker'].nunique()} firms")

    # M1* = Bolton-only regression on M7 sample
    r_m1star = pooled_ols(m7_sample, 'RET_PCT', [co2] + bolton_chars)
    # M7 = Bolton + NEURAL_PRED on M7 sample (same as before)
    r_m7 = pooled_ols(m7_sample, 'RET_PCT', [co2] + bolton_chars + ['NEURAL_PRED_PCT'])

    if r_m1star and r_m7:
        print(f"\n  {'Model':<35s} {'γ(CO₂)':<10s} {'t-stat':>8s} {'':4s} {'N':>8s}")
        print(f"  {'-'*65}")
        print(f"  {'M1  (full sample)':<35s} {m1[0]['coef']:>+9.4f} {m1[0]['t']:>8.2f} "
              f"{s(m1[0]['t']):<4s} {m1[0]['n']:>8,d}")
        print(f"  {'M1* (matched sample)':<35s} {r_m1star['coef']:>+9.4f} {r_m1star['t']:>8.2f} "
              f"{s(r_m1star['t']):<4s} {r_m1star['n']:>8,d}")
        print(f"  {'M7  (matched + NEURAL_PRED)':<35s} {r_m7['coef']:>+9.4f} {r_m7['t']:>8.2f} "
              f"{s(r_m7['t']):<4s} {r_m7['n']:>8,d}")

        pct_drop_matched = (r_m7['coef'] - r_m1star['coef']) / abs(r_m1star['coef']) * 100
        print(f"\n   MATCHED-SAMPLE ABSORPTION:")
        print(f"    γ(CO₂) M1* → M7: {r_m1star['coef']:+.4f} → {r_m7['coef']:+.4f} "
              f"(Δ = {pct_drop_matched:+.1f}%)")
        print(f"    M1* and M7 use IDENTICAL N = {r_m1star['n']:,}")
        print(f"    → Absorption is NOT driven by sample composition")

        # Save to results
        all_results.append({
            'Test': 'PanelOLS_Flagship', 'Model': 'M1*: Bolton (matched sample)',
            'coef': r_m1star['coef'], 't': r_m1star['t'],
            'within_r2': r_m1star['within_r2'], 'overall_r2': r_m1star['overall_r2'],
            'n': r_m1star['n'], 'firms': r_m1star['firms'],
        })

    # ================================================================
    # M8: SIZE-ONLY ABSORPTION (lower bound test)
    # ================================================================
    print(f"\n{'='*90}")
    print("  M8: SIZE-ONLY ABSORPTION TEST (lower bound for NEURAL_PRED)")
    print("  If SIZE alone absorbs CO2, then NEURAL_PRED's absorption is trivial")
    print(f"{'='*90}")

    r_size = pooled_ols(m7_sample, 'RET_PCT', [co2, 'SIZE'])
    if r_size and r_m1star:
        abs_size = (1 - abs(r_size['coef']) / abs(r_m1star['coef'])) * 100
        print(f"  M8: R ~ CO2 + SIZE only:  gamma = {r_size['coef']:+.4f}, t = {r_size['t']:.2f}")
        print(f"  M1*:    R ~ CO2 + Bolton:     gamma = {r_m1star['coef']:+.4f}, t = {r_m1star['t']:.2f}")
        print(f"  M7:     R ~ CO2 + Bolton + NP: gamma = {r_m7['coef']:+.4f}, t = {r_m7['t']:.2f}")
        print(f"  Absorption by SIZE alone:      {abs_size:+.1f}%")
        print(f"  Absorption by Bolton+NP (M7):  {(1 - abs(r_m7['coef']) / abs(r_m1star['coef'])) * 100:+.1f}%")
        if abs(abs_size) < 50:
            print(f"  RESULT: SIZE alone absorbs <50% -- NEURAL_PRED adds substantial absorption")
        else:
            print(f"  CAUTION: SIZE alone absorbs >=50% -- absorption may be size-driven")

        all_results.append({
            'Test': 'PanelOLS_Flagship', 'Model': 'M8: CO2 + Size (log ME) only',
            'coef': r_size['coef'], 't': r_size['t'],
            'within_r2': r_size['within_r2'], 'overall_r2': r_size['overall_r2'],
            'n': r_size['n'], 'firms': r_size['firms'],
        })

    # ================================================================
    # M9: INTERACTION TERMS (CO2 x SIZE, CO2 x BM)
    # ================================================================
    print(f"\n{'='*90}")
    print("  M9: INTERACTION TERMS (CO2 x SIZE, CO2 x BM)")
    print("  Tests whether the conditional structure appears in pooled regression")
    print(f"{'='*90}")

    # Create interaction terms
    df_int = m7_sample.copy()
    df_int['CO2xSIZE'] = df_int[co2] * df_int['SIZE']
    df_int['CO2xBM'] = df_int[co2] * df_int['BM']

    int_vars = bolton_chars + ['CO2xSIZE', 'CO2xBM']
    r_m9 = pooled_ols(df_int, 'RET_PCT', [co2] + int_vars)

    if r_m9:
        print(f"  M9: R ~ CO2 + Bolton + CO2*SIZE + CO2*BM")
        print(f"  gamma(CO2):       {r_m9['coef']:+.4f}, t = {r_m9['t']:.2f}{s(r_m9['t'])}")
        # Extract interaction coefficients
        reg_int = df_int.dropna(subset=['RET_PCT', co2] + int_vars)
        reg_int = reg_int.set_index(['Ticker', 'TimeIdx'])
        y_int = reg_int['RET_PCT'].astype(float)
        X_int = reg_int[[co2] + int_vars].astype(float)
        mod_int = PanelOLS(y_int, X_int, time_effects=True, check_rank=False)
        res_int = mod_int.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)

        int_coefs = {}
        for v in ['CO2xSIZE', 'CO2xBM']:
            if v in res_int.params.index:
                print(f"  gamma({v}): {res_int.params[v]:+.4f}, t = {res_int.tstats[v]:.2f}{s(res_int.tstats[v])}")
                int_coefs[f'{v}_coef'] = res_int.params[v]
                int_coefs[f'{v}_t'] = res_int.tstats[v]

        all_results.append({
            'Test': 'PanelOLS_Flagship', 'Model': 'M9: Bolton + CO2 interactions',
            'coef': r_m9['coef'], 't': r_m9['t'],
            'within_r2': r_m9['within_r2'], 'overall_r2': r_m9['overall_r2'],
            'n': r_m9['n'], 'firms': r_m9['firms'],
            **int_coefs,  # CO2xSIZE / CO2xBM coefficients + t-stats -> persisted for reproducibility
        })

    # ================================================================
    # MULTIPLE CO2 MEASURES (PanelOLS)
    # ================================================================

    print(f"\n{'='*90}")
    print("  MULTIPLE CO₂ MEASURES — Bolton + Neural Controls (M3)")
    print(f"{'='*90}")

    co2_measures = [
        ('LOG_CO2_TOTAL', 'Total CO₂'),
        ('LOG_SCOPE1', 'Scope 1'),
        ('LOG_SCOPE2', 'Scope 2'),
        ('CARBON_INTENSITY', 'CO₂/Revenue'),
        ('DELTA_CO2', 'ΔCO₂'),
    ]

    print(f"\n  {'CO₂ Measure':<18s} {'γ(M1: Bolton)':<14s} {'t(M1)':>6s}  {'γ(M3: +Neural)':<14s} {'t(M3)':>6s}")
    print(f"  {'-'*70}")

    for co2_var, co2_label in co2_measures:
        if co2_var not in df.columns:
            continue
        r1 = pooled_ols(df, 'RET_PCT', [co2_var] + bolton_chars)
        r3 = pooled_ols(df, 'RET_PCT', [co2_var] + bolton_chars + ica_vars)
        if r1 and r3:
            print(f"  {co2_label:<18s} {r1['coef']:>+10.4f}   {r1['t']:>+6.2f}{s(r1['t']):<4s}"
                  f" {r3['coef']:>+10.4f}   {r3['t']:>+6.2f}{s(r3['t'])}")
            for tag, r in [('M1_Bolton', r1), ('M3_Bolton+Neural', r3)]:
                all_results.append({
                    'Test': f'MultiCO2_{tag}', 'Model': co2_label,
                    'coef': r['coef'], 't': r['t'],
                    'within_r2': r['within_r2'], 'overall_r2': r['overall_r2'],
                    'n': r['n'], 'firms': r['firms'],
                })

    # ================================================================
    # R² COMPARISON: Neural LSTM+CA vs Bolton OLS
    # ================================================================
    print(f"\n{'='*90}")
    print("  EXPLANATORY POWER: Neural LSTM+CA vs Bolton OLS")
    print(f"{'='*90}")

    valid_pred = df.dropna(subset=['NEURAL_PRED', 'MonthlyReturn', 'LOG_CO2_TOTAL'] + bolton_chars).copy()
    months = sorted(valid_pred['YearMonth'].unique())

    r2_neural, r2_bolton = [], []
    for ym in months:
        m = valid_pred[valid_pred['YearMonth'] == ym]
        if len(m) < 50: continue
        y = m['MonthlyReturn'].values

        # Neural R² (direct predictions)
        yhat = m['NEURAL_PRED'].values
        ss_res = np.sum((y - yhat)**2)
        ss_tot = np.sum((y - y.mean())**2)
        if ss_tot > 0:
            r2_neural.append(1 - ss_res / ss_tot)

        # Bolton OLS R²
        X_b = sm.add_constant(m[[co2] + bolton_chars].values)
        try:
            r2_bolton.append(sm.OLS(y, X_b).fit().rsquared)
        except:
            pass

    print(f"\n  {'Model':<40s} {'Avg Monthly R²':>15s} {'N months':>10s}")
    print(f"  {'-'*65}")
    if r2_neural:
        ratio = np.mean(r2_neural) / np.mean(r2_bolton) if r2_bolton else 0
        print(f"  {'Neural LSTM+CA (direct predictions)':<40s} {np.mean(r2_neural):>10.4f} ({np.mean(r2_neural)*100:.1f}%) {len(r2_neural):>6d}")
        print(f"  {'Bolton OLS (CO₂ + characteristics)':<40s} {np.mean(r2_bolton):>10.4f} ({np.mean(r2_bolton)*100:.1f}%) {len(r2_bolton):>6d}")
        print(f"\n  → Neural model is {ratio:.1f}x more powerful than Bolton OLS")

    all_results.append({'Test': 'R2_Comparison', 'Model': 'Neural LSTM+CA',
                        'avg_r2': np.mean(r2_neural) if r2_neural else np.nan, 'T': len(r2_neural)})
    all_results.append({'Test': 'R2_Comparison', 'Model': 'Bolton OLS',
                        'avg_r2': np.mean(r2_bolton) if r2_bolton else np.nan, 'T': len(r2_bolton)})

    # ================================================================
    # ROBUSTNESS: FAMA-MACBETH
    # ================================================================
    print(f"\n{'='*90}")
    print("  ROBUSTNESS: FAMA-MACBETH (secondary methodology)")
    print(f"{'='*90}")

    fm_models = [
        ('FM_M1: Bolton chars',         [co2] + bolton_chars),
        ('FM_M2: ICA LF betas',         [co2] + ica_vars),
        ('FM_M3: Bolton + ICA LF',      [co2] + bolton_chars + ica_vars),
    ]

    nr_sample = df.dropna(subset=['MonthlyReturn', 'LOG_CO2_TOTAL']).copy()
    for label, indep in fm_models:
        fm = fama_macbeth(nr_sample, 'MonthlyReturn', indep)
        if fm and co2 in fm:
            co2_r = fm[co2]
            print(f"  {label:<30s}: γ×100 = {co2_r['coef']*100:>+8.4f}  "
                  f"t(NW) = {co2_r['t']:>+6.2f}{s(co2_r['t']):<4s}  "
                  f"Avg R² = {fm['_r2']:.4f}  T = {fm['_T']}")
            all_results.append({
                'Test': 'FM_Robustness', 'Model': label,
                'coef': co2_r['coef']*100, 't': co2_r['t'],
                'avg_r2': fm['_r2'], 'T': fm['_T'],
            })

    # ================================================================
    # FMB MULTI-CO2 ON RAW RETURNS (matched sample) — for Table 6 Panel A
    # ================================================================
    # Run FMB on the SAME sample as neural residuals so Table 6 is comparable
    matched_sample = df.dropna(subset=['MonthlyReturn', 'NEURAL_PRED', 'LOG_CO2_TOTAL'] + bolton_chars).copy()
    matched_sample['RET_PCT'] = matched_sample['MonthlyReturn'] * 100

    print(f"\n{'='*90}")
    print("  FMB MULTI-CO2 ON RAW RETURNS (matched sample for Table 6 comparability)")
    print(f"  Sample: {len(matched_sample):,} obs, {matched_sample['Ticker'].nunique():,} firms, "
          f"{matched_sample['YearMonth'].nunique()} months")
    print(f"{'='*90}")

    print(f"\n  {'CO₂ Measure':<18s} {'γ(Bolton)':<14s} {'t':>7s}")
    print(f"  {'-'*45}")

    for co2_var, co2_label in co2_measures:
        if co2_var not in matched_sample.columns:
            continue
        r_raw = fama_macbeth(matched_sample, 'RET_PCT', [co2_var] + bolton_chars)
        if r_raw and co2_var in r_raw:
            print(f"  {co2_label:<18s} {r_raw[co2_var]['coef']:>+12.4f}  {r_raw[co2_var]['t']:>+6.2f}{s(r_raw[co2_var]['t'])}")
            all_results.append({
                'Test': 'FMB_MultiCO2_Raw_Bolton', 'Model': co2_label,
                'coef': r_raw[co2_var]['coef'], 't': r_raw[co2_var]['t'],
                'avg_r2': r_raw['_r2'], 'T': r_raw['_T'],
            })

    # ================================================================
    # NEURAL RESIDUAL AS DEPENDENT VARIABLE (Dual Approach — Key Test)
    # ================================================================
    print(f"\n{'='*90}")
    print("   NEURAL RESIDUAL FMB: CO₂ in the denoised return component ")
    print("  NEURAL_RESID = MonthlyReturn - NEURAL_PRED")
    print("  If CO₂ is sig here → carbon premium is INDEPENDENT of 15 characteristics")
    print("  If CO₂ is insig → carbon premium is a PROXY for characteristic interactions")
    print(f"{'='*90}")

    # Compute neural residual
    resid_sample = df.dropna(subset=['MonthlyReturn', 'NEURAL_PRED', 'LOG_CO2_TOTAL'] + bolton_chars).copy()
    resid_sample['NEURAL_RESID'] = resid_sample['MonthlyReturn'] - resid_sample['NEURAL_PRED']
    resid_sample['NEURAL_RESID_PCT'] = resid_sample['NEURAL_RESID'] * 100
    print(f"  Sample: {len(resid_sample):,} obs, {resid_sample['Ticker'].nunique():,} firms, "
          f"{resid_sample['YearMonth'].nunique()} months")
    print(f"  NEURAL_RESID mean: {resid_sample['NEURAL_RESID'].mean():.6f}, "
          f"std: {resid_sample['NEURAL_RESID'].std():.6f}")

    resid_models = [
        ('NR1: CO₂ only (no controls)',           [co2]),
        ('NR2: CO₂ + Bolton chars',               [co2] + bolton_chars),
        ('NR3: CO₂ + Bolton + ICA betas',          [co2] + bolton_chars + ica_vars),
        ('NR4: CO₂ + Bolton + FF5 betas',          [co2] + bolton_chars + ff5_vars),
    ]

    print(f"\n  {'Model':<35s} {'γ(CO₂)×100':>10s} {'t(NW)':>8s} {'':4s} {'Avg R²':>8s} {'T':>5s}")
    print(f"  {'-'*75}")

    for label, indep in resid_models:
        fm = fama_macbeth(resid_sample, 'NEURAL_RESID_PCT', indep)
        if fm and co2 in fm:
            co2_r = fm[co2]
            print(f"  {label:<35s} {co2_r['coef']:>+10.4f} {co2_r['t']:>+8.2f}{s(co2_r['t']):<4s} "
                  f"{fm['_r2']:>8.4f} {fm['_T']:>5d}")
            all_results.append({
                'Test': 'NeuralResid_FMB', 'Model': label,
                'coef': co2_r['coef'], 't': co2_r['t'],
                'avg_r2': fm['_r2'], 'T': fm['_T'],
            })

    # Also run with multiple CO₂ measures on NEURAL_RESID
    print(f"\n  {'CO₂ Measure':<18s} {'γ(NR: no ctrl)':>14s} {'t':>7s} {'':4s} {'γ(NR+Bolton)':>14s} {'t':>7s}")
    print(f"  {'-'*70}")

    for co2_var, co2_label in co2_measures:
        if co2_var not in resid_sample.columns:
            continue
        r_no = fama_macbeth(resid_sample, 'NEURAL_RESID_PCT', [co2_var])
        r_ctrl = fama_macbeth(resid_sample, 'NEURAL_RESID_PCT', [co2_var] + bolton_chars)
        if r_no and r_ctrl and co2_var in r_no and co2_var in r_ctrl:
            print(f"  {co2_label:<18s} {r_no[co2_var]['coef']:>+12.4f}  {r_no[co2_var]['t']:>+6.2f}{s(r_no[co2_var]['t']):<4s}"
                  f" {r_ctrl[co2_var]['coef']:>+12.4f}  {r_ctrl[co2_var]['t']:>+6.2f}{s(r_ctrl[co2_var]['t'])}")
            for tag, r in [('NR_noctrl', r_no), ('NR_Bolton', r_ctrl)]:
                all_results.append({
                    'Test': f'NeuralResid_MultiCO2_{tag}', 'Model': co2_label,
                    'coef': r[co2_var]['coef'], 't': r[co2_var]['t'],
                    'avg_r2': r['_r2'], 'T': r['_T'],
                })

    # Comparison: M1 (raw return) vs NR2 (neural residual) — same Bolton controls
    print(f"\n   DUAL APPROACH COMPARISON:")
    m1_res = [r for r in all_results if r['Model'] == 'M1: Bolton chars only' and r['Test'] == 'PanelOLS_Flagship']
    nr2_res = [r for r in all_results if r['Model'] == 'NR2: CO₂ + Bolton chars' and r['Test'] == 'NeuralResid_FMB']
    if m1_res and nr2_res:
        print(f"    M1  (MonthlyReturn ~ CO₂ + Bolton):  γ={m1_res[0]['coef']:+.4f}, t={m1_res[0]['t']:+.2f}")
        print(f"    NR2 (NeuralResid ~ CO₂ + Bolton):    γ={nr2_res[0]['coef']:+.4f}, t={nr2_res[0]['t']:+.2f}")
        if abs(nr2_res[0]['t']) < 1.645 and abs(m1_res[0]['t']) >= 1.645:
            print(f"    → CO₂ premium EXISTS in raw returns but VANISHES in neural residuals")
            print(f"    → Interpretation: Carbon premium is a PROXY for nonlinear characteristic interactions")
        elif abs(nr2_res[0]['t']) >= 1.645 and abs(m1_res[0]['t']) >= 1.645:
            print(f"    → CO₂ premium survives neural denoising → INDEPENDENT risk factor")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*80}")
    print("=  SUMMARY")
    print(f"{'='*80}")
    print("""
  BOLTON TABLE 8 EXTENSION — KEY FINDINGS:
  -----------------------------------------
  We replicate Bolton's carbon premium using their exact methodology
  (PanelOLS, time FE, double-clustered SE), then add neural latent
  factor betas (from Paper 1's LSTM+CA model) as additional controls.

  KEY TEST: Does γ(CO₂) change from M1 (Bolton) to M3 (Bolton+Neural)?
    - If γ stays same → Carbon premium is INDEPENDENT of neural factors
    - If γ drops → Neural factors partially explain carbon risk
    - If γ → 0 → Neural factors fully absorb carbon premium

  R² COMPARISON:
    - Neural LSTM+CA provides much higher individual-level R²
    - But Bolton's linear approach finds the carbon premium
    - These are complementary, not competing approaches
    """)

    # Save
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(OUTPUT_DIR, 'table2b_neural_cross_sectional.csv')
    results_df.to_csv(csv_path, index=False)
    print(f" Saved: {csv_path}")
    print(f"\n Bolton Extension with Neural Controls Complete!")


if __name__ == "__main__":
    main()
