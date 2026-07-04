"""
08_temporal_subsample.py — Temporal Stability of the Carbon Premium
===================================================================
Tests whether the carbon premium and its neural absorption are stable
across pre- and post-COVID subsamples.

METHODOLOGY:
  Identical to script 2 (04_neural_cross_sectional.py):
    - PanelOLS with time fixed effects
    - Double-clustered standard errors (firm × time)
    - Same Bolton control variables
    - Same CO₂ measure (LOG_CO2_TOTAL)

  Split point: December 2020
    Pre-COVID:  2018-02 to 2020-12  (early ESG adoption)
    Post-COVID: 2021-01 to 2024-12  (ESG mainstreaming + regulation)

  Models tested per subsample:
    M1: R ~ CO₂ + Bolton characteristics        (Bolton baseline)
    M7: R ~ CO₂ + Bolton characteristics + NEURAL_PRED  (Neural control)

  The absorption percentage = (1 - |γ_M7|/|γ_M1|) × 100

INPUT:  data_clean/final_monthly_panel_clean.csv
        data_clean/neural_predicted_returns.csv
OUTPUT: results/tables/temporal_subsample.csv
"""
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(SCRIPT_DIR)

PANEL_FILE = os.path.join(PAPER_DIR, 'data_clean', 'final_monthly_panel_clean.csv')
NEURAL_PRED_FILE = os.path.join(PAPER_DIR, 'data_clean', 'neural_predicted_returns.csv')
OUTPUT_DIR = os.path.join(PAPER_DIR, 'results', 'tables')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Split point
SPLIT_DATE = '2020-12'


# ============================================================
# HELPER: Bolton-style PanelOLS
# ============================================================
def run_panelols(df, dep_var, indep_vars, label=''):
    """
    Run PanelOLS with time fixed effects and double-clustered SE.
    
    This mirrors script 2's `pooled_ols()` function exactly:
      - dep_var: dependent variable (RET_PCT = MonthlyReturn × 100)
      - indep_vars: [CO₂_measure] + control variables
      - Index: (Ticker, TimeIdx) for entity/time clustering
      - time_effects=True absorbs time fixed effects
      - cov_type='clustered', cluster_entity=True, cluster_time=True
    
    Returns dict with coefficient, t-stat, R², N, and firm count
    for the FIRST variable in indep_vars (assumed to be the CO₂ measure).
    """
    # Prepare regression data — drop any row with NaN in dep or indep
    reg_cols = ['Ticker', 'TimeIdx', 'YearMonth', dep_var] + indep_vars
    reg = df[reg_cols].dropna(subset=[dep_var] + indep_vars).copy()
    
    if len(reg) < 200:
        print(f"     {label}: too few observations ({len(reg)})")
        return None
    
    try:
        # Set panel index (entity = Ticker, time = TimeIdx)
        reg_p = reg.set_index(['Ticker', 'TimeIdx'])
        
        # PanelOLS with time FE (equivalent to year-month dummies)
        mod = PanelOLS(
            reg_p[dep_var],
            reg_p[indep_vars],
            time_effects=True,
            check_rank=False
        )
        
        # Double-clustered SE (firm × time)
        res = mod.fit(
            cov_type='clustered',
            cluster_entity=True,
            cluster_time=True
        )
        
        # Extract CO₂ coefficient (first variable)
        co2_var = indep_vars[0]
        
        return {
            'coef': res.params[co2_var],
            't': res.tstats[co2_var],
            'within_r2': res.rsquared,
            'n': int(res.nobs),
            'firms': int(reg['Ticker'].nunique()),
            'months': int(reg['YearMonth'].nunique()),
        }
    except Exception as e:
        print(f"     {label}: {e}")
        return None


def sig(t_val):
    """Significance stars."""
    at = abs(t_val) if not np.isnan(t_val) else 0
    if at >= 2.576:
        return '***'
    elif at >= 1.960:
        return '**'
    elif at >= 1.645:
        return '*'
    return ''


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 80)
    print("=  TEMPORAL SUBSAMPLE ANALYSIS: PRE- vs POST-COVID")
    print(f"=  Split: {SPLIT_DATE}")
    print("=  Method: PanelOLS, Time FE, Double-Clustered SE (Firm × Time)")
    print("=" * 80)
    
    # ----------------------------------------------------------
    # 1. Load and prepare data (same as script 2)
    # ----------------------------------------------------------
    print("\n Loading panel data...")
    df = pd.read_csv(PANEL_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    df['RET_PCT'] = df['MonthlyReturn'] * 100
    df['TimeIdx'] = df['Date'].astype(np.int64) // 10**9
    
    print(f"  Panel: {len(df):,} obs, {df['Ticker'].nunique()} tickers")
    print(f"  Period: {df['YearMonth'].min()} to {df['YearMonth'].max()}")
    
    # Merge neural predictions
    print("  Loading neural predictions...")
    npred = pd.read_csv(NEURAL_PRED_FILE)
    df = pd.merge(df, npred[['Ticker', 'YearMonth', 'NEURAL_PRED']], 
                  on=['Ticker', 'YearMonth'], how='left')
    df['NEURAL_PRED_PCT'] = df['NEURAL_PRED'] * 100
    print(f"  Neural prediction coverage: {df['NEURAL_PRED'].notna().sum():,}")
    
    # ----------------------------------------------------------
    # 2. Define variable groups (identical to script 2)
    # ----------------------------------------------------------
    # Bolton control variables — same as script 2's bolton_chars
    # Note: script 2 uses ROE_PCT and INVEST_A; we use the same set
    # but some may not be available. We'll use the available subset.
    bolton_chars_full = ['SIZE', 'BM', 'ROE_PCT', 'MOM', 'VOLAT', 'INVEST_A',
                         'LEVERAGE', 'HHI', 'IO', 'LOG_PPE', 'SALESGR']
    
    # Apply same winsorization as script 2
    if 'INVEST_A' in df.columns:
        df['INVEST_A'] = df['INVEST_A'].abs()
    for v, p in [('INVEST_A', 0.025), ('BM', 0.025), ('LEVERAGE', 0.025),
                 ('MOM', 0.005), ('VOLAT', 0.005)]:
        if v in df.columns:
            lo, hi = df[v].quantile(p), df[v].quantile(1 - p)
            df[v] = df[v].clip(lo, hi)
    if 'ROE' in df.columns:
        lo, hi = df['ROE'].quantile(0.025), df['ROE'].quantile(0.975)
        df['ROE_PCT'] = df['ROE'].clip(lo, hi) * 100
    
    # Use only columns that exist
    bolton_chars = [c for c in bolton_chars_full if c in df.columns]
    missing = set(bolton_chars_full) - set(bolton_chars)
    if missing:
        print(f"   Missing Bolton chars (skipped): {missing}")
    
    co2_var = 'LOG_CO2_TOTAL'
    
    # M1 controls: CO₂ + Bolton chars
    m1_controls = [co2_var] + bolton_chars
    # M7 controls: CO₂ + Bolton chars + NEURAL_PRED
    m7_controls = [co2_var] + bolton_chars + ['NEURAL_PRED_PCT']
    
    # ----------------------------------------------------------
    # 3. Define subsamples
    # ----------------------------------------------------------
    # Only use observations where neural predictions exist
    # (to keep M1 and M7 on identical samples for valid comparison)
    sample = df.dropna(subset=m7_controls + ['RET_PCT']).copy()
    
    pre_covid = sample[sample['YearMonth'] <= SPLIT_DATE].copy()
    post_covid = sample[sample['YearMonth'] > SPLIT_DATE].copy()
    
    subsamples = [
        ('Full Sample',  sample),
        ('Pre-COVID',    pre_covid),
        ('Post-COVID',   post_covid),
    ]
    
    print(f"\n  Subsamples:")
    for name, sub in subsamples:
        ym = sub['YearMonth']
        print(f"    {name:15s}: {len(sub):>8,} obs, {sub['Ticker'].nunique():>5} firms, "
              f"{ym.nunique():>3} months ({ym.min()} to {ym.max()})")
    
    # ----------------------------------------------------------
    # 4. Run regressions
    # ----------------------------------------------------------
    print(f"\n{'=' * 90}")
    print("  TEMPORAL SUBSAMPLE RESULTS")
    print(f"  {'Period':<15s} {'Model':<25s} {'γ(CO₂)':>10s} {'t-stat':>8s}     "
          f"{'W-R²':>7s} {'N':>8s} {'Firms':>6s} {'Months':>7s}")
    print(f"  {'-' * 88}")
    
    results = []
    
    for period_name, sub_df in subsamples:
        for model_name, controls in [('M1: Bolton', m1_controls),
                                     ('M7: Bolton+Neural', m7_controls)]:
            
            r = run_panelols(sub_df, 'RET_PCT', controls, 
                             label=f"{period_name} {model_name}")
            
            if r:
                stars = sig(r['t'])
                print(f"  {period_name:<15s} {model_name:<25s} {r['coef']:>+10.4f} "
                      f"{r['t']:>8.2f} {stars:<4s} {r['within_r2']:>7.4f} "
                      f"{r['n']:>8,d} {r['firms']:>6d} {r['months']:>7d}")
                
                results.append({
                    'Period': period_name,
                    'Model': model_name,
                    'gamma': r['coef'],
                    't': r['t'],
                    'within_r2': r['within_r2'],
                    'n': r['n'],
                    'firms': r['firms'],
                    'months': r['months'],
                })
            else:
                print(f"  {period_name:<15s} {model_name:<25s}  — FAILED —")
        
        # Absorption analysis for this subsample
        m1_res = [r for r in results if r['Period'] == period_name 
                  and r['Model'] == 'M1: Bolton']
        m7_res = [r for r in results if r['Period'] == period_name 
                  and r['Model'] == 'M7: Bolton+Neural']
        
        if m1_res and m7_res and abs(m1_res[0]['gamma']) > 1e-10:
            absorption = (1 - abs(m7_res[0]['gamma']) / abs(m1_res[0]['gamma'])) * 100
            print(f"  {'':15s} {'→ Absorption:':25s} {absorption:>+10.1f}%")
        print()
    
    # ----------------------------------------------------------
    # 5. Save results
    # ----------------------------------------------------------
    out_path = os.path.join(OUTPUT_DIR, 'temporal_subsample.csv')
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"  Results saved: {out_path}")
    print(f"  Temporal Subsample Analysis Complete!")

    # ----------------------------------------------------------
    # 6. ROLLING 24-MONTH FAMA-MACBETH
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print("  ROLLING 24-MONTH FAMA-MACBETH CO2 GAMMA")
    print(f"{'='*60}")

    # FMB: for each month, run cross-sectional OLS(R ~ CO2 + Bolton controls)
    # Then average gammas over 24-month rolling windows

    bolton_controls = ['SIZE_L1', 'BM_L1', 'ROE', 'INVEST_A', 'LEVERAGE', 'BETA',
                       'VOLAT', 'HHI', 'IO', 'LOG_PPE', 'SALESGR']
    bolton_controls = [c for c in bolton_controls if c in sample.columns]
    co2_var = 'LOG_CO2_TOTAL'
    xvars = [co2_var] + bolton_controls

    # Step 1: Cross-sectional OLS each month
    import statsmodels.api as sm_

    months_all = sorted(sample['YearMonth'].unique())
    monthly_gammas = []

    for ym in months_all:
        cross = sample[sample['YearMonth'] == ym].copy()
        cross = cross.dropna(subset=['RET_PCT'] + xvars)
        if len(cross) < 30:
            continue
        Y = cross['RET_PCT']
        X = sm_.add_constant(cross[xvars])
        try:
            res = sm_.OLS(Y, X).fit()
            monthly_gammas.append({
                'YearMonth': ym,
                'gamma_co2': res.params[co2_var],
                'se_co2': res.bse[co2_var],
                't_co2': res.tvalues[co2_var],
                'N': len(cross)
            })
        except Exception:
            continue

    gamma_df = pd.DataFrame(monthly_gammas)
    gamma_df['YearMonth'] = pd.to_datetime(gamma_df['YearMonth'])
    gamma_df = gamma_df.sort_values('YearMonth').reset_index(drop=True)

    # Step 2: Rolling 24-month average gamma with NW t-stat
    window = 24
    rolling_results = []

    for i in range(window - 1, len(gamma_df)):
        window_df = gamma_df.iloc[i - window + 1: i + 1]
        gammas = window_df['gamma_co2'].values
        T_w = len(gammas)
        mu = gammas.mean()

        # Newey-West SE (6 lags for monthly, consistent with scripts 2/6)
        dm = gammas - mu
        gamma0 = np.sum(dm**2) / T_w
        nw_var = gamma0
        nw_lags = min(6, T_w - 1)
        for lag in range(1, nw_lags + 1):
            w = 1 - lag / (nw_lags + 1)
            gamma_lag = np.sum(dm[lag:] * dm[:-lag]) / T_w
            nw_var += 2 * w * gamma_lag
        se = np.sqrt(nw_var / T_w)
        t_val = mu / se if se > 0 else 0

        rolling_results.append({
            'YearMonth': window_df.iloc[-1]['YearMonth'],
            'rolling_gamma': mu,
            'rolling_se': se,
            'rolling_t': t_val,
            'rolling_ci_lo': mu - 1.96 * se,
            'rolling_ci_hi': mu + 1.96 * se,
            'window_N_avg': window_df['N'].mean()
        })

    rolling_df = pd.DataFrame(rolling_results)
    rolling_path = os.path.join(OUTPUT_DIR, 'rolling_co2_gamma.csv')
    rolling_df.to_csv(rolling_path, index=False)
    print(f"\n  Rolling gamma series: {len(rolling_df)} windows")
    print(f"  Date range: {rolling_df['YearMonth'].min()} to {rolling_df['YearMonth'].max()}")

    # Summary
    for _, row in rolling_df.iterrows():
        ym = row['YearMonth']
        if ym.month in [1, 7]:  # Print every 6 months
            print(f"    {ym.strftime('%Y-%m')}: gamma={row['rolling_gamma']:+.4f}, "
                  f"t={row['rolling_t']:.2f}{sig(row['rolling_t'])}, "
                  f"CI=[{row['rolling_ci_lo']:.4f}, {row['rolling_ci_hi']:.4f}]")

    print(f"  Saved: {rolling_path}")


if __name__ == '__main__':
    main()
