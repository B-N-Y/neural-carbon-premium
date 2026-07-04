"""
00_preprocess_panel.py — Panel Data Preprocessing

Academic-standard preprocessing for Fama-MacBeth cross-sectional regressions.
Follows Bolton & Kacperczyk (2021) and standard empirical asset pricing conventions.

STEPS:
  1. Sample Filters (penny stocks, micro-caps, minimum observations)
  2. Outlier Winsorization (1st/99th percentile, cross-sectional per month)
  3. Variable Transformations (already done in build_monthly_panel, verified here)
  4. Missing Data Diagnostics
  5. Summary Statistics (Table 1 draft)

INPUT:  data_clean/final_monthly_panel.csv
OUTPUT: data_clean/final_monthly_panel_clean.csv
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data_clean')
INPUT_FILE = os.path.join(DATA_DIR, 'final_monthly_panel.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'final_monthly_panel_clean.csv')


# ============================================================
# STEP 1: SAMPLE FILTERS
# ============================================================
def apply_sample_filters(df):
    """
    Bolton (2021) style sample filters:
    - Remove penny stocks (price < $1)
    - Remove micro-cap firms (MarketCap < $10M)
    - Require minimum trading days
    - Remove extreme monthly returns (> 200% or < -90%)
    """
    print("\n" + "=" * 70)
    print("STEP 1: SAMPLE FILTERS")
    print("=" * 70)
    n0 = len(df)

    # 1a. Micro-cap filter (subsumes penny stocks)
    mask_micro = df['MarketCap_EOM'].notna() & (df['MarketCap_EOM'] < 10e6)
    n_micro = mask_micro.sum()
    df = df[~mask_micro].copy()
    print(f"  Micro-caps (MktCap < $10M):    {n_micro:>8,} rows removed")

    # 1c. Minimum trading days (already filtered at 15 in build script)
    mask_days = df['TradingDays'] < 15
    n_days = mask_days.sum()
    df = df[~mask_days].copy()
    print(f"  Low trading days (< 15):       {n_days:>8,} rows removed")

    # 1d. Extreme returns
    mask_ext = (df['MonthlyReturn'] > 2.0) | (df['MonthlyReturn'] < -0.9)
    n_ext = mask_ext.sum()
    df = df[~mask_ext].copy()
    print(f"  Extreme returns (>200%/<-90%): {n_ext:>8,} rows removed")

    print(f"\n  Before: {n0:,} → After: {len(df):,} ({len(df)/n0*100:.1f}%)")
    print(f"  Tickers: {df['Ticker'].nunique():,}")
    return df


# ============================================================
# STEP 2: WINSORIZATION
# ============================================================
def winsorize_variables(df):
    """
    Cross-sectional winsorization at 1st/99th percentiles per month.
    This is the standard approach in Bolton (2021), Fama-French (1993), etc.
    Winsorize per month — not globally — to handle time-varying distributions.
    """
    print("\n" + "=" * 70)
    print("STEP 2: CROSS-SECTIONAL WINSORIZATION (1%/99% per month)")
    print("=" * 70)

    # Variables to winsorize
    continuous_vars = [
        # Dependent
        'MonthlyReturn',
        # Carbon
        'LOG_CO2_TOTAL', 'LOG_SCOPE1', 'LOG_SCOPE2', 'LOG_SCOPE3',
        'LOG_EST_CO2', 'CARBON_INTENSITY', 'DELTA_CO2',
        # Bolton controls
        'SIZE', 'BM', 'MOM', 'VOLAT', 'BETA', 'ROE',
        'INVEST_A', 'SALESGR', 'LOG_PPE', 'HHI', 'IO',
        # Extended controls
        'LEVERAGE', 'RD_A', 'DIV_YIELD', 'ANALYSTS', 'TOBINS_Q', 'FIRM_AGE',
        # ESG
        'ESG_SCORE', 'ENV_SCORE',
        # Neural betas
        'LF1_beta', 'LF2_beta', 'LF3_beta', 'LF4_beta', 'LF5_beta',
    ]

    existing_vars = [v for v in continuous_vars if v in df.columns]

    for var in existing_vars:
        before_mean = df[var].mean()
        before_std = df[var].std()

        # Winsorize per month at 1%/99%
        df[var] = df.groupby('YearMonth')[var].transform(
            lambda x: x.clip(
                lower=x.quantile(0.01),
                upper=x.quantile(0.99)
            )
        )

        after_mean = df[var].mean()
        after_std = df[var].std()
        pct_change = abs(after_std - before_std) / before_std * 100 if before_std > 0 else 0

        if pct_change > 5:  # Only print if winsorization had meaningful impact
            print(f"  {var:25s}: std {before_std:.4f} → {after_std:.4f} ({pct_change:.1f}% change)")

    print(f"\n  Winsorized {len(existing_vars)} variables at 1%/99% per month")
    return df


# ============================================================
# STEP 3: VERIFY TRANSFORMATIONS
# ============================================================
def verify_transformations(df):
    """
    Verify that log transformations and ratios are correctly specified.
    """
    print("\n" + "=" * 70)
    print("STEP 3: TRANSFORMATION VERIFICATION")
    print("=" * 70)

    checks = {
        'LOG_CO2_TOTAL': {'min_expected': 0, 'transform': 'log1p'},
        'LOG_SCOPE1':    {'min_expected': 0, 'transform': 'log1p'},
        'SIZE':          {'min_expected': 0, 'transform': 'log'},
        'LOG_PPE':       {'min_expected': 0, 'transform': 'log'},
        'BM':            {'min_expected': 0, 'transform': 'ratio'},
        'ROE':           {'min_expected': -5, 'transform': 'ratio'},
        'INVEST_A':      {'min_expected': -2, 'transform': 'ratio'},
        'LEVERAGE':      {'min_expected': 0, 'transform': 'ratio'},
    }

    all_ok = True
    for var, spec in checks.items():
        if var not in df.columns:
            continue
        vals = df[var].dropna()
        if len(vals) == 0:
            continue

        issues = []
        if vals.min() < spec['min_expected'] - 0.01:
            issues.append(f"min={vals.min():.4f} < expected {spec['min_expected']}")
        if np.isinf(vals).any():
            issues.append(f"{np.isinf(vals).sum()} inf values")
        if vals.std() == 0:
            issues.append("zero variance")

        status = "" if not issues else ""
        extra = f" [{'; '.join(issues)}]" if issues else ""
        print(f"  {status} {var:20s}: mean={vals.mean():.4f}, "
              f"std={vals.std():.4f}, "
              f"[{vals.quantile(0.01):.4f}, {vals.quantile(0.99):.4f}]{extra}")
        if issues:
            all_ok = False

    if all_ok:
        print("\n   All transformations verified")
    return df


# ============================================================
# STEP 4: MISSING DATA DIAGNOSTICS
# ============================================================
def missing_data_report(df):
    """
    Comprehensive missing data report for regression planning.
    """
    print("\n" + "=" * 70)
    print("STEP 4: MISSING DATA DIAGNOSTICS")
    print("=" * 70)

    # Core regression variables — what's the effective sample?
    bolton_core = ['MonthlyReturn', 'LOG_CO2_TOTAL', 'SIZE', 'BM',
                   'MOM', 'VOLAT', 'BETA', 'ROE', 'INVEST_A',
                   'SALESGR', 'LOG_PPE', 'HHI', 'IO']

    neural_add = [c for c in ['LF1_beta', 'LF2_beta', 'LF3_beta', 'LF4_beta', 'LF5_beta']
                  if c in df.columns]

    sbti_add = ['D_SBTI', 'ESG_SCORE']

    # Bolton model effective sample
    bolton_mask = df[bolton_core].notna().all(axis=1)
    n_bolton = bolton_mask.sum()
    t_bolton = df.loc[bolton_mask, 'Ticker'].nunique()

    # Neural augmented
    if neural_add:
        neural_mask = bolton_mask & df[neural_add].notna().all(axis=1)
        n_neural = neural_mask.sum()
        t_neural = df.loc[neural_mask, 'Ticker'].nunique()
    else:
        n_neural = 0
        t_neural = 0

    # SBTi analysis
    sbti_mask = bolton_mask & df[sbti_add].notna().all(axis=1)
    n_sbti = sbti_mask.sum()

    print(f"\n  EFFECTIVE SAMPLE SIZES:")
    print(f"  {'-'*50}")
    print(f"  Bolton Baseline (Model 1):   {n_bolton:>8,} obs, {t_bolton:>5} tickers")
    print(f"  Neural Augmented (Model 3):  {n_neural:>8,} obs, {t_neural:>5} tickers{' ( no neural betas yet)' if not neural_add else ''}")
    print(f"  SBTi Analysis (Model 6):     {n_sbti:>8,} obs")
    print(f"  {'-'*50}")

    # Time coverage
    if n_bolton > 0:
        bolton_dates = df.loc[bolton_mask, 'Date']
        print(f"\n  Bolton sample period: {bolton_dates.min()} → {bolton_dates.max()}")
        print(f"  Months covered: {df.loc[bolton_mask, 'YearMonth'].nunique()}")

    if n_neural > 0:
        neural_dates = df.loc[neural_mask, 'Date']
        print(f"  Neural sample period: {neural_dates.min()} → {neural_dates.max()}")

    # Per-variable missing
    print(f"\n  VARIABLE-LEVEL COVERAGE:")
    all_vars = bolton_core + neural_add + ['ESG_SCORE', 'D_SBTI', 'LEVERAGE', 'RD_A', 'ANALYSTS']
    for v in all_vars:
        if v in df.columns:
            n = df[v].notna().sum()
            pct = n / len(df) * 100
            bar = "=" * int(pct / 5)
            print(f"  {v:25s}: {n:>8,} ({pct:5.1f}%) {bar}")

    return df


# ============================================================
# STEP 5: SUMMARY STATISTICS (TABLE 1 DRAFT)
# ============================================================
def generate_summary_stats(df):
    """
    Generate summary statistics table (Table 1 of the paper).
    """
    print("\n" + "=" * 70)
    print("STEP 5: SUMMARY STATISTICS (TABLE 1 DRAFT)")
    print("=" * 70)

    stats_vars = [
        ('MonthlyReturn', 'Monthly Return'),
        ('LOG_CO2_TOTAL', 'log(1+CO2 Total)'),
        ('LOG_SCOPE1', 'log(1+Scope 1)'),
        ('LOG_SCOPE2', 'log(1+Scope 2)'),
        ('CARBON_INTENSITY', 'CO2/Revenue'),
        ('DELTA_CO2', 'ΔCO2 (YoY)'),
        ('SIZE', 'log(MarketCap)'),
        ('BM', 'Book/Market'),
        ('MOM', 'Momentum (12m)'),
        ('VOLAT', 'Volatility'),
        ('BETA', 'CAPM Beta'),
        ('ROE', 'Return on Equity'),
        ('INVEST_A', 'CAPEX/Assets'),
        ('SALESGR', 'Sales/MktCap'),
        ('LOG_PPE', 'log(PPE)'),
        ('HHI', 'Herfindahl Index'),
        ('IO', 'Inst. Ownership'),
        ('LEVERAGE', 'Debt/Assets'),
        ('RD_A', 'R&D/Assets'),
        ('ANALYSTS', 'Analyst Coverage'),
        ('ESG_SCORE', 'ESG Score'),
        ('ENV_SCORE', 'Env. Pillar Score'),
        ('LF1_beta', 'Neural β_LF1'),
        ('LF2_beta', 'Neural β_LF2'),
        ('LF3_beta', 'Neural β_LF3'),
        ('LF4_beta', 'Neural β_LF4'),
        ('LF5_beta', 'Neural β_LF5'),
    ]

    print(f"\n  {'Variable':<25s} {'N':>8s} {'Mean':>10s} {'Std':>10s} "
          f"{'p1':>10s} {'p25':>10s} {'p50':>10s} {'p75':>10s} {'p99':>10s}")
    print(f"  {'-'*103}")

    for var, label in stats_vars:
        if var not in df.columns:
            continue
        vals = df[var].dropna()
        if len(vals) == 0:
            continue
        print(f"  {label:<25s} {len(vals):>8,} {vals.mean():>10.4f} {vals.std():>10.4f} "
              f"{vals.quantile(0.01):>10.4f} {vals.quantile(0.25):>10.4f} "
              f"{vals.quantile(0.50):>10.4f} {vals.quantile(0.75):>10.4f} "
              f"{vals.quantile(0.99):>10.4f}")

    # SBTi dummy distribution
    print(f"\n  DUMMY VARIABLE FREQUENCIES:")
    dummy_vars = ['D_REPORTED', 'D_SBTI', 'D_TARGETS_SET', 'D_COMMITTED',
                  'D_REMOVED', 'D_WITHDRAWN', 'D_NET_ZERO', 'D_15C_TARGET']
    for v in dummy_vars:
        if v in df.columns:
            n1 = (df[v] == 1).sum()
            n0 = (df[v] == 0).sum()
            pct = n1 / (n1 + n0) * 100 if (n1 + n0) > 0 else 0
            print(f"  {v:25s}: {n1:>8,} (=1, {pct:.1f}%)")

    # Quadrant distribution
    if 'CARBON_QUADRANT' in df.columns:
        print(f"\n  CARBON QUADRANT DISTRIBUTION:")
        for q in ['True Green', 'Greenwashing', 'Silent Green', 'Brown', 'Unknown']:
            n = (df['CARBON_QUADRANT'] == q).sum()
            pct = n / len(df) * 100
            print(f"  {q:25s}: {n:>8,} ({pct:.1f}%)")

    # Correlation: CO2 vs Neural betas (exogeneity check)
    print(f"\n  EXOGENEITY CHECK: CO2 vs Neural Beta correlations")
    if 'LOG_CO2_TOTAL' in df.columns and 'LF1_beta' in df.columns:
        co2_data = df[df['LOG_CO2_TOTAL'].notna() & df['LF1_beta'].notna()]
        if len(co2_data) > 100:
            for k in range(1, 6):
                col = f'LF{k}_beta'
                if col in co2_data.columns:
                    r = co2_data['LOG_CO2_TOTAL'].corr(co2_data[col])
                    print(f"  log(CO2) vs β_LF{k}: r = {r:.4f}")
    else:
        print("  ⏭ Skipped — neural betas not available yet")

    return df


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print(" PANEL PREPROCESSING — Academic Standard")
    print("=" * 70)

    # Load
    print(f"\n  Loading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"  Raw panel: {len(df):,} rows × {df.shape[1]} cols")

    # Step 1: Sample filters
    df = apply_sample_filters(df)

    # Step 2: Winsorization
    df = winsorize_variables(df)

    # Step 3: Verify transformations
    df = verify_transformations(df)

    # Step 4: Missing data diagnostics
    df = missing_data_report(df)

    # Step 5: Summary statistics
    df = generate_summary_stats(df)

    # Save clean panel
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n{'='*70}")
    print(f" SAVED: {OUTPUT_FILE}")
    print(f" Clean panel: {len(df):,} rows × {df.shape[1]} cols")
    print(f" Tickers: {df['Ticker'].nunique():,}")
    print(f" Ready for Fama-MacBeth regressions!")


if __name__ == "__main__":
    main()
