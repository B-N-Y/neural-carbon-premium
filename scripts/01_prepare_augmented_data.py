"""
01_prepare_augmented_data.py — Create augmented dataset for Paper 2 (v3)
=========================================================================

GOAL: Use IDENTICAL control variables as Bolton & Kacperczyk (2021)
      in the LSTM+CA neural model, enabling clean comparison.

Strategy:
  - Start from daily dataset (final_dataset_filtered.csv)
  - Compute LEVERAGE and BETA directly from daily data (better coverage)
  - Merge Bolton controls from monthly Refinitiv panel:
      ROE, INVEST_A, VOLAT, HHI, IO, LOG_PPE, SALESGR
  - NO cross-sectional median fill — only ffill within ticker (standard)
  - Drop firms that lack required features entirely

Features (13):
  Paper 1 base (4):   Log_Size, BookToMarket, Momentum_12M, Mkt-RF
  From daily (2):     LEVERAGE (Debt/TotalAssets), BETA (rolling OLS 252d)
  Bolton monthly (7): ROE, INVEST_A, VOLAT, HHI, IO, LOG_PPE, SALESGR

NOT included:
  - OpProfit_Ratio (Paper 1 FF5 definition — replaced by Bolton's ROE)
  - Asset_Growth   (Paper 1 FF5 definition — replaced by Bolton's INVEST_A)
  - TOBINS_Q       (not in Bolton)
  - FIRM_AGE       (not in Bolton)
  - EPSGR          (Bolton has it but coverage ~55%, too low)
  - DIV_YIELD      (55% genuinely missing)
  - CO₂            (test variable, NOT a control)

FILLNA STRATEGY:
  - ffill within ticker for annual Bolton variables → standard Bolton timing
  - NO cross-sectional median fill
  - NO arbitrary 0-fill
  - Firms without ANY data for a required feature → DROPPED
  - Remaining NaN after ffill: handled in data_loader (z-score → 0 = average)

v3 changes from v2:
  - Replaced OpProfit_Ratio → ROE (Bolton definition: NI/Equity)
  - Replaced Asset_Growth → INVEST_A (Bolton definition: CAPEX/Assets)
  - Removed TOBINS_Q, FIRM_AGE
  - Changed fillna: no more cross-sectional median, only ffill
"""

import pandas as pd
import numpy as np
import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Input files
DAILY_FILE = os.path.join(PROJECT_DIR, 'data_clean', 'final_dataset_filtered.csv')
MONTHLY_PANEL = os.path.join(PROJECT_DIR, 'data_clean',
                              'final_monthly_panel_clean.csv')

# Output
OUTPUT_FILE = os.path.join(SCRIPT_DIR, 'final_dataset_augmented_v3.csv')
TRUBA_DATA_PATH = os.path.join(SCRIPT_DIR, 'final_dataset_augmented_v3.csv')

# Bolton controls to merge from monthly panel
# These are computed in build_monthly_panel.py with Bolton's exact definitions:
#   ROE = Net Income / Shareholders Equity
#   INVEST_A = CAPEX / Total Assets
BOLTON_MONTHLY_CONTROLS = [
    'ROE', 'INVEST_A', 'VOLAT', 'HHI', 'IO', 'LOG_PPE', 'SALESGR'
]

# CO₂ variables for later analysis (NOT model features)
CO2_VARS = ['LOG_CO2_TOTAL', 'LOG_SCOPE1', 'LOG_SCOPE2']


def compute_rolling_beta(daily, window=252, min_periods=60):
    """
    Compute CAPM beta for each stock using rolling OLS.
    β = Cov(R_i, R_m) / Var(R_m) over 'window' trading days.
    """
    print(f"\n  Computing BETA (rolling {window}d OLS)...")
    daily = daily.sort_values(['Ticker', 'Date']).copy()

    # Excess returns
    daily['ExRet'] = daily['Log_Return'] - daily['RF'].fillna(0)
    daily['MktExRet'] = daily['Mkt-RF'].fillna(0)

    # Rolling covariance and variance per ticker
    def rolling_beta_group(group):
        ex_ret = group['ExRet']
        mkt = group['MktExRet']

        cov = ex_ret.rolling(window, min_periods=min_periods).cov(mkt)
        var = mkt.rolling(window, min_periods=min_periods).var()
        beta = cov / var.replace(0, np.nan)
        return beta.clip(-3, 5)  # Sanity clip

    daily['BETA'] = daily.groupby('Ticker', group_keys=False).apply(
        lambda g: rolling_beta_group(g)
    )

    n_valid = daily['BETA'].notna().sum()
    n_na = daily['BETA'].isna().sum()
    print(f"    BETA computed: {n_valid:,} valid ({n_valid/len(daily)*100:.1f}%)")
    print(f"    BETA NA: {n_na:,} ({n_na/len(daily)*100:.1f}%) — first {min_periods}d per ticker")
    print(f"    BETA mean: {daily['BETA'].mean():.4f}, std: {daily['BETA'].std():.4f}")

    # Clean up temp columns
    daily.drop(columns=['ExRet', 'MktExRet'], inplace=True)
    return daily


def main():
    print("=" * 70)
    print("PREPARE AUGMENTED DATASET v3 — Bolton-Identical Controls")
    print(f"  Daily data:    {DAILY_FILE}")
    print(f"  Monthly panel: {MONTHLY_PANEL}")
    print(f"  Output:        {OUTPUT_FILE}")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Load daily dataset
    # --------------------------------------------------
    print("\n  Loading daily dataset...")
    daily = pd.read_csv(DAILY_FILE)
    daily['Date'] = pd.to_datetime(daily['Date'])
    daily = daily.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    print(f"    {len(daily):,} rows, {daily['Ticker'].nunique()} tickers")
    print(f"    Date range: {daily['Date'].min().date()} → {daily['Date'].max().date()}")

    # --------------------------------------------------
    # 2. Compute LEVERAGE from daily data
    # --------------------------------------------------
    print("\n  Computing LEVERAGE (Debt/TotalAssets) from daily data...")
    daily['LEVERAGE'] = daily['Debt'] / daily['TotalAssets'].replace(0, np.nan)
    daily['LEVERAGE'] = daily['LEVERAGE'].clip(0, 2)  # Cap at 200%
    n_lev = daily['LEVERAGE'].notna().sum()
    print(f"    LEVERAGE: {n_lev:,} valid ({n_lev/len(daily)*100:.1f}%)")
    print(f"    Mean: {daily['LEVERAGE'].mean():.4f}")

    # --------------------------------------------------
    # 3. Compute BETA from daily data (rolling 252d OLS)
    # --------------------------------------------------
    daily = compute_rolling_beta(daily, window=252, min_periods=60)

    # --------------------------------------------------
    # 4. Merge Bolton controls from monthly panel
    # --------------------------------------------------
    daily['YearMonth'] = daily['Date'].dt.to_period('M').astype(str)

    print("\n  Loading monthly panel for Bolton controls...")
    monthly = pd.read_csv(MONTHLY_PANEL)
    print(f"    {len(monthly):,} rows")

    available = [c for c in BOLTON_MONTHLY_CONTROLS if c in monthly.columns]
    co2_available = [c for c in CO2_VARS if c in monthly.columns]
    print(f"    Bolton controls: {len(available)}/{len(BOLTON_MONTHLY_CONTROLS)}")
    print(f"    CO₂ variables: {len(co2_available)}/{len(CO2_VARS)}")

    missing_controls = [c for c in BOLTON_MONTHLY_CONTROLS if c not in monthly.columns]
    if missing_controls:
        print(f"     MISSING from monthly panel: {missing_controls}")

    merge_cols = ['Ticker', 'YearMonth'] + available + co2_available
    monthly_subset = monthly[merge_cols].copy()

    print("\n  Merging Bolton controls into daily data...")
    n_before = len(daily)
    merged = pd.merge(daily, monthly_subset, on=['Ticker', 'YearMonth'], how='left')
    print(f"    Merged: {len(merged):,} rows (was {n_before:,})")

    # Coverage report (BEFORE any fill)
    print("\n  RAW coverage report (before any fill):")
    print(f"    {'LEVERAGE':<15s}: {merged['LEVERAGE'].notna().mean()*100:6.1f}% (from daily)")
    print(f"    {'BETA':<15s}: {merged['BETA'].notna().mean()*100:6.1f}% (from daily)")
    for col in available:
        coverage = merged[col].notna().mean() * 100
        print(f"    {col:<15s}: {coverage:6.1f}% (from monthly)")

    # --------------------------------------------------
    # 5. Fill missing values — ONLY ffill within ticker
    #    NO cross-sectional median, NO arbitrary fills
    # --------------------------------------------------
    print("\n  Filling missing values (ffill within ticker ONLY)...")

    all_fill_cols = ['LEVERAGE', 'BETA'] + available
    for col in all_fill_cols:
        n_miss_before = merged[col].isna().sum()
        if n_miss_before == 0:
            continue

        # Step 1: Forward-fill within ticker (standard Bolton timing)
        merged[col] = merged.groupby('Ticker')[col].ffill()
        n_after_ffill = merged[col].isna().sum()

        # Step 2: Backfill within ticker (only for start of series)
        merged[col] = merged.groupby('Ticker')[col].bfill()
        n_after_bfill = merged[col].isna().sum()

        # NO Step 3 — no cross-sectional median, no arbitrary fill
        # Remaining NaN = firms that never have this data → will be dropped

        print(f"    {col:<15s}: {n_miss_before:>10,} NA → "
              f"ffill:{n_after_ffill:>8,} → bfill:{n_after_bfill:>8,} → "
              f"REMAINING: {n_after_bfill:>6,}")

    # --------------------------------------------------
    # 6. Drop firms without required features
    # --------------------------------------------------
    print("\n  Dropping firms without required features...")

    # Paper 1 base features (already in daily data)
    base_features = ['Log_Size', 'BookToMarket', 'Momentum_12M', 'Mkt-RF']
    daily_computed = ['LEVERAGE', 'BETA']
    bolton_monthly = [c for c in available if c in merged.columns]

    all_features = base_features + daily_computed + bolton_monthly

    n_tickers_before = merged['Ticker'].nunique()

    # For each feature, find tickers that have at least SOME non-null values
    tickers_with_all = set(merged['Ticker'].unique())
    for col in all_features:
        tickers_col = set(merged[merged[col].notna()]['Ticker'].unique())
        dropped = tickers_with_all - tickers_col
        if dropped:
            print(f"    {col}: dropping {len(dropped)} tickers with no data")
        tickers_with_all = tickers_with_all & tickers_col

    # Filter to tickers with all features
    merged = merged[merged['Ticker'].isin(tickers_with_all)].copy()
    n_tickers_after = merged['Ticker'].nunique()
    print(f"    Tickers: {n_tickers_before} → {n_tickers_after} "
          f"(dropped {n_tickers_before - n_tickers_after})")

    # --------------------------------------------------
    # 7. Coverage report AFTER filtering
    # --------------------------------------------------
    print(f"\n  Coverage report AFTER filtering:")
    for f in all_features:
        n_na = merged[f].isna().sum()
        pct = n_na / len(merged) * 100
        print(f"    {f:<20s}: {n_na:>8,} NaN ({pct:5.1f}%)")

    # --------------------------------------------------
    # 8. Feature list and statistics
    # --------------------------------------------------
    print(f"\n  Total features for LSTM+CA: {len(all_features)}")
    for i, f in enumerate(all_features):
        if f in base_features:
            origin = "Paper 1 (daily)"
        elif f in daily_computed:
            origin = "Computed from daily"
        else:
            origin = "Bolton (monthly Refinitiv)"
        vals = merged[f].dropna()
        print(f"    {i+1:2d}. {f:<20s} [{origin}] "
              f"mean={vals.mean():>10.4f}, std={vals.std():>10.4f}")

    # --------------------------------------------------
    # 9. Save metadata
    # --------------------------------------------------
    meta = {
        'version': 'v3_bolton_identical',
        'all_features': all_features,
        'base_features': base_features,
        'daily_computed': daily_computed,
        'bolton_features': bolton_monthly,
        'co2_vars': co2_available,
        'n_features': len(all_features),
        'fillna_strategy': 'ffill_bfill_within_ticker_only',
        'notes': [
            'ROE = Net Income / Equity (Bolton definition, NOT FF5 OpProfit_Ratio)',
            'INVEST_A = CAPEX / Assets (Bolton definition, NOT FF5 Asset_Growth)',
            'EPSGR excluded due to low coverage (~55%)',
            'Mkt-RF included as conditioning variable (replaces Bolton time FE)',
            'No cross-sectional median imputation',
            'Firms without any data for a feature are dropped entirely',
        ]
    }
    meta_file = os.path.join(SCRIPT_DIR, 'feature_meta.json')
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Feature metadata saved: {meta_file}")

    # Also save to truba directory
    truba_meta = os.path.join(os.path.dirname(SCRIPT_DIR), 'truba', 'feature_meta.json')
    if os.path.exists(os.path.dirname(truba_meta)):
        with open(truba_meta, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"  Feature metadata also saved: {truba_meta}")

    # --------------------------------------------------
    # 10. Save augmented dataset
    # --------------------------------------------------
    print(f"\n  Saving augmented dataset...")
    keep_cols = (['Date', 'Ticker', 'Log_Return', 'RF', 'YearMonth'] +
                 all_features + co2_available)
    for extra in ['Volume', 'MarketCap', 'Close', 'TRBC Economic Sector Name']:
        if extra in merged.columns:
            keep_cols.append(extra)

    # Only keep columns that exist
    keep_cols = [c for c in keep_cols if c in merged.columns]

    output = merged[keep_cols].copy()
    output.to_csv(OUTPUT_FILE, index=False)
    size_mb = os.path.getsize(OUTPUT_FILE) / 1e6
    print(f"  Saved: {OUTPUT_FILE}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Rows: {len(output):,}")
    print(f"  Tickers: {output['Ticker'].nunique()}")
    print(f"  Features: {len(all_features)}")
    print(f"\n  NOTE: Upload to TRUBA at: {TRUBA_DATA_PATH}")
    print("  Done!")


if __name__ == "__main__":
    main()
