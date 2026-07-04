"""
12_amihud_illiquidity.py -- Construct Amihud (2002) Illiquidity from Daily Data
=================================================================================
Amihud (2002) ILLIQ = (1/D_m) * sum_{d=1}^{D_m} |r_{i,d}| / DollarVolume_{i,d}

where DollarVolume = Volume * Close_Price.

This is the standard illiquidity proxy in asset pricing
(Amihud, 2002, "Illiquidity and stock returns").

Steps:
  1. Load daily data (Return_1D, Volume, Close)
  2. Compute daily |Return| / DollarVolume
  3. Aggregate to monthly average
  4. Merge into monthly panel
  5. Add to conditioning variable list in script 6

INPUT:  data_clean/final_dataset_filtered.csv  (daily)
        data_clean/final_monthly_panel_clean.csv  (monthly)
OUTPUT: data_clean/final_monthly_panel_clean.csv  (updated with AMIHUD column)
        results/tables/amihud_conditioning.csv
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(PAPER_DIR)

DAILY_FILE = os.path.join(PROJECT_DIR, 'data_clean', 'final_dataset_filtered.csv')
PANEL_FILE = os.path.join(PAPER_DIR, 'data_clean', 'final_monthly_panel_clean.csv')
OUTPUT_DIR = os.path.join(PAPER_DIR, 'results', 'tables')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    # ================================================================
    # 1. LOAD DAILY DATA
    # ================================================================
    print("=" * 70)
    print("  AMIHUD (2002) ILLIQUIDITY CONSTRUCTION")
    print("=" * 70)

    print("\n  Loading daily data...")
    daily = pd.read_csv(DAILY_FILE, usecols=['Ticker', 'Date', 'Return_1D', 'Volume', 'Close'])
    daily['Date'] = pd.to_datetime(daily['Date'])
    print(f"  Daily observations: {len(daily):,}")

    # Filter: need non-zero volume and valid returns
    daily = daily.dropna(subset=['Return_1D', 'Volume', 'Close'])
    daily = daily[daily['Volume'] > 0].copy()
    daily = daily[daily['Close'] > 0].copy()
    print(f"  After filtering: {len(daily):,}")

    # ================================================================
    # 2. COMPUTE DAILY AMIHUD RATIO
    # ================================================================
    # DollarVolume = Volume * Close
    # Amihud_daily = |Return| / DollarVolume
    # Scale by 1e6 to avoid tiny numbers (standard in literature)
    daily['DollarVolume'] = daily['Volume'] * daily['Close']
    daily['Amihud_daily'] = np.abs(daily['Return_1D']) / daily['DollarVolume'] * 1e6

    # Remove extreme outliers (top 0.1% within each month)
    daily['YearMonth'] = daily['Date'].dt.to_period('M').astype(str)
    p999 = daily.groupby('YearMonth')['Amihud_daily'].transform(lambda x: x.quantile(0.999))
    daily.loc[daily['Amihud_daily'] > p999, 'Amihud_daily'] = np.nan

    print(f"  Amihud_daily: mean={daily['Amihud_daily'].mean():.4f}, "
          f"median={daily['Amihud_daily'].median():.4f}, "
          f"std={daily['Amihud_daily'].std():.4f}")

    # ================================================================
    # 3. AGGREGATE TO MONTHLY
    # ================================================================
    # Require at least 10 trading days in the month
    monthly_amihud = (
        daily.groupby(['Ticker', 'YearMonth'])
        .agg(
            AMIHUD=('Amihud_daily', 'mean'),
            n_days=('Amihud_daily', 'count')
        )
        .reset_index()
    )
    monthly_amihud = monthly_amihud[monthly_amihud['n_days'] >= 10].copy()
    print(f"\n  Monthly Amihud observations: {len(monthly_amihud):,}")
    print(f"  Unique tickers: {monthly_amihud['Ticker'].nunique():,}")
    print(f"  Period: {monthly_amihud['YearMonth'].min()} to {monthly_amihud['YearMonth'].max()}")

    # Log transform (standard in literature, reduces skewness)
    monthly_amihud['LOG_AMIHUD'] = np.log(monthly_amihud['AMIHUD'] + 1e-10)

    print(f"\n  AMIHUD distribution:")
    print(f"    Mean:   {monthly_amihud['AMIHUD'].mean():.4f}")
    print(f"    Median: {monthly_amihud['AMIHUD'].median():.4f}")
    print(f"    Std:    {monthly_amihud['AMIHUD'].std():.4f}")
    print(f"    P10:    {monthly_amihud['AMIHUD'].quantile(0.10):.4f}")
    print(f"    P90:    {monthly_amihud['AMIHUD'].quantile(0.90):.4f}")

    # ================================================================
    # 4. MERGE INTO MONTHLY PANEL
    # ================================================================
    print("\n  Loading monthly panel...")
    panel = pd.read_csv(PANEL_FILE)
    panel_n_before = len(panel)
    print(f"  Panel before merge: {panel_n_before:,} rows, "
          f"columns: {panel.shape[1]}")

    # Drop existing AMIHUD/LOG_AMIHUD if present (re-run safety)
    for col in ['AMIHUD', 'LOG_AMIHUD']:
        if col in panel.columns:
            panel = panel.drop(columns=[col])

    # Merge
    panel = panel.merge(
        monthly_amihud[['Ticker', 'YearMonth', 'AMIHUD', 'LOG_AMIHUD']],
        on=['Ticker', 'YearMonth'],
        how='left'
    )
    panel_n_after = len(panel)
    assert panel_n_after == panel_n_before, "Merge changed row count!"

    coverage = panel['AMIHUD'].notna().sum()
    print(f"  Amihud coverage: {coverage:,} / {panel_n_after:,} "
          f"({coverage/panel_n_after*100:.1f}%)")

    # Save updated panel
    panel.to_csv(PANEL_FILE, index=False)
    print(f"  Updated panel saved: {PANEL_FILE}")

    # ================================================================
    # 5. VERIFICATION: Correlation with SIZE
    # ================================================================
    # Amihud should be negatively correlated with SIZE (illiquid = small)
    sub = panel.dropna(subset=['AMIHUD', 'SIZE'])
    corr = sub['LOG_AMIHUD'].corr(sub['SIZE'])
    print(f"\n  Verification:")
    print(f"    Corr(LOG_AMIHUD, SIZE) = {corr:.3f}")
    if corr < -0.3:
        print(f"    Expected: negative (illiquid firms are smaller)")
    else:
        print(f"    WARNING: Expected strongly negative correlation")

    # Correlation with IO (institutional ownership)
    if 'IO' in sub.columns:
        corr_io = sub['LOG_AMIHUD'].corr(sub['IO'])
        print(f"    Corr(LOG_AMIHUD, IO)   = {corr_io:.3f}")

    print("\n  Done.")


if __name__ == '__main__':
    main()
