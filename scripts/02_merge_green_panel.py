"""
02_merge_green_panel.py
=======================
Merges CO₂ data + Bolton controls into model predictions panel.
Creates neural residuals for downstream absorption analysis.

Output: results/analysis_panel_green.csv
"""
import pandas as pd
import numpy as np
import os

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GREEN_PATH = os.path.join(PROJECT, 'data_raw', 'green_variables.csv')
RESULTS_DIR = os.path.join(PROJECT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ===============================================================
# LOAD DATA
# ===============================================================

# 1. Load model predictions — Model A (baseline, K=5) daily test predictions
PREDS_FILE = os.path.join(RESULTS_DIR, 'paper2_exports', 'daily_test_predictions.csv')
print(f"Loading daily test predictions from: {PREDS_FILE}")

preds = pd.read_csv(PREDS_FILE, low_memory=False)
preds['Date'] = pd.to_datetime(preds['Date'])
if 'Ticker' in preds.columns:
    preds = preds.rename(columns={'Ticker': 'Instrument'})
preds = preds[['Date', 'Instrument', 'Predicted', 'Actual']]
print(f"  Total test predictions: {preds.shape[0]:,} rows, {preds['Instrument'].nunique()} tickers")
print(f"  Date range: {preds['Date'].min().date()} → {preds['Date'].max().date()}")

# 2. Load emissions/accounting panel (for CO₂ + controls)
PANEL_PATH = os.path.join(PROJECT, 'data_raw', 'emissions_accounting_panel.csv')
panel = pd.read_csv(PANEL_PATH, low_memory=False)
# Panel already has FiscalYear column — use it directly
# Filter to reasonable fiscal years (some entries have FY>2025 from Refinitiv estimates)
panel['FiscalYear'] = pd.to_numeric(panel['FiscalYear'], errors='coerce')
panel = panel[(panel['FiscalYear'] >= 1996) & (panel['FiscalYear'] <= 2025)]
print(f"  Emissions panel: {panel.shape[0]:,} rows, {panel['Instrument'].nunique()} tickers")
print(f"  FiscalYear range: {panel['FiscalYear'].min():.0f} → {panel['FiscalYear'].max():.0f}")

# 3. Load green variables
green = pd.read_csv(GREEN_PATH, low_memory=False)
green['Date'] = pd.to_datetime(green['Date'], errors='coerce')
green['FiscalYear'] = green['Date'].dt.year
print(f"  Green variables: {green.shape[0]:,} rows, {green['Instrument'].nunique()} tickers")

# ===============================================================
# PREPARE CO₂ DATA (from emissions panel, annual → merge by FiscalYear)
# ===============================================================
co2_cols = ['Instrument', 'FiscalYear', 
            'TR.CO2EmissionTotal', 'TR.CO2DirectScope1', 'TR.CO2IndirectScope2',
            'TR.Revenue', 'TR.F.TotAssets', 'TR.CompanyMarketCap',
            'TR.F.DebtTot', 'TR.F.TotLiab', 'TR.ShareholdersEquity', 'TR.WACCBeta',
            'TR.F.PPENetTot', 'TR.CapitalExpenditures']
co2_cols = [c for c in co2_cols if c in panel.columns]

# Convert all numeric columns and deduplicate (keep first per firm-year)
for c in co2_cols:
    if c not in ['Instrument', 'FiscalYear']:
        panel[c] = pd.to_numeric(panel[c], errors='coerce')

# Aggregate: take first non-null value per (Instrument, FiscalYear)
panel_annual = panel[co2_cols].groupby(['Instrument', 'FiscalYear']).first().reset_index()

# Create CARBON_INTENSITY = (Scope1 + Scope2) / Revenue
# NOTE: Do NOT fillna(0) — NULL means "data unavailable", not "zero emissions"
if 'TR.CO2DirectScope1' in panel_annual.columns and 'TR.Revenue' in panel_annual.columns:
    scope1 = pd.to_numeric(panel_annual['TR.CO2DirectScope1'], errors='coerce')
    scope2 = pd.to_numeric(panel_annual.get('TR.CO2IndirectScope2', pd.Series(dtype=float)), errors='coerce')
    # Sum: if both are NaN → NaN. If one is NaN → use the other (conservative)
    co2_total = scope1.add(scope2, fill_value=0)  # NaN+NaN=NaN, 100+NaN=100
    # But require at least Scope1 to exist (main component)
    co2_total = co2_total.where(scope1.notna())
    revenue = pd.to_numeric(panel_annual['TR.Revenue'], errors='coerce')
    panel_annual['CARBON_INTENSITY'] = co2_total / revenue.replace(0, np.nan)
    
    co2_emission = pd.to_numeric(panel_annual['TR.CO2EmissionTotal'], errors='coerce')
    panel_annual['LOG_CO2_TOTAL'] = np.log1p(co2_emission)  # NaN stays NaN

# Create SIZE, BM, LEVERAGE
panel_annual['SIZE'] = np.log(pd.to_numeric(panel_annual['TR.CompanyMarketCap'], errors='coerce').replace(0, np.nan))
total_assets = pd.to_numeric(panel_annual['TR.F.TotAssets'], errors='coerce')
mcap = pd.to_numeric(panel_annual['TR.CompanyMarketCap'], errors='coerce')
panel_annual['BM'] = total_assets / mcap.replace(0, np.nan)
# Leverage: TotLiab / TotAssets (Debt-to-Assets ratio)
# NOTE: TR.ShareholdersEquity has only 4.6% coverage → Debt/Equity impossible
# TotLiab and TotAssets both have ~22.5% coverage → much better
liab = pd.to_numeric(panel_annual['TR.F.TotLiab'], errors='coerce')
panel_annual['LEVERAGE'] = liab / total_assets.replace(0, np.nan)  # NaN stays NaN
ppe = pd.to_numeric(panel_annual.get('TR.F.PPENetTot', pd.Series(dtype=float)), errors='coerce')
panel_annual['LOG_PPE'] = np.log1p(ppe)  # NaN stays NaN
panel_annual['BETA'] = pd.to_numeric(panel_annual.get('TR.WACCBeta', np.nan), errors='coerce')

# GREEN_CAPEX_RATIO = GreenCapex(bool) indicator from green data
# (will be merged separately from green vars)

n_co2 = panel_annual['CARBON_INTENSITY'].notna().sum()
print(f"\n  CO₂ coverage: {n_co2:,} firm-years with CARBON_INTENSITY")

# ===============================================================
# PREPARE GREEN VARIABLES (convert bools, keep key columns)
# ===============================================================
bool_cols = ['TR.GreenCapex', 'TR.PolicyEmissions', 'TR.TargetsEmissions',
             'TR.PolicyEnergyEfficiency', 'TR.PolicyWaterEfficiency',
             'TR.PolicySustainablePackaging',
             'TR.PolicyEnvSupplyChain', 'TR.WasteReductionInitiatives']
for col in bool_cols:
    if col in green.columns:
        green[col] = green[col].map({True: 1, False: 0, 'True': 1, 'False': 0})
        green[col] = pd.to_numeric(green[col], errors='coerce')

green_keep = ['Instrument', 'FiscalYear',
              'TR.GreenCapex', 'TR.PolicyEmissions', 'TR.TargetsEmissions',
              'TR.PolicyEnergyEfficiency', 'TR.PolicyWaterEfficiency',
              'TR.PolicySustainablePackaging',
              'TR.PolicyEnvSupplyChain', 'TR.WasteReductionInitiatives',
              'TR.TRESGInnovationScore', 'TR.TRESGEmissionsScore',
              'TR.TRESGResourceUseScore', 'TR.EnvironmentPillarScore',
              'TR.EnergyUseTotal', 'TR.WasteTotal', 'TR.WasteRecycledTotal',
              'TR.WaterWithdrawalTotal', 'TR.EnvExpenditures']
green_keep = [c for c in green_keep if c in green.columns]
green_sub = green[green_keep].drop_duplicates(subset=['Instrument', 'FiscalYear'])

# ===============================================================
# MERGE: CO₂ + Green → lagged → Predictions
# ===============================================================

# Combine CO₂ panel + green vars on (Instrument, FiscalYear)
annual = pd.merge(panel_annual, green_sub, on=['Instrument', 'FiscalYear'], how='outer')
annual = annual.drop_duplicates(subset=['Instrument', 'FiscalYear'])
print(f"\n  Combined annual data: {annual.shape[0]:,} rows")

# LAG by 1 year: FiscalYear_annual + 1 = FiscalYear_return
annual_lagged = annual.copy()
annual_lagged['FiscalYear'] = annual_lagged['FiscalYear'] + 1

# Rename to _L1
rename_cols = {c: f"{c}_L1" if c not in ['Instrument', 'FiscalYear'] else c 
               for c in annual_lagged.columns}
annual_lagged = annual_lagged.rename(columns=rename_cols)

# Add FiscalYear to predictions (fiscal year alignment: Jul-Jun)
preds['FiscalYear'] = (preds['Date'] - pd.DateOffset(months=6)).dt.year

# Merge
panel_merged = pd.merge(preds, annual_lagged, on=['Instrument', 'FiscalYear'], how='left')
print(f"  Merged panel: {panel_merged.shape[0]:,} rows")

# ===============================================================
# BOLTON CONTROLS from augmented dataset (same as LSTM training input)
# Ensures consistency: FMB regressions use IDENTICAL variables as LSTM
# ===============================================================
AUGMENTED_PATH = os.path.join(PROJECT, 'scripts', 'final_dataset_augmented_v3.csv')
if not os.path.exists(AUGMENTED_PATH):
    # Fallback to truba copy
    AUGMENTED_PATH = os.path.join(PROJECT, 'truba', 'final_dataset_augmented_v3.csv')
print(f"\n  Loading Bolton controls from: {AUGMENTED_PATH}")
bolton_cols = ['Ticker', 'Date', 'ROE', 'INVEST_A', 'VOLAT', 'HHI', 'IO', 'LOG_PPE',
               'SALESGR', 'LEVERAGE', 'BETA']
aug = pd.read_csv(AUGMENTED_PATH, usecols=bolton_cols, low_memory=False)
aug['Date'] = pd.to_datetime(aug['Date'])
aug = aug.rename(columns={'Ticker': 'Instrument'})
aug = aug.drop_duplicates(subset=['Instrument', 'Date'])
print(f"  Augmented: {len(aug):,} rows, {aug['Instrument'].nunique()} tickers")

# Drop old LEVERAGE/BETA from annual merge (use augmented versions for consistency)
for old_col in ['LEVERAGE_L1', 'BETA_L1']:
    if old_col in panel_merged.columns:
        panel_merged = panel_merged.drop(columns=[old_col])

# Merge Bolton controls on Date+Instrument (daily level — exact match)
panel_merged = pd.merge(panel_merged, aug, on=['Instrument', 'Date'], how='left')
bolton_vars = ['ROE', 'INVEST_A', 'VOLAT', 'HHI', 'IO', 'LOG_PPE', 'SALESGR', 'LEVERAGE', 'BETA']
for bv in bolton_vars:
    n = panel_merged[bv].notna().sum()
    print(f"  {bv:<12s}: {n:>10,} valid ({n/len(panel_merged)*100:.1f}%)")
print(f"  Bolton controls merged ")

# ===============================================================
# NEURAL RESIDUAL
# ===============================================================
panel_merged['NEURAL_RESID'] = panel_merged['Actual'] - panel_merged['Predicted']
print(f"\n  Neural residual: mean={panel_merged['NEURAL_RESID'].mean():.6f}, "
      f"std={panel_merged['NEURAL_RESID'].std():.4f}")

# ===============================================================
# SECTOR DATA
# ===============================================================
SECTOR_PATH = os.path.join(PROJECT, 'data_clean', 'final_dataset_filtered.csv')
print(f"\n  Loading sector data from: {SECTOR_PATH}")
sector_df = pd.read_csv(SECTOR_PATH, usecols=['Ticker', 'TRBC Economic Sector Name'], 
                         low_memory=False)
sector_df = sector_df.drop_duplicates('Ticker').rename(
    columns={'Ticker': 'Instrument', 'TRBC Economic Sector Name': 'Sector'})
print(f"  Sectors: {sector_df['Sector'].nunique()} unique, {len(sector_df):,} tickers")
panel_merged = pd.merge(panel_merged, sector_df, on='Instrument', how='left')
panel_merged['Year'] = panel_merged['Date'].dt.year
print(f"  Sector coverage: {panel_merged['Sector'].notna().mean()*100:.1f}%")

# ===============================================================
# SUMMARY
# ===============================================================
print(f"\n{'='*70}")
print("FINAL PANEL SUMMARY")
print(f"{'='*70}")
print(f"Total rows: {panel_merged.shape[0]:,}")
print(f"Unique tickers: {panel_merged['Instrument'].nunique()}")
print(f"Date range: {panel_merged['Date'].min().date()} → {panel_merged['Date'].max().date()}")

co2_n = panel_merged['CARBON_INTENSITY_L1'].notna().sum()
co2_t = panel_merged[panel_merged['CARBON_INTENSITY_L1'].notna()]['Instrument'].nunique()
print(f"CO₂ coverage: {co2_n:,} obs, {co2_t} tickers")

# Key columns for downstream scripts
key_cols = ['Date', 'Instrument', 'Predicted', 'Actual', 'NEURAL_RESID',
            'CARBON_INTENSITY_L1', 'LOG_CO2_TOTAL_L1', 'SIZE_L1', 'BM_L1', 
            'LEVERAGE_L1', 'BETA_L1', 'LOG_PPE_L1',
            'TR.EnvironmentPillarScore_L1',
            'ROE', 'INVEST_A', 'VOLAT', 'HHI', 'IO', 'LOG_PPE',
            'SALESGR', 'LEVERAGE', 'BETA',
            'Sector', 'Year']
key_cols = [c for c in key_cols if c in panel_merged.columns]

print(f"\nKey columns available: {len(key_cols)}")
print(f"  {key_cols}")

# Save
out_path = os.path.join(RESULTS_DIR, 'analysis_panel_green.csv')
panel_merged.to_csv(out_path, index=False)
print(f"\n Saved: {out_path}")
print(f"   Size: {os.path.getsize(out_path)/1e6:.1f} MB")

