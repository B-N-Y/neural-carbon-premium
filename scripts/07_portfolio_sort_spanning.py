"""
07_portfolio_sort_spanning.py — Portfolio Sort & Spanning Tests
================================================================
Non-regression evidence for the carbon premium.

METHODOLOGY:
  Equal-weighted portfolio sorts with Newey-West (1987) t-statistics.
  NO regression, NO generated regressors, NO multicollinearity issues.

  Sort variable information timing:
    - CO₂ emissions: annual Refinitiv data, matched to fiscal year t-1
      (lagged at panel construction time, no look-ahead bias)
    - NEURAL_PRED: out-of-sample prediction from K=5 LSTM+CA model,
      formed using only data up to month t-1

  Portfolio construction:
    - Each month, sort firms into terciles/quintiles by sort variable
    - Compute EQUAL-WEIGHTED returns within each group
    - H-L spread = High group return - Low group return
    - Time-series mean with Newey-West HAC t-stat (max_lag=6, Bartlett kernel)

  FF5 factor aggregation:
    - Daily Fama-French 5 factors from Kenneth French's data library
    - Monthly = sum of unique daily factor returns within each month

TESTS:
  PART A: Single sorts (CO₂, Neural Prediction)
  PART B: Independent 3×3 double sort (CO₂ × Neural Prediction)
  PART C: Spanning regression (CO₂ spread ~ FF5, Neural HmL)
  PART D: Conditional CO₂ premium by NP quintile (5×3 sort)
  PART E: Conditional CO₂ premium by firm characteristic (SIZE, BM, ESG, etc.)
  PART F: Industry-level CO₂ premium (raw + FF5-adjusted)

INPUT:  data_clean/final_monthly_panel_clean.csv
        data_clean/neural_predicted_returns.csv
        data_clean/final_dataset_filtered.csv (daily FF5 factors)
OUTPUT: results/tables/portfolio_sort_results.csv
        results/tables/spanning_test_results.csv
        results/tables/industry_carbon_premium.csv
        results/tables/conditional_carbon_premium.csv
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os, warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(PAPER_DIR)

PANEL_FILE = os.path.join(PAPER_DIR, 'data_clean', 'final_monthly_panel_clean.csv')
NEURAL_PRED_FILE = os.path.join(PAPER_DIR, 'data_clean', 'neural_predicted_returns.csv')
DAILY_FILE = os.path.join(PROJECT_DIR, 'data_clean', 'final_dataset_filtered.csv')
OUTPUT_DIR = os.path.join(PAPER_DIR, 'results', 'tables')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================
def nw_tstat(series, max_lag=6):
    """Newey-West (1987) t-statistic for the mean of a time-series.

    Uses Bartlett kernel weights: w(j) = 1 - j/(max_lag+1).
    Returns (mean, t-stat, T).
    """
    x = series.dropna().values
    T = len(x)
    if T < 10:
        return np.nan, np.nan, T
    mu = x.mean()
    dm = x - mu
    gamma0 = np.sum(dm**2) / T
    nw_var = gamma0
    for lag in range(1, max_lag + 1):
        w = 1 - lag / (max_lag + 1)
        gamma_lag = np.sum(dm[lag:] * dm[:-lag]) / T
        nw_var += 2 * w * gamma_lag
    se = np.sqrt(nw_var / T)
    t = mu / se if se > 1e-15 else np.nan
    return mu, t, T


def sig(t):
    """Significance stars based on absolute t-stat."""
    if np.isnan(t): return ''
    at = abs(t)
    if at >= 2.576: return '***'
    if at >= 1.960: return '**'
    if at >= 1.645: return '*'
    return ''


def compute_co2_spread(data, months, min_firms=20):
    """Compute monthly CO₂ H-L spread within a subsample.

    Each month: sort firms into terciles by LOG_CO2_TOTAL,
    compute equal-weighted returns, return H-L spread series.
    """
    spreads = []
    month_list = []
    for ym in months:
        cross = data[data['YearMonth'] == ym]
        if len(cross) < min_firms:
            continue
        try:
            cross = cross.copy()
            cross['CO2_G'] = pd.qcut(
                cross['LOG_CO2_TOTAL'], 3,
                labels=False, duplicates='drop') + 1
        except ValueError:
            continue
        groups = cross.groupby('CO2_G')['MonthlyReturn'].mean()
        if 1 not in groups.index or 3 not in groups.index:
            continue
        spreads.append(groups[3] - groups[1])
        month_list.append(ym)
    return pd.Series(spreads, index=month_list)


# ============================================================
# LOAD & PREPARE DATA
# ============================================================
print("=" * 80)
print("PORTFOLIO SORT & SPANNING TESTS")
print("=" * 80)

# Monthly panel
print("\n Loading panel...")
df = pd.read_csv(PANEL_FILE, low_memory=False)
df['Date'] = pd.to_datetime(df['Date'])
df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)
df['INVEST_A'] = df['INVEST_A'].abs()

# Merge TRBC Business Sector for the industry-level sort (Part F). The panel's
# own 'Industry' column is the finer TRBC Industry Group, kept for HHI and the
# industry fixed-effects test; the within-sector carbon sort instead uses the
# broader Business Sector (~30 groups) so that peer groups are large enough and
# carbon-coherent (fossil fuels stay separate from renewable/uranium). The
# TRBC classification is derived from LSEG data (proprietary; not redistributed).
SECTOR_FILE = os.path.join(PAPER_DIR, 'data_raw', 'company_sectors_trbc.csv')
_sec = pd.read_csv(SECTOR_FILE)[['Ticker', 'TRBC_Business_Sector']]
df = pd.merge(df, _sec, on='Ticker', how='left')

# Merge neural predictions
npred = pd.read_csv(NEURAL_PRED_FILE)
df = pd.merge(df, npred, on=['Ticker', 'YearMonth'], how='left')

# Subsample: firms with CO₂ AND neural predictions
sample = df.dropna(subset=['MonthlyReturn', 'LOG_CO2_TOTAL', 'NEURAL_PRED']).copy()
print(f"  Full panel: {len(df):,} obs, {df['Ticker'].nunique()} tickers")
print(f"  CO₂ + Neural subsample: {len(sample):,} obs, {sample['Ticker'].nunique()} tickers")
print(f"  Months: {sample['YearMonth'].nunique()} "
      f"({sample['YearMonth'].min()} — {sample['YearMonth'].max()})")

# Monthly FF5 factors — use cache if available, else read from 13GB daily file
print("\n Loading FF5 factors...")
FF5_CACHE = os.path.join(PAPER_DIR, 'data_clean', 'ff5_monthly_cache.csv')
if os.path.exists(FF5_CACHE):
    ff5_monthly = pd.read_csv(FF5_CACHE)
    print(f"  FF5 monthly (from cache): {len(ff5_monthly)} months")
else:
    print("  No cache found, reading from daily file (this may take a while)...")
    daily_cols = ['Date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
    daily = pd.read_csv(DAILY_FILE, usecols=daily_cols)
    daily['Date'] = pd.to_datetime(daily['Date'])
    daily['YearMonth'] = daily['Date'].dt.to_period('M').astype(str)
    daily_unique = daily.drop_duplicates(subset=['Date']).copy()
    ff5_monthly = (daily_unique
                   .groupby('YearMonth')[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']]
                   .sum()
                   .reset_index())
    ff5_monthly.to_csv(FF5_CACHE, index=False)
    print(f"  FF5 monthly: {len(ff5_monthly)} months (cached to {FF5_CACHE})")

months = sorted(sample['YearMonth'].unique())


# ============================================================
# PART A: SINGLE SORTS
# ============================================================
print(f"{'='*80}")
print("  PART A: SINGLE PORTFOLIO SORTS (EW + VW)")
print(f"{'='*80}")


def single_sort(data, sort_var, sort_label, n_groups=3):
    """Sort firms into groups each month by sort_var. EW and VW returns."""
    months_s = sorted(data['YearMonth'].unique())
    port_returns_ew = {g: [] for g in range(1, n_groups + 1)}
    port_returns_ew['H-L'] = []
    port_returns_vw = {g: [] for g in range(1, n_groups + 1)}
    port_returns_vw['H-L'] = []
    month_list = []

    has_mcap = 'MarketCap_EOM' in data.columns

    for ym in months_s:
        cross = data[data['YearMonth'] == ym].copy()
        if len(cross) < n_groups * 10:
            continue
        try:
            cross['Group'] = pd.qcut(
                cross[sort_var], n_groups,
                labels=False, duplicates='drop') + 1
        except ValueError:
            continue

        group_rets_ew = cross.groupby('Group')['MonthlyReturn'].mean()
        if len(group_rets_ew) < n_groups:
            continue

        # Value-weighted returns
        if has_mcap:
            def vw_ret(grp):
                w = grp['MarketCap_EOM']
                if w.sum() <= 0:
                    return grp['MonthlyReturn'].mean()
                return np.average(grp['MonthlyReturn'], weights=w)
            group_rets_vw = cross.dropna(subset=['MarketCap_EOM']).groupby('Group').apply(vw_ret)
        else:
            group_rets_vw = group_rets_ew  # fallback to EW

        for g in range(1, n_groups + 1):
            port_returns_ew[g].append(group_rets_ew.get(g, np.nan))
            port_returns_vw[g].append(group_rets_vw.get(g, np.nan))
        port_returns_ew['H-L'].append(group_rets_ew[n_groups] - group_rets_ew[1])
        port_returns_vw['H-L'].append(group_rets_vw[n_groups] - group_rets_vw[1])
        month_list.append(ym)

    results = {}
    labels_map = {1: 'Low', 2: 'Med', 3: 'High', 'H-L': 'High-Low'}

    for weighting, port_returns in [('EW', port_returns_ew), ('VW', port_returns_vw)]:
        print(f"\n  -- {sort_label} Sort ({n_groups} groups, {weighting}) --")
        print(f"  {'Group':<10s} {'Mean(mo)':>10s} {'t(NW)':>8s} {'':4s} "
              f"{'Sharpe':>8s} {'N months':>8s}")
        print(f"  {'-'*48}")

        for g in list(range(1, n_groups + 1)) + ['H-L']:
            series = pd.Series(port_returns[g], index=month_list)
            mu, t, T = nw_tstat(series)
            sharpe = mu / series.std() * np.sqrt(12) if series.std() > 0 else 0
            label = labels_map.get(g, str(g))
            print(f"  {label:<10s} {mu*100:>10.3f}% {t:>8.2f} {sig(t):<4s} "
                  f"{sharpe:>8.2f}   {T:>5d}")
            results[(weighting, g)] = {'mean': mu, 't': t, 'T': T,
                                        'sharpe': sharpe, 'series': series}
    return results


co2_sort = single_sort(sample, 'LOG_CO2_TOTAL', 'CO₂ Emissions')
np_sort = single_sort(sample, 'NEURAL_PRED', 'Neural Prediction')

# Save single sort results
single_sort_results = []
for label, res in [('CO2', co2_sort), ('NEURAL_PRED', np_sort)]:
    for w in ['EW', 'VW']:
        hl = res[(w, 'H-L')]
        single_sort_results.append({
            'Sort': label, 'Weighting': w,
            'HmL_mean': hl['mean'],
            'HmL_t': hl['t'],
            'HmL_T': hl['T'],
            'HmL_sharpe': hl['sharpe'],
        })
pd.DataFrame(single_sort_results).to_csv(
    os.path.join(OUTPUT_DIR, 'single_sort_results.csv'), index=False)


# ============================================================
# PART B: INDEPENDENT 3×3 DOUBLE SORT
# ============================================================
print(f"\n{'='*80}")
print("  PART B: INDEPENDENT DOUBLE SORT (CO₂ × Neural Prediction)")
print(f"{'='*80}")
print("  Within each Neural Prediction tercile: is there a CO₂ spread?\n")

double_returns = {}
double_returns_vw = {}
for np_g in range(1, 4):
    for co2_g in range(1, 4):
        double_returns[(np_g, co2_g)] = []
        double_returns_vw[(np_g, co2_g)] = []
co2_spread_within = {1: [], 2: [], 3: []}
co2_spread_within_vw = {1: [], 2: [], 3: []}
month_list_ds = []

has_mcap_ds = 'MarketCap_EOM' in sample.columns

for ym in months:
    cross = sample[sample['YearMonth'] == ym].copy()
    if len(cross) < 45:
        continue
    try:
        cross['NP_Group'] = pd.qcut(
            cross['NEURAL_PRED'], 3, labels=False, duplicates='drop') + 1
        cross['CO2_Group'] = pd.qcut(
            cross['LOG_CO2_TOTAL'], 3, labels=False, duplicates='drop') + 1
    except ValueError:
        continue

    cell_counts = cross.groupby(['NP_Group', 'CO2_Group']).size()
    if len(cell_counts) < 9:
        continue
    month_list_ds.append(ym)

    for np_g in range(1, 4):
        for co2_g in range(1, 4):
            cell = cross[(cross['NP_Group'] == np_g) &
                         (cross['CO2_Group'] == co2_g)]
            double_returns[(np_g, co2_g)].append(
                cell['MonthlyReturn'].mean() if len(cell) > 0 else np.nan)
            # VW
            if has_mcap_ds and len(cell) > 0:
                cell_valid = cell.dropna(subset=['MarketCap_EOM', 'MonthlyReturn'])
                if len(cell_valid) > 0:
                    w = cell_valid['MarketCap_EOM']
                    double_returns_vw[(np_g, co2_g)].append(
                        np.average(cell_valid['MonthlyReturn'], weights=w))
                else:
                    double_returns_vw[(np_g, co2_g)].append(np.nan)
            else:
                double_returns_vw[(np_g, co2_g)].append(np.nan)

        # EW spread
        hi = cross[(cross['NP_Group'] == np_g) &
                   (cross['CO2_Group'] == 3)]['MonthlyReturn'].mean()
        lo = cross[(cross['NP_Group'] == np_g) &
                   (cross['CO2_Group'] == 1)]['MonthlyReturn'].mean()
        co2_spread_within[np_g].append(hi - lo)

        # VW spread
        if has_mcap_ds:
            hi_cell = cross[(cross['NP_Group'] == np_g) & (cross['CO2_Group'] == 3)].dropna(subset=['MarketCap_EOM'])
            lo_cell = cross[(cross['NP_Group'] == np_g) & (cross['CO2_Group'] == 1)].dropna(subset=['MarketCap_EOM'])
            if len(hi_cell) > 0 and len(lo_cell) > 0:
                hi_vw = np.average(hi_cell['MonthlyReturn'], weights=hi_cell['MarketCap_EOM'])
                lo_vw = np.average(lo_cell['MonthlyReturn'], weights=lo_cell['MarketCap_EOM'])
                co2_spread_within_vw[np_g].append(hi_vw - lo_vw)
            else:
                co2_spread_within_vw[np_g].append(np.nan)
        else:
            co2_spread_within_vw[np_g].append(np.nan)

np_labels = {1: 'NP Low', 2: 'NP Med', 3: 'NP High'}

# --- EW Results ---
print(f"  Equal-Weighted Monthly Returns (x100):")
print(f"\n  {'':15s} {'CO2 Low':>10s} {'CO2 Med':>10s} {'CO2 High':>10s} "
      f"{'H-L Spread':>12s} {'t(NW)':>8s}")
print(f"  {'-'*65}")

ds_results = []
for np_g in range(1, 4):
    row = f"  {np_labels[np_g]:<15s}"
    for co2_g in range(1, 4):
        mu = np.nanmean(double_returns[(np_g, co2_g)]) * 100
        row += f" {mu:>10.3f}"
    spread_series = pd.Series(co2_spread_within[np_g], index=month_list_ds)
    mu_sp, t_sp, T_sp = nw_tstat(spread_series)
    row += f" {mu_sp*100:>12.3f} {t_sp:>8.2f}{sig(t_sp)}"
    print(row)
    ds_results.append({'NP_Tercile': np_labels[np_g],
                        'CO2_spread': mu_sp, 't': t_sp, 'T': T_sp})

avg_spread = np.mean([co2_spread_within[g] for g in range(1, 4)], axis=0)
mu_avg, t_avg, _ = nw_tstat(pd.Series(avg_spread, index=month_list_ds))
print(f"  {'-'*65}")
print(f"  {'Average':<15s} {'':>10s} {'':>10s} {'':>10s} "
      f"{mu_avg*100:>12.3f} {t_avg:>8.2f}{sig(t_avg)}")

# --- VW Results ---
if has_mcap_ds:
    print(f"\n  Value-Weighted Monthly Returns (x100):")
    print(f"\n  {'':15s} {'CO2 Low':>10s} {'CO2 Med':>10s} {'CO2 High':>10s} "
          f"{'H-L Spread':>12s} {'t(NW)':>8s}")
    print(f"  {'-'*65}")

    for np_g in range(1, 4):
        row = f"  {np_labels[np_g]:<15s}"
        for co2_g in range(1, 4):
            mu = np.nanmean(double_returns_vw[(np_g, co2_g)]) * 100
            row += f" {mu:>10.3f}"
        spread_series_vw = pd.Series(co2_spread_within_vw[np_g], index=month_list_ds)
        mu_sp_vw, t_sp_vw, T_sp_vw = nw_tstat(spread_series_vw)
        row += f" {mu_sp_vw*100:>12.3f} {t_sp_vw:>8.2f}{sig(t_sp_vw)}"
        print(row)
        ds_results[np_g - 1]['CO2_spread_VW'] = mu_sp_vw
        ds_results[np_g - 1]['t_VW'] = t_sp_vw

# FF5-adjusted alphas for each NP tercile's CO2 H-L spread
print(f"\n  FF5-Adjusted Double-Sort Alphas (EW):")
print(f"  {'NP Tercile':<15s} {'FF5 alpha':<12s} {'t(alpha)':<10s} {'R2':<8s}")
print(f"  {'-'*50}")

for np_g in range(1, 4):
    spread_s = pd.Series(co2_spread_within[np_g], index=month_list_ds, name='spread')
    spread_ff5 = pd.DataFrame({'YearMonth': spread_s.index, 'spread': spread_s.values})
    spread_ff5 = pd.merge(spread_ff5, ff5_monthly, on='YearMonth', how='inner')
    if len(spread_ff5) < 20:
        ds_results[np_g - 1]['ff5_alpha'] = np.nan
        ds_results[np_g - 1]['ff5_t'] = np.nan
        ds_results[np_g - 1]['ff5_r2'] = np.nan
        continue
    Y = spread_ff5['spread']
    X = sm.add_constant(spread_ff5[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']])
    res = sm.OLS(Y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 6})
    alpha_val = res.params['const']
    t_alpha = res.tvalues['const']
    r2_val = res.rsquared
    ds_results[np_g - 1]['ff5_alpha'] = alpha_val
    ds_results[np_g - 1]['ff5_t'] = t_alpha
    ds_results[np_g - 1]['ff5_r2'] = r2_val
    print(f"  {np_labels[np_g]:<15s} {alpha_val*100:>+10.4f}% {t_alpha:>8.2f}{sig(t_alpha):<4s} {r2_val:>7.4f}")

# FF5-adjusted alphas for VW
if has_mcap_ds:
    print(f"\n  FF5-Adjusted Double-Sort Alphas (VW):")
    print(f"  {'NP Tercile':<15s} {'FF5 alpha':<12s} {'t(alpha)':<10s} {'R2':<8s}")
    print(f"  {'-'*50}")

    for np_g in range(1, 4):
        spread_s_vw = pd.Series(co2_spread_within_vw[np_g], index=month_list_ds, name='spread')
        spread_ff5_vw = pd.DataFrame({'YearMonth': spread_s_vw.index, 'spread': spread_s_vw.values})
        spread_ff5_vw = pd.merge(spread_ff5_vw, ff5_monthly, on='YearMonth', how='inner')
        if len(spread_ff5_vw) < 20:
            ds_results[np_g - 1]['ff5_alpha_VW'] = np.nan
            ds_results[np_g - 1]['ff5_t_VW'] = np.nan
            continue
        Y_vw = spread_ff5_vw['spread']
        X_vw = sm.add_constant(spread_ff5_vw[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']])
        res_vw = sm.OLS(Y_vw, X_vw).fit(cov_type='HAC', cov_kwds={'maxlags': 6})
        ds_results[np_g - 1]['ff5_alpha_VW'] = res_vw.params['const']
        ds_results[np_g - 1]['ff5_t_VW'] = res_vw.tvalues['const']
        print(f"  {np_labels[np_g]:<15s} {res_vw.params['const']*100:>+10.4f}% {res_vw.tvalues['const']:>8.2f}{sig(res_vw.tvalues['const']):<4s} {res_vw.rsquared:>7.4f}")




# ============================================================
# PART C: SPANNING TEST
# ============================================================
print(f"\n{'='*80}")
print("  PART C: SPANNING TEST")
print(f"{'='*80}")
print("  Dep var: CO₂ High-minus-Low spread (monthly)")
print("  Method: OLS with Newey-West HAC (maxlags=6)")

co2_hl = co2_sort[('EW', 'H-L')]['series']
np_hl = np_sort[('EW', 'H-L')]['series']

spread_df = pd.DataFrame({
    'YearMonth': co2_hl.index,
    'CO2_HmL': co2_hl.values,
    'NP_HmL': np_hl.values,
}).dropna()
spread_df = pd.merge(spread_df, ff5_monthly, on='YearMonth', how='inner')
print(f"\n  Spanning sample: {len(spread_df)} months")

spanning_models = [
    ('CAPM',         ['Mkt-RF']),
    ('FF5',          ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']),
    ('Neural HmL',   ['NP_HmL']),
    ('FF5 + Neural', ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'NP_HmL']),
]

print(f"\n  {'Model':<20s} {'α (mo)':>10s} {'t(α)':>8s} {'':<4s} {'R²':>7s}")
print(f"  {'-'*52}")

spanning_results = []
for name, factors in spanning_models:
    y = spread_df['CO2_HmL'].values
    X = sm.add_constant(spread_df[factors].values)
    res = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 6})
    alpha, t_a, r2 = res.params[0], res.tvalues[0], res.rsquared
    print(f"  {name:<20s} {alpha*100:>10.4f}% {t_a:>8.2f} {sig(t_a):<4s} {r2:>7.4f}")
    spanning_results.append({'Model': name, 'alpha': alpha,
                              'alpha_pct': alpha*100, 't_alpha': t_a,
                              'r2': r2, 'T': len(y)})


# ============================================================
# PART D: CONDITIONAL CO₂ PREMIUM BY NP QUINTILE (5×3)
# ============================================================
print(f"\n{'='*80}")
print("  PART D: CONDITIONAL CO₂ PREMIUM BY NEURAL PREDICTION QUINTILE")
print(f"{'='*80}\n")

np_q_spreads = {q: [] for q in range(1, 6)}
np_q_months = []

for ym in months:
    cross = sample[sample['YearMonth'] == ym].copy()
    if len(cross) < 75:
        continue
    try:
        cross['NP_Q'] = pd.qcut(cross['NEURAL_PRED'], 5,
                                 labels=False, duplicates='drop') + 1
        cross['CO2_T'] = pd.qcut(cross['LOG_CO2_TOTAL'], 3,
                                  labels=False, duplicates='drop') + 1
    except ValueError:
        continue
    np_q_months.append(ym)
    for q in range(1, 6):
        sub = cross[cross['NP_Q'] == q]
        hi = sub[sub['CO2_T'] == 3]['MonthlyReturn'].mean()
        lo = sub[sub['CO2_T'] == 1]['MonthlyReturn'].mean()
        np_q_spreads[q].append(hi - lo)

print(f"  {'NP Quintile':<15s} {'CO₂ H-L (mo)':>14s} {'t(NW)':>8s} {'':<4s} "
      f"{'N months':>8s}")
print(f"  {'-'*52}")
quintile_results = []
for q in range(1, 6):
    series = pd.Series(np_q_spreads[q], index=np_q_months)
    mu, t, T = nw_tstat(series)
    ql = f"Q{q} ({'Lowest NP' if q==1 else 'Highest NP' if q==5 else ''})"
    print(f"  {ql:<15s} {mu*100:>14.4f}% {t:>8.2f} {sig(t):<4s} {T:>8d}")
    quintile_results.append({'Quintile': f'Q{q}', 'spread': mu, 't': t, 'T': T})

# Monotonicity test: Q5 - Q1 spread-of-spreads
q1s = pd.Series(np_q_spreads[1], index=np_q_months)
q5s = pd.Series(np_q_spreads[5], index=np_q_months)
diff_q5q1 = q5s - q1s
mu_diff, t_diff, T_diff = nw_tstat(diff_q5q1)
print(f"\n  Q5−Q1 spread-of-spreads: {mu_diff*100:+.4f}%, t={t_diff:.2f}{sig(t_diff)}")
print(f"  → {'MONOTONIC DECLINE confirmed' if t_diff < -1.96 else 'Monotonic decline not statistically significant'}")
quintile_results.append({'Quintile': 'Q5-Q1', 'spread': mu_diff, 't': t_diff, 'T': T_diff})


# ============================================================
# PART E: CONDITIONAL CO₂ PREMIUM BY FIRM CHARACTERISTIC
# ============================================================
print(f"\n{'='*80}")
print("  PART E: CONDITIONAL CO₂ PREMIUM BY FIRM CHARACTERISTIC")
print(f"{'='*80}")
print("  Sort firms by [Variable], then compute CO₂ H-L within each tercile\n")

sort_vars = [
    ('SIZE',              'Firm Size',              False),
    ('BM',                'Book-to-Market',         False),
    ('LEVERAGE',          'Leverage',               False),
    ('IO',                'Institutional Ownership', False),
    ('ANALYSTS',          'Analyst Coverage',       True),
    ('LOG_AMIHUD',        'Amihud Illiquidity',     True),
    ('ENV_SCORE',         'Environmental Score',    True),
    ('CARBON_INTENSITY',  'Carbon Intensity',       True),
    ('DELTA_CO2',         'Emission Growth',        True),
    ('VOLAT',             'Volatility',             False),
]

cond_results = []
for sort_var, label, has_nan in sort_vars:
    if sort_var not in sample.columns:
        continue
    sub = sample.dropna(subset=[sort_var]) if has_nan else sample

    group_spreads = {g: [] for g in range(1, 4)}
    month_list_c = []

    for ym in months:
        cross = sub[sub['YearMonth'] == ym].copy()
        if len(cross) < 60:
            continue
        try:
            cross['SortG'] = pd.qcut(cross[sort_var], 3,
                                      labels=False, duplicates='drop') + 1
            cross['CO2_T'] = pd.qcut(cross['LOG_CO2_TOTAL'], 3,
                                      labels=False, duplicates='drop') + 1
        except ValueError:
            continue
        month_list_c.append(ym)
        for g in range(1, 4):
            s = cross[cross['SortG'] == g]
            hi = s[s['CO2_T'] == 3]['MonthlyReturn'].mean()
            lo = s[s['CO2_T'] == 1]['MonthlyReturn'].mean()
            group_spreads[g].append(hi - lo)

    print(f"  -- {label} ({sort_var}) --")
    print(f"  {'Group':<15s} {'CO₂ H-L (mo%)':>14s} {'t(NW)':>8s} {'':<4s}")
    print(f"  {'-'*42}")
    gl = {1: 'Low', 2: 'Med', 3: 'High'}
    for g in range(1, 4):
        series = pd.Series(group_spreads[g], index=month_list_c)
        mu, t, T = nw_tstat(series)
        print(f"  {gl[g]:<15s} {mu*100:>14.4f} {t:>8.2f} {sig(t):4s}")
        cond_results.append({'Variable': sort_var, 'Group': gl[g],
                              'spread': mu, 't': t, 'T': T})

    # Spread-of-spreads (H - L) with NW(6)
    min_len = min(len(group_spreads[3]), len(group_spreads[1]))
    sos_series = pd.Series(
        [group_spreads[3][i] - group_spreads[1][i] for i in range(min_len)],
        index=month_list_c[:min_len]
    )
    sos_mu, sos_t, sos_T = nw_tstat(sos_series)
    print(f"  {'H - L':<15s} {sos_mu*100:>14.4f} {sos_t:>8.2f} {sig(sos_t):4s}")
    cond_results.append({'Variable': sort_var, 'Group': 'H-L',
                          'spread': sos_mu, 't': sos_t, 'T': sos_T})
    print()


# ============================================================
# PART F: INDUSTRY-LEVEL CO₂ PREMIUM (RAW + FF5-ADJUSTED)
# ============================================================
print(f"\n{'='*80}")
print("  PART F: INDUSTRY-LEVEL CARBON PREMIUM")
print(f"{'='*80}")
print("  Within each industry: CO₂ H-L spread (raw + FF5 alpha)\n")

# Industries with >=20 firms/month for >=30 months
ind_counts = (sample.groupby(['TRBC_Business_Sector', 'YearMonth']).size()
              .reset_index(name='n'))
ind_valid = ind_counts[ind_counts['n'] >= 20].groupby('TRBC_Business_Sector').size()
valid_industries = ind_valid[ind_valid >= 30].index.tolist()
print(f"  Valid industries (≥20 firms/month, ≥30 months): {len(valid_industries)}")

industry_results = []
for ind in sorted(valid_industries):
    ind_data = sample[sample['TRBC_Business_Sector'] == ind]
    # Raw CO₂ spread
    spread_series = compute_co2_spread(ind_data, months, min_firms=20)
    if len(spread_series) < 20:
        continue
    mu_raw, t_raw, T = nw_tstat(spread_series)

    # FF5-adjusted: regress spread on FF5, alpha = industry carbon premium
    sp_df = pd.DataFrame({'YearMonth': spread_series.index,
                           'spread': spread_series.values})
    sp_df = pd.merge(sp_df, ff5_monthly, on='YearMonth', how='inner')

    alpha_ff5, t_ff5 = np.nan, np.nan
    if len(sp_df) >= 20:
        y = sp_df['spread'].values
        X = sm.add_constant(
            sp_df[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']].values)
        res = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 6})
        alpha_ff5 = res.params[0]
        t_ff5 = res.tvalues[0]

    n_firms = ind_data['Ticker'].nunique()
    med_co2 = ind_data['LOG_CO2_TOTAL'].median()

    industry_results.append({
        'Industry': ind, 'n_firms': n_firms, 'med_co2': med_co2,
        'raw_spread': mu_raw, 'raw_t': t_raw, 'T': T,
        'ff5_alpha': alpha_ff5, 'ff5_t': t_ff5,
    })

# Sort by raw t-stat
industry_results = sorted(industry_results,
                           key=lambda x: x['raw_t'] if not np.isnan(x['raw_t']) else 0,
                           reverse=True)

print(f"\n  {'Industry':<45s} {'Raw H-L':>9s} {'t(raw)':>7s} {'':4s} "
      f"{'FF5 α':>9s} {'t(FF5)':>7s} {'':4s} {'Firms':>5s}")
print(f"  {'-'*95}")
for r in industry_results:
    print(f"  {r['Industry'][:43]:<45s} "
          f"{r['raw_spread']*100:>+8.3f}% {r['raw_t']:>7.2f} {sig(r['raw_t']):4s} "
          f"{r['ff5_alpha']*100:>+8.3f}% {r['ff5_t']:>7.2f} {sig(r['ff5_t']):4s} "
          f"{r['n_firms']:>5d}")

# Summary
sig_raw_pos = [r for r in industry_results if r['raw_t'] > 1.96]
sig_raw_neg = [r for r in industry_results if r['raw_t'] < -1.96]
sig_ff5_pos = [r for r in industry_results
               if not np.isnan(r['ff5_t']) and r['ff5_t'] > 1.96]
sig_ff5_neg = [r for r in industry_results
               if not np.isnan(r['ff5_t']) and r['ff5_t'] < -1.96]

print(f"\n  Raw significant:  {len(sig_raw_pos)} positive, {len(sig_raw_neg)} negative")
print(f"  FF5 significant:  {len(sig_ff5_pos)} positive, {len(sig_ff5_neg)} negative")


# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*80}")
print("  SUMMARY")
print(f"{'='*80}")

co2_mu, co2_t, _ = nw_tstat(co2_sort[('EW', 'H-L')]['series'])
ns = [r for r in spanning_results if r['Model'] == 'Neural HmL']
fns = [r for r in spanning_results if r['Model'] == 'FF5 + Neural']

print(f"""
  A. SINGLE SORT:
     CO₂ H-L spread: {co2_mu*100:+.3f}%/mo, t={co2_t:.2f}{sig(co2_t)}
     → Unconditional carbon premium is {'significant' if abs(co2_t) > 1.96 else 'NOT significant'}

  B. DOUBLE SORT (CO₂ × NP):
     CO₂ spread in NP Low:  {ds_results[0]['CO2_spread']*100:+.3f}%, t={ds_results[0]['t']:.2f}{sig(ds_results[0]['t'])}
     CO₂ spread in NP High: {ds_results[2]['CO2_spread']*100:+.3f}%, t={ds_results[2]['t']:.2f}{sig(ds_results[2]['t'])}

  C. SPANNING:""")
if ns:
    print(f"     α after Neural: {ns[0]['alpha_pct']:+.4f}%, t={ns[0]['t_alpha']:.2f}")
if fns:
    print(f"     α after FF5+Neural: {fns[0]['alpha_pct']:+.4f}%, t={fns[0]['t_alpha']:.2f}")
print(f"""
  D. NP QUINTILE MONOTONICITY:
     Q1→Q5: {quintile_results[0]['spread']*100:+.3f}% → {quintile_results[4]['spread']*100:+.3f}%
     Q5−Q1 spread: {mu_diff*100:+.4f}%, t={t_diff:.2f}{sig(t_diff)}

  E. INDUSTRY: {len(sig_raw_pos)} sectors with significant positive raw premium
                {len(sig_ff5_pos)} sectors with significant FF5-adjusted premium
""")

# Save all results
pd.DataFrame(ds_results).to_csv(
    os.path.join(OUTPUT_DIR, 'portfolio_double_sort.csv'), index=False)
pd.DataFrame(spanning_results).to_csv(
    os.path.join(OUTPUT_DIR, 'spanning_test_results.csv'), index=False)
pd.DataFrame(quintile_results).to_csv(
    os.path.join(OUTPUT_DIR, 'conditional_np_quintile.csv'), index=False)
pd.DataFrame(cond_results).to_csv(
    os.path.join(OUTPUT_DIR, 'conditional_carbon_premium.csv'), index=False)
pd.DataFrame(industry_results).to_csv(
    os.path.join(OUTPUT_DIR, 'industry_carbon_premium.csv'), index=False)

print("   All results saved to results/tables/")
print("   Portfolio Sort & Spanning Tests Complete!")
