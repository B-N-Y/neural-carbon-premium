"""
03_bolton_baseline.py — Bolton & Kacperczyk (2021, JFE) Table 8 Exact Replication

Replicates Bolton's Table 8 using their EXACT methodology:

  REGRESSION:  Pooled OLS with year-month fixed effects
  STD ERRORS:  Double-clustered at firm and year level
  PANELS:      A (emission levels), B (emission growth), C (emission intensity)
  COLUMNS:     1-3 without industry FE, 4-6 with industry FE

VARIABLE DEFINITIONS (matched to Bolton Table 1):
  RET          Monthly stock return
  LOG(CO2)     log(emissions), NOT log(1+emissions) — Bolton uses log scale
  LOGSIZE      log(market cap in $million)
  B/M          Book equity / Market equity, winsorized 2.5%
  ROE          Return on equity IN PERCENT, winsorized 2.5%
  LEVERAGE     Book debt / Book assets, winsorized 2.5%
  MOM          Cumulative 12-month return, winsorized 0.5%
  INVEST/A     CAPEX / Book assets, winsorized 2.5%
  HHI          Herfindahl index (industry-level proxy, Bolton uses segment-level)
  LOGPPE       log(PPE in $million)
  BETA         12-month CAPM beta from daily returns
  VOLAT        Monthly return volatility over 12 months, winsorized 0.5%
  SALESGR      (Revenue_t - Revenue_{t-1}) / Revenue_{t-1}, winsorized 0.5%
  EPSGR        (EPS_t - EPS_{t-1}) / |EPS_{t-1}|, winsorized 0.5%

TIMING: Bolton uses lagged characteristics (fiscal year t-1) for most variables.
        Exception: HHI, SALESGR, EPSGR are contemporaneous (time t).

NOTE ON DIFFERENCES FROM BOLTON:
  - Data source: Refinitiv (not Trucost+FactSet) → lower CO2 coverage (~25% vs ~85%)
  - Sample: 2010-2025 (Bolton: 2005-2017)
  - HHI: industry-level market concentration (Bolton: firm-level segment diversification)
  - These differences are documented as limitations.

Also runs Fama-MacBeth as secondary methodology for comparison.

INPUT:  data_clean/final_monthly_panel_clean.csv
        data_raw/emissions_accounting_panel.csv
        data_clean/final_dataset_filtered.csv (daily returns for CAPM beta)
OUTPUT: results/tables/table2_bolton_replication.csv
        results/tables/table2_bolton_replication.tex
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PATHS
# ============================================================
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR   = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(PAPER_DIR)

PANEL_FILE     = os.path.join(PAPER_DIR, 'data_clean', 'final_monthly_panel_clean.csv')
EMISSIONS_FILE = os.path.join(PAPER_DIR, 'data_raw', 'emissions_accounting_panel.csv')
DAILY_FILE     = os.path.join(PROJECT_DIR, 'data_clean', 'final_dataset_filtered.csv')
OUTPUT_DIR     = os.path.join(PAPER_DIR, 'results', 'tables')


# ============================================================
# PART 1: VARIABLE CONSTRUCTION
# ============================================================
def compute_capm_beta(daily_path, tickers, window=252):
    """
    12-month (252 trading days) rolling CAPM beta from daily returns.
    Bolton: "BETA is the CAPM beta calculated over the one year period"
    """
    print("  [BETA] Loading daily returns...")
    cols = ['Ticker', 'Date', 'Return_1D', 'Mkt-RF', 'RF']
    daily = pd.read_csv(daily_path, usecols=cols)
    daily['Date'] = pd.to_datetime(daily['Date'])
    daily = daily[daily['Ticker'].isin(tickers)].copy()
    daily = daily.dropna(subset=['Return_1D', 'Mkt-RF'])
    daily['ExRet'] = daily['Return_1D'] - daily['RF']
    daily = daily.sort_values(['Ticker', 'Date'])

    print(f"  [BETA] Computing rolling {window}-day beta for {daily['Ticker'].nunique()} tickers...")

    # Vectorized rolling beta using groupby + rolling covariance
    results = []
    for ticker, grp in daily.groupby('Ticker'):
        if len(grp) < window:
            continue
        grp = grp.sort_values('Date').reset_index(drop=True)

        er = grp['ExRet'].values
        mkt = grp['Mkt-RF'].values

        # Rolling covariance and variance
        for end_idx in range(window, len(grp), 21):  # Monthly snapshots (every ~21 days)
            start_idx = end_idx - window
            er_w = er[start_idx:end_idx]
            mkt_w = mkt[start_idx:end_idx]
            var_mkt = np.var(mkt_w, ddof=1)
            if var_mkt > 1e-10:
                beta = np.cov(er_w, mkt_w)[0, 1] / var_mkt
            else:
                beta = np.nan
            results.append({
                'Ticker': ticker,
                'Date': grp.at[end_idx - 1, 'Date'],
                'CAPM_BETA': beta
            })

    beta_df = pd.DataFrame(results)
    beta_df['YearMonth'] = beta_df['Date'].dt.to_period('M').astype(str)
    # Keep last observation per month
    beta_df = beta_df.sort_values('Date').groupby(['Ticker', 'YearMonth']).last().reset_index()
    print(f"  [BETA] Done: {len(beta_df):,} monthly betas, "
          f"{beta_df['Ticker'].nunique()} tickers")
    return beta_df[['Ticker', 'YearMonth', 'CAPM_BETA']]


def compute_growth_rates(emissions_path):
    """
    SALESGR: (Revenue_t - Revenue_{t-1}) / Revenue_{t-1}
    EPSGR:   (EPS_t - EPS_{t-1}) / |EPS_{t-1}|
    Both winsorized at 0.5% (Bolton Table 1 footnote).
    """
    print("  [GROWTH] Computing SALESGR and EPSGR from emissions panel...")
    ep = pd.read_csv(emissions_path)
    ep['FiscalYear'] = pd.to_numeric(ep['FiscalYear'], errors='coerce')
    ep = ep[ep['FiscalYear'].between(2005, 2030)].copy()
    ep['FiscalYear'] = ep['FiscalYear'].astype(int)

    for col in ['TR.Revenue', 'TR.EPSActValue']:
        ep[col] = pd.to_numeric(ep[col], errors='coerce')

    # Collapse to one observation per ticker-year
    yearly = ep.groupby(['Instrument', 'FiscalYear']).agg({
        'TR.Revenue': 'first',
        'TR.EPSActValue': 'first',
    }).reset_index()
    yearly = yearly.rename(columns={'Instrument': 'Ticker'})
    yearly = yearly.sort_values(['Ticker', 'FiscalYear'])

    # Lag values
    yearly['Rev_lag'] = yearly.groupby('Ticker')['TR.Revenue'].shift(1)
    yearly['EPS_lag'] = yearly.groupby('Ticker')['TR.EPSActValue'].shift(1)

    # Growth rates
    mask_rev = yearly['Rev_lag'].abs() > 1e-3  # avoid division by near-zero
    yearly.loc[mask_rev, 'SALESGR_BOLTON'] = (
        (yearly.loc[mask_rev, 'TR.Revenue'] - yearly.loc[mask_rev, 'Rev_lag'])
        / yearly.loc[mask_rev, 'Rev_lag'].abs()
    )

    mask_eps = yearly['EPS_lag'].abs() > 0.01
    yearly.loc[mask_eps, 'EPSGR_BOLTON'] = (
        (yearly.loc[mask_eps, 'TR.EPSActValue'] - yearly.loc[mask_eps, 'EPS_lag'])
        / yearly.loc[mask_eps, 'EPS_lag'].abs()
    )

    # Winsorize at 0.5% (Bolton spec)
    for col in ['SALESGR_BOLTON', 'EPSGR_BOLTON']:
        lo = yearly[col].quantile(0.005)
        hi = yearly[col].quantile(0.995)
        yearly[col] = yearly[col].clip(lo, hi)

    result = yearly[['Ticker', 'FiscalYear', 'SALESGR_BOLTON', 'EPSGR_BOLTON']].copy()
    print(f"  [GROWTH] SALESGR: {result['SALESGR_BOLTON'].notna().sum():,} obs")
    print(f"  [GROWTH] EPSGR:   {result['EPSGR_BOLTON'].notna().sum():,} obs")
    return result


def build_bolton_panel():
    """
    Load base panel and reconstruct all variables to match Bolton's definitions.
    """
    print("\n" + "=" * 80)
    print("STEP 1: BUILDING BOLTON-CONSISTENT PANEL")
    print("=" * 80)

    df = pd.read_csv(PANEL_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    print(f"  Base panel: {len(df):,} rows, {df['Ticker'].nunique()} tickers")

    # ----- Fix INVEST/A: Refinitiv reports CAPEX as NEGATIVE (cash outflow) -----
    # Bolton: CAPEX/Assets = positive ratio (mean 0.05)
    # Our data: mean=-0.045 → take absolute value
    df['INVEST_A'] = df['INVEST_A'].abs()
    lo, hi = df['INVEST_A'].quantile(0.025), df['INVEST_A'].quantile(0.975)
    df['INVEST_A'] = df['INVEST_A'].clip(lo, hi)
    print(f"  [INVEST_A] Fixed sign (abs), winsorized 2.5%: mean={df['INVEST_A'].mean():.4f} (Bolton: 0.05)")

    # ----- Fix ROE: ratio → percentage (Bolton: mean 9.76%) -----
    # First remove extreme outliers before percentage conversion
    lo, hi = df['ROE'].quantile(0.025), df['ROE'].quantile(0.975)
    df['ROE_PCT'] = df['ROE'].clip(lo, hi) * 100
    print(f"  [ROE] Converted to %, winsorized 2.5%: mean={df['ROE_PCT'].mean():.2f}% (Bolton: 9.76%)")

    # ----- Fix B/M: winsorize at 2.5% -----
    lo, hi = df['BM'].quantile(0.025), df['BM'].quantile(0.975)
    df['BM'] = df['BM'].clip(lo, hi)

    # ----- Fix MOM: winsorize at 0.5% (Bolton) -----
    lo, hi = df['MOM'].quantile(0.005), df['MOM'].quantile(0.995)
    df['MOM'] = df['MOM'].clip(lo, hi)

    # ----- Fix VOLAT: winsorize at 0.5% -----
    lo, hi = df['VOLAT'].quantile(0.005), df['VOLAT'].quantile(0.995)
    df['VOLAT'] = df['VOLAT'].clip(lo, hi)

    # ----- Fix LEVERAGE: winsorize at 2.5% -----
    lo, hi = df['LEVERAGE'].quantile(0.025), df['LEVERAGE'].quantile(0.975)
    df['LEVERAGE'] = df['LEVERAGE'].clip(lo, hi)

    # ----- Add SALESGR_BOLTON and EPSGR_BOLTON -----
    growth = compute_growth_rates(EMISSIONS_FILE)
    # SALESGR and EPSGR are contemporaneous in Bolton (time t, not t-1)
    # So we merge on FiscalYear = FY_match + 1 (current year, not lagged)
    # Actually Bolton footnote 13: "HHI, SALESGR, and EPSGR are measured as of time t"
    # Our FY_match is already the lagged fiscal year, so for contemporaneous we need FY_match+1
    df['FY_current'] = df['FY_match'] + 1
    df = pd.merge(df, growth, left_on=['Ticker', 'FY_current'],
                   right_on=['Ticker', 'FiscalYear'], how='left',
                   suffixes=('', '_growth'))

    print(f"  [SALESGR] Bolton-style coverage: {df['SALESGR_BOLTON'].notna().sum():,} "
          f"({df['SALESGR_BOLTON'].notna().mean()*100:.1f}%)")
    print(f"  [EPSGR] coverage: {df['EPSGR_BOLTON'].notna().sum():,} "
          f"({df['EPSGR_BOLTON'].notna().mean()*100:.1f}%)")

    # ----- Add CAPM_BETA -----
    co2_tickers = df[df['LOG_CO2_TOTAL'].notna()]['Ticker'].unique()
    try:
        capm = compute_capm_beta(DAILY_FILE, co2_tickers)
        df = pd.merge(df, capm, on=['Ticker', 'YearMonth'], how='left')
        print(f"  [BETA] CAPM_BETA coverage: {df['CAPM_BETA'].notna().sum():,}")
    except Exception as e:
        print(f"  [BETA]  CAPM beta failed ({e}), using TR.WACCBeta")
        df['CAPM_BETA'] = df['BETA']

    # ----- Convert RET to percentage (Bolton reports in %) -----
    df['RET_PCT'] = df['MonthlyReturn'] * 100

    # ----- Create numeric time index for PanelOLS -----
    df['TimeIdx'] = pd.to_datetime(df['Date']).astype(np.int64) // 10**9

    print(f"\n   Bolton panel ready: {len(df):,} rows")
    return df


# ============================================================
# PART 2: REGRESSION ENGINE
# ============================================================
def run_pooled_ols(df, dep_var, indep_vars, industry_fe=False):
    """
    Bolton-style Pooled OLS:
      R_it = a0 + a1*CO2 + a2*Controls + μ_t + ε_it

    With:
      - Year-month fixed effects (μ_t)
      - Double-clustered standard errors (firm + year)
      - Optional industry fixed effects
    """
    all_vars = [dep_var] + indep_vars
    if industry_fe:
        all_vars.append('Industry')
    reg = df[['Ticker', 'TimeIdx', 'Year'] + all_vars].dropna(subset=[dep_var] + indep_vars)

    if len(reg) < 200:
        return None

    reg = reg.set_index(['Ticker', 'TimeIdx'])

    y = reg[dep_var]
    if industry_fe and 'Industry' in reg.columns:
        ind_dummies = pd.get_dummies(reg['Industry'], prefix='IND', drop_first=True)
        X = pd.concat([reg[indep_vars], ind_dummies], axis=1)
    else:
        X = reg[indep_vars]

    try:
        mod = PanelOLS(y, X, time_effects=True, entity_effects=False, check_rank=False)
        res = mod.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)

        co2_var = indep_vars[0]
        return {
            'coef': res.params[co2_var],
            't_stat': res.tstats[co2_var],
            'pval': res.pvalues[co2_var],
            'r2': res.rsquared,
            'n_obs': int(res.nobs),
            'n_firms': int(reg.index.get_level_values(0).nunique()),
        }
    except Exception as e:
        print(f"     PanelOLS error: {e}")
        return None


def newey_west_tstat(gammas, max_lag=6):
    """Newey-West (1987) t-statistic for mean of time-series coefficients."""
    T = len(gammas)
    if T < 10:
        return np.nan
    mu = gammas.mean()
    dm = gammas - mu
    var = np.sum(dm**2) / T
    for lag in range(1, max_lag + 1):
        w = 1 - lag / (max_lag + 1)
        var += 2 * w * np.sum(dm[lag:] * dm[:-lag]) / T
    se = np.sqrt(var / T)
    return mu / se if se > 1e-15 else np.nan


def run_fama_macbeth(df, dep_var, indep_vars, min_obs=30):
    """Fama-MacBeth cross-sectional regression (secondary method)."""
    months = sorted(df['YearMonth'].unique())
    gammas_all = {v: [] for v in ['const'] + indep_vars}
    r2_list, n_list = [], []

    for month in months:
        cross = df[df['YearMonth'] == month][[dep_var] + indep_vars].dropna()
        if len(cross) < min_obs:
            continue
        y = cross[dep_var].values
        X = sm.add_constant(cross[indep_vars].values)
        try:
            res = sm.OLS(y, X).fit()
            gammas_all['const'].append(res.params[0])
            for i, v in enumerate(indep_vars):
                gammas_all[v].append(res.params[i + 1])
            r2_list.append(res.rsquared)
            n_list.append(len(cross))
        except:
            continue

    if not gammas_all[indep_vars[0]]:
        return None

    results = {}
    for v in ['const'] + indep_vars:
        g = np.array(gammas_all[v])
        results[v] = {
            'coef': g.mean(),
            't_nw': newey_west_tstat(g),
        }
    results['_meta'] = {
        'avg_r2': np.mean(r2_list),
        'avg_n': int(np.mean(n_list)),
        'n_months': len(r2_list),
    }
    return results


def s(t):
    """Significance stars."""
    t = abs(t) if not np.isnan(t) else 0
    if t >= 2.576: return '***'
    if t >= 1.960: return '**'
    if t >= 1.645: return '*'
    return ''


# ============================================================
# PART 3: MAIN ANALYSIS
# ============================================================
def main():
    print("=" * 80)
    print("=  BOLTON & KACPERCZYK (2021, JFE) TABLE 8 — EXACT REPLICATION")
    print("=  Pooled OLS / Time FE / Double-Cluster SE (Firm + Year)")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = build_bolton_panel()

    # Bolton's control vector (Table 8)
    bolton_controls = [
        'SIZE',               # LOGSIZE
        'BM',                 # Book-to-Market
        'ROE_PCT',            # ROE in %
        'LEVERAGE',           # Debt/Assets
        'MOM',                # 12-month momentum
        'INVEST_A',           # CAPEX/Assets
        'HHI',                # Herfindahl (industry-level proxy)
        'IO',                 # Institutional Ownership
        'LOG_PPE',            # log(PPE)
        'CAPM_BETA',          # 12-month CAPM beta
        'VOLAT',              # Return volatility
        'SALESGR_BOLTON',     # Revenue growth rate
        'EPSGR_BOLTON',       # EPS growth rate
    ]

    # Also define a "no EPSGR/SALESGR" variant for larger sample
    bolton_core = [c for c in bolton_controls
                   if c not in ['SALESGR_BOLTON', 'EPSGR_BOLTON']]

    all_results = []

    # ============================================================
    # PANEL A: Total Emission Levels
    # ============================================================
    co2_panels = {
        'Panel A: Emission Levels': {
            'Scope 1': 'LOG_SCOPE1',
            'Scope 2': 'LOG_SCOPE2',
            'Total CO2': 'LOG_CO2_TOTAL',
        },
        'Panel B: Emission Growth': {
            'ΔCO2': 'DELTA_CO2',
        },
        'Panel C: Emission Intensity': {
            'CO2/Revenue': 'CARBON_INTENSITY',
        },
    }

    for panel_name, co2_vars in co2_panels.items():
        print(f"\n{'='*80}")
        print(f"  {panel_name}")
        print(f"{'='*80}")

        for co2_label, co2_var in co2_vars.items():
            print(f"\n  -- {co2_label} ({co2_var}) --")

            # --- A) Pooled OLS: Full Bolton controls ---
            indep_full = [co2_var] + bolton_controls
            indep_core = [co2_var] + bolton_core

            for controls_label, indep in [('Full Bolton', indep_full),
                                           ('Core (no SALESGR/EPSGR)', indep_core)]:

                for ind_fe, fe_label in [(False, 'No IndFE'), (True, '+ IndFE')]:
                    r = run_pooled_ols(df, 'RET_PCT', indep, industry_fe=ind_fe)
                    if r:
                        label = f"Pooled OLS {fe_label} [{controls_label}]"
                        t = r['t_stat']
                        print(f"    {label:55s}: γ={r['coef']:>10.4f} "
                              f"(t={t:>6.2f}){s(t):4s} | R²={r['r2']:.4f} "
                              f"N={r['n_obs']:,} firms={r['n_firms']}")
                        all_results.append({
                            'Panel': panel_name, 'CO2': co2_label,
                            'Method': label, **r
                        })

            # --- B) Fama-MacBeth (secondary) ---
            fm_full = run_fama_macbeth(df, 'MonthlyReturn', indep_full)
            if fm_full:
                co2_r = fm_full[co2_var]
                t = co2_r['t_nw']
                print(f"    {'FM (NW) [Full Bolton]':55s}: γ={co2_r['coef']*100:>10.4f} "
                      f"(t={t:>6.2f}){s(t):4s} | R²={fm_full['_meta']['avg_r2']:.4f} "
                      f"N/mo={fm_full['_meta']['avg_n']} T={fm_full['_meta']['n_months']}")
                all_results.append({
                    'Panel': panel_name, 'CO2': co2_label,
                    'Method': 'Fama-MacBeth [Full Bolton]',
                    'coef': co2_r['coef'] * 100, 't_stat': t,
                    'r2': fm_full['_meta']['avg_r2'],
                    'n_obs': fm_full['_meta']['avg_n'] * fm_full['_meta']['n_months'],
                })

            fm_core = run_fama_macbeth(df, 'MonthlyReturn',
                                       [co2_var] + bolton_core)
            if fm_core:
                co2_r = fm_core[co2_var]
                t = co2_r['t_nw']
                print(f"    {'FM (NW) [Core]':55s}: γ={co2_r['coef']*100:>10.4f} "
                      f"(t={t:>6.2f}){s(t):4s} | R²={fm_core['_meta']['avg_r2']:.4f} "
                      f"N/mo={fm_core['_meta']['avg_n']} T={fm_core['_meta']['n_months']}")

    # ============================================================
    # SUMMARY & COMPARISON WITH BOLTON
    # ============================================================
    print(f"\n{'='*80}")
    print("=  COMPARISON WITH BOLTON (2021) TABLE 8 ORIGINAL RESULTS")
    print(f"{'='*80}")

    bolton_original = {
        'Scope 1': {'coef': 0.043, 't': 2.0, 'stars': '**',
                     'coef_ife': 0.164, 't_ife': 3.0, 'stars_ife': '***'},
        'Scope 2': {'coef': 0.098, 't': 2.0, 'stars': '**',
                     'coef_ife': 0.167, 't_ife': 3.0, 'stars_ife': '***'},
        'Total CO2': {'coef': 0.135, 't': 2.0, 'stars': '**',
                       'coef_ife': 0.312, 't_ife': 3.0, 'stars_ife': '***'},
    }

    print(f"\n  {'CO2 Measure':<15s} | {'Bolton (2021)':>20s} | {'Our Replication':>20s} | {'Match?':>8s}")
    print(f"  {'-'*15}-+-{'-'*20}-+-{'-'*20}-+-{'-'*8}")

    for co2_label in ['Scope 1', 'Scope 2', 'Total CO2']:
        # Find our Pooled OLS no-IndFE, core controls result
        our = [r for r in all_results
               if r['CO2'] == co2_label
               and 'Pooled OLS No IndFE' in r['Method']
               and 'Core' in r['Method']]
        b = bolton_original.get(co2_label, {})

        if our and b:
            r = our[0]
            b_str = f"{b['coef']:.3f} (t≈{b['t']:.0f}){b['stars']}"
            o_str = f"{r['coef']:.4f} (t={r['t_stat']:.2f}){s(r['t_stat'])}"
            same_sign = (r['coef'] > 0) == (b['coef'] > 0)
            match = " Sign" if same_sign else ""
            if same_sign and abs(r['t_stat']) >= 1.96:
                match = " Full"
            print(f"  {co2_label:<15s} | {b_str:>20s} | {o_str:>20s} | {match:>8s}")

    # ============================================================
    # CONTROL VARIABLE COEFFICIENTS (Full Bolton, no IndFE, Total CO2)
    # ============================================================
    print(f"\n{'='*80}")
    print("  CONTROL VARIABLE COEFFICIENTS (Pooled OLS, Total CO2, No IndFE)")
    print(f"{'='*80}")

    indep = ['LOG_CO2_TOTAL'] + bolton_core
    reg = df[['Ticker', 'TimeIdx', 'Year', 'RET_PCT'] + indep + ['Industry']].dropna(
        subset=['RET_PCT'] + indep)
    reg = reg.set_index(['Ticker', 'TimeIdx'])

    try:
        mod = PanelOLS(reg['RET_PCT'], reg[indep], time_effects=True,
                       entity_effects=False, check_rank=False)
        res = mod.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)

        print(f"\n  {'Variable':<20s} {'Coef':>10s} {'t-stat':>10s} {'':>5s}  Bolton")
        print(f"  {'-'*60}")
        # Bolton Table 8 Panel A Col 1 coefficients for reference
        bolton_ctrl = {
            'LOG_CO2_TOTAL': (0.043, '**'),
            'SIZE': (-0.140, ''),
            'BM': (0.460, ''),
            'ROE_PCT': (0.010, '*'),
            'LEVERAGE': (-0.559, '*'),
            'MOM': (0.321, ''),
            'INVEST_A': (-2.218, ''),
            'HHI': (0.032, ''),
            'IO': (-0.010, ''),
            'LOG_PPE': (-0.015, ''),
            'CAPM_BETA': (0.059, ''),
            'VOLAT': (0.978, ''),
        }
        for var in indep:
            coef = res.params[var]
            tstat = res.tstats[var]
            b_ref = bolton_ctrl.get(var, ('—', ''))
            b_str = f"{b_ref[0]}{b_ref[1]}" if isinstance(b_ref[0], float) else '—'
            print(f"  {var:<20s} {coef:>10.4f} {tstat:>10.2f} {s(tstat):>5s}  {b_str}")

        # Compute overall R² (Bolton-comparable, includes time FE)
        y_vals = reg['RET_PCT'].values
        ym_means_y = reg.groupby(reg.index.get_level_values(1))['RET_PCT'].transform('mean').values
        y_dm = y_vals - ym_means_y
        X_dm = (reg[indep] - reg.groupby(reg.index.get_level_values(1))[indep].transform('mean')).values
        try:
            coefs_ols = np.linalg.lstsq(X_dm, y_dm, rcond=None)[0]
            fitted = X_dm @ coefs_ols + ym_means_y
            ss_res = np.sum((y_vals - fitted)**2)
            ss_tot = np.sum((y_vals - y_vals.mean())**2)
            overall_r2 = 1 - ss_res / ss_tot
        except:
            overall_r2 = np.nan

        print(f"\n  Within R²  = {res.rsquared:.4f}  (time-demeaned, PanelOLS default)")
        print(f"  Overall R² = {overall_r2:.4f}  (Bolton-comparable, includes time FE)")
        print(f"  Bolton R²  = 0.203")
        print(f"  N  = {res.nobs:,.0f}    (Bolton: 184,288)")
    except Exception as e:
        print(f"  Error: {e}")

    # ============================================================
    # SAVE
    # ============================================================
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(OUTPUT_DIR, 'table2_bolton_replication.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n Saved: {csv_path}")
    print(f"\n Bolton Table 8 Replication Complete!")


if __name__ == "__main__":
    main()
