"""
10_causal_forest_cate.py — Causal Forest: CO₂ Effect Heterogeneity (v3)
========================================================================
Estimates heterogeneous treatment effects of high carbon intensity on
stock returns (neural residuals), identifying which firm characteristics
moderate the carbon premium.

Design:
  - Unit: FIRM-YEAR (annual panel)
  - Treatment: D_HIGH_CO2 (binary: top tercile of CARBON_INTENSITY_L1)
  - Outcome: Annual mean NEURAL_RESID
  - Confounders (W): 11 Bolton controls
  - Effect modifiers (X): Size, Leverage, BM, Volatility, IO
  - CATE(x) = E[ε(High CO₂) - ε(Low CO₂) | X=x]

Key question: Where does the carbon premium survive neural controls?
"""
import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings('ignore')

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT, 'results')
TABLE_DIR = os.path.join(RESULTS_DIR, 'tables')
FIG_DIR = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ===============================================================
# LOAD & AGGREGATE TO FIRM-YEAR
# ===============================================================
RESULTS_DIR = os.path.join(PROJECT, 'results')
MONTHLY_PATH = os.path.join(RESULTS_DIR, 'analysis_panel_monthly.csv')

print(f"Loading monthly panel: {MONTHLY_PATH}")
df = pd.read_csv(MONTHLY_PATH, low_memory=False)
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
print(f"  Monthly: {len(df):,} rows")

# Aggregate to firm-year
return_cols = ['Actual', 'NEURAL_RESID']
annual_cols = ['CARBON_INTENSITY_L1', 'SIZE_L1', 'BM_L1',
               'ROE', 'INVEST_A', 'LEVERAGE', 'BETA', 'VOLAT', 'HHI', 'IO', 'LOG_PPE',
               'SALESGR', 'Sector']
annual_cols = [c for c in annual_cols if c in df.columns]

agg_dict = {c: 'mean' for c in return_cols}
for c in annual_cols:
    agg_dict[c] = 'first'

firm_year = df.groupby(['Instrument', 'Year']).agg(agg_dict).reset_index()
print(f"  Firm-year: {len(firm_year):,} obs, {firm_year['Instrument'].nunique()} tickers, "
      f"{firm_year['Year'].nunique()} years")

# ===============================================================
# PREPARE VARIABLES
# ===============================================================
# Bolton controls
W_cols_raw = ['SIZE_L1', 'BM_L1', 'ROE', 'INVEST_A', 'LEVERAGE', 'BETA', 'VOLAT', 'HHI', 'IO',
              'LOG_PPE', 'SALESGR']
W_cols_raw = [c for c in W_cols_raw if c in firm_year.columns]

# Effect modifiers (firm characteristics that might moderate the CO₂ effect)
X_cols_raw = ['SIZE_L1', 'LEVERAGE', 'BM_L1', 'VOLAT', 'IO']
X_cols_raw = [c for c in X_cols_raw if c in firm_year.columns]

# Require CO₂ + confounders
required = ['NEURAL_RESID', 'Actual', 'CARBON_INTENSITY_L1'] + W_cols_raw
df_clean = firm_year.dropna(subset=required).copy()

# Winsorize BEFORE z-scoring
for col in ['CARBON_INTENSITY_L1', 'NEURAL_RESID', 'Actual'] + W_cols_raw:
    if col in df_clean.columns:
        q = df_clean[col].quantile([0.01, 0.99])
        df_clean[col] = df_clean[col].clip(q.iloc[0], q.iloc[1])

# Z-SCORE all continuous variables
z_cols = list(set(W_cols_raw + X_cols_raw + ['CARBON_INTENSITY_L1']))
for col in z_cols:
    mu, sigma = df_clean[col].mean(), df_clean[col].std()
    df_clean[f'{col}_Z'] = (df_clean[col] - mu) / (sigma + 1e-8)

# Treatment: HIGH CO₂ (top tercile vs bottom tercile)
co2_rank = df_clean.groupby('Year')['CARBON_INTENSITY_L1'].rank(pct=True)
df_clean['D_HIGH_CO2'] = np.where(co2_rank >= 0.667, 1, np.where(co2_rank <= 0.333, 0, np.nan))
df_clean = df_clean.dropna(subset=['D_HIGH_CO2']).copy()
df_clean['D_HIGH_CO2'] = df_clean['D_HIGH_CO2'].astype(int)

W_cols = [f'{c}_Z' for c in W_cols_raw]
X_cols = [f'{c}_Z' for c in X_cols_raw]

print(f"\n  Clean sample: {len(df_clean):,} firm-years, {df_clean['Instrument'].nunique()} tickers")
print(f"  Treatment (D_HIGH_CO2): {df_clean['D_HIGH_CO2'].value_counts().to_dict()}")
print(f"  Treatment balance: {df_clean['D_HIGH_CO2'].mean():.1%} treated")

# ===============================================================
# CAUSAL FOREST — Treatment: D_HIGH_CO2
# ===============================================================
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

print(f"\n{'='*60}")
print("CAUSAL FOREST: Effect of High CO₂ on Neural Residuals")
print(f"{'='*60}")

# --- Neural Residuals ---
Y_col = 'NEURAL_RESID'
T_col = 'D_HIGH_CO2'

print(f"  Y: {Y_col} | T: {T_col}")
print(f"  X (effect modifiers): {X_cols_raw}")
print(f"  W (confounders, {len(W_cols)}): {W_cols_raw}")

Y = df_clean[Y_col].values
T = df_clean[T_col].values.astype(int)
X = df_clean[X_cols].values
W = df_clean[W_cols].values

cf = CausalForestDML(
    model_y=GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42),
    model_t=GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42),
    discrete_treatment=True,
    n_estimators=1000,
    min_samples_leaf=30,
    random_state=42,
    cv=5
)

cf.fit(Y=Y, T=T, X=X, W=W)
print(" CausalForest fitted (Neural Residuals)")

# ATE
ate = cf.ate(X)
ate_inf = cf.ate_inference(X)
try:
    ate_ci = ate_inf.conf_int_mean()
    ate_pval = float(np.ravel(ate_inf.pvalue())[0])
except:
    ate_ci = None
    ate_pval = None

print(f"\n  ATE (High CO₂ → Neural Resid): {ate:.6f}")
if ate_pval is not None:
    s = '***' if ate_pval<0.01 else '**' if ate_pval<0.05 else '*' if ate_pval<0.10 else ''
    print(f"  p-value = {ate_pval:.4f}{s}")
if ate_ci is not None:
    try:
        print(f"  95% CI = [{np.ravel(ate_ci[0])[0]:.6f}, {np.ravel(ate_ci[1])[0]:.6f}]")
    except:
        print(f"  95% CI = {ate_ci}")

# CATE
cate = cf.effect(X)
df_clean['CATE_neural'] = cate

# --- Raw Returns (comparison) ---
print(f"\n  Fitting comparison CF on raw returns...")
Y_lin = df_clean['Actual'].values
cf_lin = CausalForestDML(
    model_y=GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42),
    model_t=GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42),
    discrete_treatment=True,
    n_estimators=1000,
    min_samples_leaf=30,
    random_state=42,
    cv=5
)
cf_lin.fit(Y=Y_lin, T=T, X=X, W=W)
ate_lin = cf_lin.ate(X)
try:
    ate_lin_pval = float(np.ravel(cf_lin.ate_inference(X).pvalue())[0])
except:
    ate_lin_pval = None

cate_lin = cf_lin.effect(X)
df_clean['CATE_raw'] = cate_lin

print(f"  ATE (High CO₂ → Raw Returns): {ate_lin:.6f}" +
      (f" (p={ate_lin_pval:.4f})" if ate_lin_pval else ""))

# ===============================================================
# CATE BY FIRM CHARACTERISTICS
# ===============================================================
print(f"\n{'='*60}")
print("CONDITIONAL AVERAGE TREATMENT EFFECT (CATE)")
print(f"{'='*60}")

# By effect modifier terciles
results_rows = []
for x_col, x_raw in zip(X_cols, X_cols_raw):
    x_rank = df_clean[x_raw].rank(pct=True)
    df_clean[f'{x_raw}_tercile'] = pd.cut(x_rank, bins=[0, 0.333, 0.667, 1.0],
                                           labels=['Low', 'Med', 'High'],
                                           include_lowest=True)
    print(f"\n  CATE by {x_raw}:")
    print(f"    {'Tercile':<10s} {'Neural CATE':>12s} {'t':>8s}  {'Raw CATE':>12s} {'t':>8s}  {'N':>6s}")
    for terc in ['Low', 'Med', 'High']:
        sub_mask = (df_clean[f'{x_raw}_tercile'] == terc).values
        if sub_mask.sum() == 0:
            continue
        sub_X = X[sub_mask]
        n_sub = int(sub_mask.sum())

        # --- Neural CATE: CLT SE with cluster adjustment ---
        # NOTE: EconML's population_summary().stderr_mean returns sqrt(mean(SE_i^2))
        # when mean_pred_stderr is None (which CausalForestDML does not compute).
        # This is a conservative upper bound that does NOT divide by sqrt(n),
        # inflating SE by sqrt(n) and producing near-zero t-stats.
        # 
        # Correct SE for the mean CATE: sqrt(mean(SE_i^2) / n_sub)
        # With cluster adjustment for within-firm correlation (avg ~7 years/firm):
        # SE_cluster = SE_CLT * sqrt(avg_years_per_firm)
        avg_years = 7.0  # from data: 20,902 firm-years / 2,986 firms
        cluster_factor = np.sqrt(avg_years)
        
        inf_n = cf.effect_inference(sub_X)
        pts_n = np.ravel(inf_n.point_estimate)
        ses_n = np.ravel(inf_n.pred_stderr)
        n_mean = float(np.mean(pts_n))
        n_se_clt = float(np.sqrt(np.mean(ses_n**2) / n_sub))
        n_se_cluster = n_se_clt * cluster_factor
        n_t = n_mean / n_se_cluster if n_se_cluster > 0 else 0.0

        n_sig = '***' if abs(n_t) > 2.58 else '**' if abs(n_t) > 1.96 else '*' if abs(n_t) > 1.65 else ''

        # --- Raw CATE: CLT SE with cluster adjustment ---
        inf_r = cf_lin.effect_inference(sub_X)
        pts_r = np.ravel(inf_r.point_estimate)
        ses_r = np.ravel(inf_r.pred_stderr)
        r_mean = float(np.mean(pts_r))
        r_se_clt = float(np.sqrt(np.mean(ses_r**2) / n_sub))
        r_se_cluster = r_se_clt * cluster_factor
        r_t = r_mean / r_se_cluster if r_se_cluster > 0 else 0.0

        r_sig = '***' if abs(r_t) > 2.58 else '**' if abs(r_t) > 1.96 else '*' if abs(r_t) > 1.65 else ''

        print(f"    {terc:<10s} {n_mean:>+12.6f} {n_t:>7.2f}{n_sig:3s} {r_mean:>+12.6f} {r_t:>7.2f}{r_sig:3s}  {n_sub:>6d}")

        results_rows.append({
            'Modifier': x_raw, 'Tercile': terc,
            'CATE_neural': n_mean, 'SE_neural': n_se_cluster, 't_neural': n_t,
            'CATE_raw': r_mean, 'SE_raw': r_se_cluster, 't_raw': r_t,
            'N': n_sub
        })

# Feature importance
print(f"\n{'='*60}")
print("FEATURE IMPORTANCE (Effect Modifiers)")
print(f"{'='*60}")
fi = cf.feature_importances_
fi_dict = {}
for i, col in enumerate(X_cols_raw):
    print(f"  {col:<20s}: {fi[i]:.4f} ({fi[i]*100:.1f}%)")
    fi_dict[f'FI_{col}'] = fi[i]

# ===============================================================
# VISUALIZATION
# ===============================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. CATE distribution — neural vs raw
axes[0].hist(df_clean['CATE_neural'], bins=50, alpha=0.6, color='steelblue',
             edgecolor='white', label=f'Neural Resid (ATE={ate:.4f})')
axes[0].hist(df_clean['CATE_raw'], bins=50, alpha=0.4, color='coral',
             edgecolor='white', label=f'Raw Returns (ATE={ate_lin:.4f})')
axes[0].axvline(0, color='black', ls='--', lw=1)
axes[0].set_xlabel('CATE (High CO₂ effect)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Treatment Effects')
axes[0].legend(fontsize=8)

# 2. CATE by Size (most important modifier typically)
size_grps = df_clean.groupby('SIZE_L1_tercile')
labels_size = ['Small', 'Mid', 'Large']
neural_means = [size_grps.get_group(t)['CATE_neural'].mean() for t in ['Low', 'Med', 'High']]
raw_means = [size_grps.get_group(t)['CATE_raw'].mean() for t in ['Low', 'Med', 'High']]
neural_ses = [size_grps.get_group(t)['CATE_neural'].std()/np.sqrt(len(size_grps.get_group(t)))
              for t in ['Low', 'Med', 'High']]
raw_ses = [size_grps.get_group(t)['CATE_raw'].std()/np.sqrt(len(size_grps.get_group(t)))
           for t in ['Low', 'Med', 'High']]
x_pos = np.arange(3)
axes[1].bar(x_pos - 0.15, raw_means, 0.3, yerr=raw_ses, label='Raw Returns',
            color='coral', edgecolor='white', capsize=4, alpha=0.7)
axes[1].bar(x_pos + 0.15, neural_means, 0.3, yerr=neural_ses, label='Neural Resid',
            color='steelblue', edgecolor='white', capsize=4, alpha=0.7)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(labels_size)
axes[1].axhline(0, color='black', ls='-', lw=0.5)
axes[1].set_ylabel('Mean CATE')
axes[1].set_title('CO₂ Effect by Firm Size')
axes[1].legend(fontsize=8)

# 3. Feature importance
fi_sorted = sorted(zip(X_cols_raw, fi), key=lambda x: x[1], reverse=True)
names, vals = zip(*fi_sorted)
axes[2].barh(range(len(names)), vals, color='steelblue', edgecolor='white')
axes[2].set_yticks(range(len(names)))
axes[2].set_yticklabels(names)
axes[2].set_xlabel('Feature Importance')
axes[2].set_title('Which Characteristics Moderate CO₂ Effect?')
axes[2].invert_yaxis()

plt.suptitle('Causal Forest: Heterogeneous Carbon Premium (High vs Low CO₂)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
fig_path = os.path.join(FIG_DIR, 'causal_forest_cate.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\n {fig_path}")

# ===============================================================
# SAVE RESULTS
# ===============================================================
# Summary CSV
summary = {
    'ATE_neural': ate, 'ATE_neural_pval': ate_pval,
    'ATE_raw': ate_lin, 'ATE_raw_pval': ate_lin_pval,
    'N_firm_years': len(df_clean), 'N_tickers': df_clean['Instrument'].nunique(),
}
summary.update(fi_dict)

pd.DataFrame([summary]).to_csv(os.path.join(TABLE_DIR, 'causal_forest_results.csv'), index=False)

# Detailed CATE by modifier
cate_df = pd.DataFrame(results_rows)
cate_df.to_csv(os.path.join(TABLE_DIR, 'causal_forest_cate_by_characteristic.csv'), index=False)

print(f" Results saved to {TABLE_DIR}")
print(" Done")

