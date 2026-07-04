"""
09_clark_west_test.py — Formal OOS Forecast Comparison (K=5)
=============================================================

Tests whether Model B (with CO₂) significantly outperforms Model A (without CO₂)
in out-of-sample prediction using daily test predictions from K=5 models.

Tests:
  1. Clark-West (2007) — nested model comparison (B = A + CO₂)
  2. Diebold-Mariano (1995) with Newey-West HAC SE
  3. Paired t-test — fold-level R² differences

Data:
  Model A: daily_test_predictions_co2sample.csv (paper2_modelA_co2sample_k5)
  Model B: daily_test_predictions_modelB.csv (paper2_modelB_k5)
  Both trained on CO₂ sample (~1,335 tickers), K=5 latent factors
"""

import pandas as pd
import numpy as np
from scipy import stats
import os, warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(SCRIPT_DIR)
EXPORTS_DIR = os.path.join(PAPER_DIR, 'results', 'paper2_exports')
TABLE_DIR = os.path.join(PAPER_DIR, 'results', 'tables')
os.makedirs(TABLE_DIR, exist_ok=True)

PRED_A_FILE = os.path.join(EXPORTS_DIR, 'daily_test_predictions_co2sample.csv')
PRED_B_FILE = os.path.join(EXPORTS_DIR, 'daily_test_predictions_modelB.csv')


# ============================================================
# HELPERS
# ============================================================
def newey_west_se(x, max_lag=None):
    """Newey-West (1987) HAC standard error for the mean of x."""
    n = len(x)
    if n < 10:
        return np.std(x, ddof=1) / np.sqrt(n)
    if max_lag is None:
        max_lag = int(np.floor(4 * (n / 100) ** (2/9)))  # Andrews (1991) rule
    mu = x.mean()
    dm = x - mu
    gamma0 = np.sum(dm**2) / n
    nw_var = gamma0
    for lag in range(1, max_lag + 1):
        w = 1 - lag / (max_lag + 1)  # Bartlett kernel
        gamma_lag = np.sum(dm[lag:] * dm[:-lag]) / n
        nw_var += 2 * w * gamma_lag
    se = np.sqrt(nw_var / n)
    return se


def clark_west_test(e1, e2):
    """
    Clark-West (2007) test for nested models.

    H₀: Model A (restricted) is correct
    H₁: Model B (unrestricted) has superior predictive ability

    CW statistic uses the adjustment:
      d_t = e1_t² - [e2_t² - (f1_t - f2_t)²]
          = e1_t² - e2_t² + (f1_t - f2_t)²

    Since f1 - f2 = (y - e1) - (y - e2) = e2 - e1:
      d_t = e1_t² - e2_t² + (e2_t - e1_t)²
    """
    d = e1**2 - e2**2 + (e2 - e1)**2

    n = len(d)
    d_mean = np.mean(d)
    d_se = newey_west_se(d)  # HAC standard error

    cw_stat = d_mean / d_se if d_se > 1e-15 else 0
    p_value = 1 - stats.norm.cdf(cw_stat)  # One-sided: B better than A

    return cw_stat, p_value, n


def diebold_mariano_test(e1, e2):
    """
    Diebold-Mariano (1995) test with Newey-West HAC SE.
    H₀: equal predictive accuracy (squared loss).
    """
    d = e1**2 - e2**2

    n = len(d)
    d_mean = np.mean(d)
    d_se = newey_west_se(d)  # HAC standard error

    dm_stat = d_mean / d_se if d_se > 1e-15 else 0
    p_value = 1 - stats.norm.cdf(dm_stat)  # One-sided: B better

    return dm_stat, p_value, n


def sig(p):
    if p < 0.01: return '***'
    if p < 0.05: return '**'
    if p < 0.10: return '*'
    return ''


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 80)
    print("CLARK-WEST & DIEBOLD-MARIANO TESTS (K=5)")
    print("Model A (no CO₂) vs Model B (+CO₂) — Daily OOS Predictions")
    print("=" * 80)

    # -- Load predictions --
    print(f"\n Loading predictions...")
    if not os.path.exists(PRED_A_FILE):
        print(f"   Model A predictions not found: {PRED_A_FILE}")
        return
    if not os.path.exists(PRED_B_FILE):
        print(f"   Model B predictions not found: {PRED_B_FILE}")
        return

    pred_a = pd.read_csv(PRED_A_FILE)
    pred_b = pd.read_csv(PRED_B_FILE)

    print(f"  Model A: {len(pred_a):,} rows, "
          f"{pred_a['Ticker'].nunique()} tickers, "
          f"Dates: {pred_a['Date'].min()} — {pred_a['Date'].max()}")
    print(f"  Model B: {len(pred_b):,} rows, "
          f"{pred_b['Ticker'].nunique()} tickers, "
          f"Dates: {pred_b['Date'].min()} — {pred_b['Date'].max()}")

    # -- Merge on Date + Ticker for exact match (FAIR comparison) --
    merged = pd.merge(
        pred_a.rename(columns={'Predicted': 'Pred_A', 'Actual': 'Actual_A'}),
        pred_b.rename(columns={'Predicted': 'Pred_B', 'Actual': 'Actual_B'}),
        on=['Date', 'Ticker'], how='inner'
    )
    print(f"\n  Merged: {len(merged):,} obs ({merged['Ticker'].nunique()} tickers)")

    # Verify actuals are the same
    actual_diff = (merged['Actual_A'] - merged['Actual_B']).abs().max()
    if actual_diff > 1e-6:
        print(f"   Actuals differ by up to {actual_diff:.6f} — using Actual_A")
    else:
        print(f"   Actuals match (max diff = {actual_diff:.2e})")

    actual = merged['Actual_A'].values
    pred_a_vals = merged['Pred_A'].values
    pred_b_vals = merged['Pred_B'].values

    e1 = actual - pred_a_vals  # Model A errors
    e2 = actual - pred_b_vals  # Model B errors

    # -- Extract fold year from Date --
    merged['Year'] = pd.to_datetime(merged['Date']).dt.year

    # ============================================================
    # TEST 1: FOLD-LEVEL R² COMPARISON
    # ============================================================
    print(f"\n{'='*80}")
    print("  TEST 1: FOLD-LEVEL R² COMPARISON")
    print(f"{'='*80}\n")

    r2_a_list, r2_b_list = [], []
    fold_years = sorted(merged['Year'].unique())

    print(f"  {'Fold':>10s}  {'R²(A)':>8s}  {'R²(B)':>8s}  {'ΔR²':>8s}  {'N':>10s}")
    print(f"  {'-'*50}")

    for year in fold_years:
        mask = merged['Year'] == year
        y = actual[mask]
        pa = pred_a_vals[mask]
        pb = pred_b_vals[mask]

        ss_tot = np.sum((y - y.mean())**2)
        if ss_tot < 1e-15:
            continue
        r2_a = 1 - np.sum((y - pa)**2) / ss_tot
        r2_b = 1 - np.sum((y - pb)**2) / ss_tot
        r2_a_list.append(r2_a)
        r2_b_list.append(r2_b)

        print(f"  {year:>10d}  {r2_a:>8.4f}  {r2_b:>8.4f}  {r2_b - r2_a:>+8.4f}  {mask.sum():>10,d}")

    diffs = np.array([b - a for a, b in zip(r2_a_list, r2_b_list)])
    mean_a = np.mean(r2_a_list)
    mean_b = np.mean(r2_b_list)
    mean_diff = np.mean(diffs)

    print(f"  {'-'*50}")
    print(f"  {'Mean':>10s}  {mean_a:>8.4f}  {mean_b:>8.4f}  {mean_diff:>+8.4f}")

    # Paired t-test
    t_stat, p_paired = stats.ttest_rel(r2_b_list, r2_a_list)
    p_one = p_paired / 2 if t_stat > 0 else 1 - p_paired / 2

    print(f"\n  Paired t-test: t = {t_stat:.3f}, p(one-sided) = {p_one:.4f} {sig(p_one)}")
    print(f"  All positive: {all(d > 0 for d in diffs)}")

    # ============================================================
    # TEST 2: CLARK-WEST (OBSERVATION-LEVEL)
    # ============================================================
    print(f"\n{'='*80}")
    print("  TEST 2: CLARK-WEST (2007) — Nested Model Comparison")
    print(f"{'='*80}\n")

    cw_stat, cw_p, cw_n = clark_west_test(e1, e2)
    print(f"  H₀: Model A (no CO₂) is correct")
    print(f"  H₁: Model B (+CO₂) has superior predictive ability")
    print(f"\n  CW statistic = {cw_stat:.4f}")
    print(f"  p-value (one-sided) = {cw_p:.6f} {sig(cw_p)}")
    print(f"  N observations = {cw_n:,}")

    if cw_p < 0.05:
        print(f"\n  → REJECT H₀: Model B significantly outperforms Model A")
    else:
        print(f"\n  → FAIL TO REJECT H₀: No evidence that CO₂ improves predictions")

    # ============================================================
    # TEST 3: DIEBOLD-MARIANO WITH NW HAC
    # ============================================================
    print(f"\n{'='*80}")
    print("  TEST 3: DIEBOLD-MARIANO (1995) — Newey-West HAC SE")
    print(f"{'='*80}\n")

    dm_stat, dm_p, dm_n = diebold_mariano_test(e1, e2)
    print(f"  H₀: Equal predictive accuracy (squared loss)")
    print(f"\n  DM statistic = {dm_stat:.4f}")
    print(f"  p-value (one-sided) = {dm_p:.6f} {sig(dm_p)}")
    print(f"  N observations = {dm_n:,}")

    if dm_p < 0.05:
        print(f"\n  → REJECT H₀: Models have significantly different accuracy")
    else:
        print(f"\n  → FAIL TO REJECT H₀: No significant difference in accuracy")

    # ============================================================
    # TEST 4: YEAR-BY-YEAR CLARK-WEST
    # ============================================================
    print(f"\n{'='*80}")
    print("  TEST 4: YEAR-BY-YEAR CLARK-WEST")
    print(f"{'='*80}\n")

    print(f"  {'Year':>6s}  {'CW stat':>8s}  {'p-value':>8s}  {'':<4s}  {'N':>10s}")
    print(f"  {'-'*42}")

    for year in fold_years:
        mask = merged['Year'] == year
        e1_y = e1[mask]
        e2_y = e2[mask]
        if len(e1_y) < 100:
            continue
        cw_y, p_y, n_y = clark_west_test(e1_y, e2_y)
        print(f"  {year:>6d}  {cw_y:>8.3f}  {p_y:>8.4f}  {sig(p_y):<4s}  {n_y:>10,d}")

    # ============================================================
    # ADDITIONAL: MSE COMPARISON
    # ============================================================
    print(f"\n{'='*80}")
    print("  MSE & RMSE COMPARISON")
    print(f"{'='*80}\n")

    mse_a = np.mean(e1**2)
    mse_b = np.mean(e2**2)
    rmse_a = np.sqrt(mse_a)
    rmse_b = np.sqrt(mse_b)
    pct_improve = (1 - mse_b / mse_a) * 100

    print(f"  Model A (no CO₂):  MSE = {mse_a:.8f}  RMSE = {rmse_a:.6f}")
    print(f"  Model B (+CO₂):    MSE = {mse_b:.8f}  RMSE = {rmse_b:.6f}")
    print(f"  MSE improvement:   {pct_improve:+.4f}%")

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*80}")
    print("  SUMMARY")
    print(f"{'='*80}")
    print(f"""
  Model A (K=5, no CO₂) vs Model B (K=5, +CO₂)
  ----------------------------------------------
  Paired t-test:    ΔR² = {mean_diff:+.4f}, t = {t_stat:.3f}, p = {p_one:.4f} {sig(p_one)}
  Clark-West:       CW  = {cw_stat:.4f}, p = {cw_p:.6f} {sig(cw_p)}
  Diebold-Mariano:  DM  = {dm_stat:.4f}, p = {dm_p:.6f} {sig(dm_p)}
  MSE improvement:  {pct_improve:+.4f}%

  Conclusion: {"CO₂ DOES NOT significantly improve OOS predictions" 
               if cw_p > 0.05 and dm_p > 0.05 
               else "CO₂ significantly improves OOS predictions"}
""")

    # Save
    results = {
        'mean_r2_a': mean_a, 'mean_r2_b': mean_b,
        'delta_r2': mean_diff,
        'paired_t': t_stat, 'paired_p': p_one,
        'cw_stat': cw_stat, 'cw_p': cw_p,
        'dm_stat': dm_stat, 'dm_p': dm_p,
        'mse_a': mse_a, 'mse_b': mse_b,
        'mse_improvement_pct': pct_improve,
        'n_obs': len(merged),
        'n_tickers': merged['Ticker'].nunique(),
    }
    pd.DataFrame([results]).to_csv(
        os.path.join(TABLE_DIR, 'clark_west_results.csv'), index=False)
    print(f"   Saved: clark_west_results.csv")


if __name__ == '__main__':
    main()
