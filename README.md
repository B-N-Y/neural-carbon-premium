# Replication Package — "Does the Carbon Premium Survive Nonlinear Controls?"

This package reproduces every table and figure in the manuscript. It follows the
*Energy Economics* replication policy: all programs are provided, the software and
toolboxes are identified below, and the (proprietary) data sources and how to
obtain them are documented.

---

## 1. Software and environment

- **Python** 3.10+
- Core packages (see `requirements.txt`):
  `pandas`, `numpy`, `scipy`, `statsmodels`, `linearmodels` (PanelOLS / Fama–MacBeth),
  `econml` (CausalForestDML), `matplotlib`, `seaborn`
- Neural model training (Stage 2) was run on the **TRUBA HPC cluster**
  (TÜBİTAK ULAKBİM); a single GPU node reproduces the rolling-window training.

Install: `pip install -r requirements.txt`

---

## 2. Data sources and access

| Data | Source | Access |
|------|--------|--------|
| Daily/monthly returns, market cap, fundamentals | **LSEG Workspace** | Proprietary — institutional license. Variable definitions follow Bolton and Kacperczyk (2021), Table 8. |
| Carbon emissions (Scope 1/2/3), ESG scores | **LSEG ESG** | Proprietary — same license. |
| TRBC industry classification (Industry Group and Business Sector) | **LSEG** | Proprietary — same license. `TR.TRBCIndustryGroup` / `TR.TRBCBusinessSector` per instrument. HHI and the industry fixed-effects test use Industry Group; the industry-level sorts (Table: industry) use Business Sector. |
| Fama–French 5 factors + momentum (MKT, SMB, HML, RMW, CMA, UMD) | **Kenneth French Data Library** | Public: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html |

Because LSEG data are proprietary, the raw firm-level panel cannot be
redistributed. Researchers with an LSEG license can reconstruct the panel
using the variable mapping files and the preprocessing scripts below. The
constructed analysis panel and all intermediate CSV outputs are available from
the authors upon request, subject to the license terms.

---

## 3. Reproduction — pipeline order

Run the scripts in `scripts/` in the following order (each writes CSV outputs
consumed by the next stage). Intermediate outputs are written to `results/` and
`data_clean/`; final LaTeX tables/figures to `results/tables/` and
`results/figures/`.

| Stage | Script | Produces |
|-------|--------|----------|
| Panel build | `00_preprocess_panel.py`, `01_prepare_augmented_data.py`, `02_merge_green_panel.py` | `data_clean/final_monthly_panel_clean.csv`, `results/analysis_panel_green.csv` |
| Neural predictions (TRUBA) | LSTM+CA training (companion architecture) | `data_clean/neural_predicted_returns.csv`, daily neural residuals |
| Bolton replication | `03_bolton_baseline.py` | Table: Bolton (2021) replication |
| Absorption (M1–M9) | `04_neural_cross_sectional.py` | `results/tables/table2b_neural_cross_sectional.csv` → absorption & neural-residual tables |
| Inference robustness | `05_absorption_robustness.py` | Block bootstrap, placebo tables |
| Industry-FE robustness | `05c_industry_fe_robustness.py` | Within-industry$\times$month demeaned FMB on neural residuals → `industry_fe_bolton.csv` (Table: industry-FE) |
| Bootstrap SE ratios | `05d_bootstrap_se_ratio.py` | Block-bootstrap vs Newey--West SE ratios on neural-residual CO2 coefficients → `bootstrap_vs_nw.csv` |
| Reported vs estimated | `05e_reported_vs_estimated.py` | Aswani (2024) robustness: carbon premium on self-reported vs vendor-estimated emissions → `reported_vs_estimated.csv` |
| Scope 3 robustness | `06_scope3_robustness.py` | Scope 3 table |
| Portfolio sorts / spanning | `07_portfolio_sort_spanning.py` | Single/double sorts, NP-quintile, conditional, industry, spanning tables |
| Temporal subsample | `08_temporal_subsample.py` | Pre/post-COVID table |
| Clark–West OOS test | `09_clark_west_test.py` | Clark–West result |
| Causal Forest CATE | `10_causal_forest_cate.py` | CATE table |
| VIF diagnostics | `11_vif_multicollinearity_test.py` | VIF table |
| Amihud illiquidity | `12_amihud_illiquidity.py` | Illiquidity measure (conditional sorts) |
| LaTeX tables (raw) | `13_generate_latex_tables.py` | All `results/tables/tab_*.tex` + `manuscript_values.tex` |
| Table formatting | `15_finalize_tables.py` | Rewrites `tab_*.tex` into submission format (bold captions, etc.) |
| Figures | `14_manuscript_figures.py` | `results/figures/fig*.pdf` |

> **Note.** `13_generate_latex_tables.py` reads the CSV outputs of the analysis
> scripts and writes every `tab_*.tex` file and the `manuscript_values.tex`
> macro file that the manuscript `\input`s. Run `15_finalize_tables.py`
> immediately afterwards: it applies the final submission presentation on top of
> the generated tables (bold-title captions with the methodological notes moved
> into the caption, `[htbp]` float placement, and minor fixes). The canonical
> captions live in that script and are its single source of truth. It is
> idempotent, so re-running it is safe.

---

## 4. Table / figure → script map (for reviewers)

- **Descriptive stats, correlations** (`tab_desc`, `tab_corr`) — `13_generate_latex_tables.py`
- **Bolton replication** (`tab_bolton`) — `03_bolton_baseline.py`
- **Absorption M1–M9** (`tab_absorption`) — `04_neural_cross_sectional.py`
- **Neural-residual FMB** (`tab_neural_resid`) — `04_neural_cross_sectional.py`
- **NP-quintile / conditional / industry / spanning** (`tab_np_quintile`, `tab_conditional`, `tab_industry`, `tab_spanning`) — `07_portfolio_sort_spanning.py`
- **Causal Forest CATE** (`tab_cate`) — `10_causal_forest_cate.py`
- **Placebo** (`tab_placebo`) — `05_absorption_robustness.py`
- **Industry-FE robustness** (`tab_industry_fe`) — `05c_industry_fe_robustness.py`
- **Temporal subsample** (`tab_temporal`) — `08_temporal_subsample.py`
- **VIF** (`tab_vif`) — `11_vif_multicollinearity_test.py`
- **Scope 3** (`tab_scope3_robustness`) — `06_scope3_robustness.py`
- **Figures 1–5** — `14_manuscript_figures.py`

---

## 5. Notes

- Use of interactive software is avoided; all results derive from the scripts above.
- Random seeds are fixed in the Causal Forest and bootstrap procedures for
  reproducibility.
- Variable definitions and the control set follow Bolton and Kacperczyk (2021),
  Table 8.
