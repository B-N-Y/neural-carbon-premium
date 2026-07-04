#!/usr/bin/env python3
"""
9_finalize_tables.py — Post-process generated LaTeX tables into submission format.

WHY: the data-generating scripts (8_generate_latex_tables.py, 5b_scope3_robustness.py,
30_vif_multicollinearity_test.py) emit correct NUMBERS but a raw layout (short caption +
tablenotes below, sometimes wrapped in threeparttable). This finalizer applies the fixed
presentation ON TOP of the freshly generated .tex files, so regeneration never loses it:

  1. replaces each caption with the canonical bold-title caption stored below,
  2. removes the now-redundant tablenotes / threeparttable wrappers (content is in the caption),
  3. forces float placement to [htbp] (no forced [H] gaps),
  4. repairs a stray TAB-instead-of-backslash bug in tab_placebo (\times / \text).

USAGE (run AFTER the generators):
    python3 scripts/13_generate_latex_tables.py
    python3 scripts/06_scope3_robustness.py
    python3 scripts/11_vif_multicollinearity_test.py
    python3 scripts/15_finalize_tables.py     # <-- this script

The bold-title captions are the single source of truth for table captions; edit them here.
Idempotent: running twice is a no-op.
"""
import os, re, glob

TEX_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'results', 'tables')
os.makedirs(TEX_DIR, exist_ok=True)

# --- canonical captions (bold title + description); one per table ---------------
CAPTIONS = {
    'tab_absorption': r'''\caption{\textbf{Carbon Premium Absorption: Nested Model Specifications.} PanelOLS with year-month FE, double-clustered SE (firm $\times$ year). Dependent variable: monthly return $\times$ 100. CO$_2$ = $\log(\text{Total CO}_2)$. M1: Bolton controls (size, book-to-market, leverage, ROE, investment-to-assets, sales-to-price, log(PPE), momentum, volatility, HHI, and closely-held ownership). M2: ICA latent factor betas only. M3: Bolton + ICA LF betas. M4: Bolton + FF5 betas. M5: Bolton + ICA LF + FF5. M6: neural prediction ($\hat{R}^{NN}$) only. M7: Bolton + neural prediction. M1*: Bolton on the M7 matched sample. M8: CO$_2$ + Size only (lower bound). M9: Bolton + CO$_2 \times$ SIZE and CO$_2 \times$ BM interactions; the $\gamma_{\text{CO}_2}$ reported for M9 is the CO$_2$ main effect in the interaction model (evaluated at mean size and book-to-market), while the interaction coefficients themselves are reported in the text. Matched sample: 2018--2024, $N = 58{,}514$. $R^2$ = within-$R^2$ (net of time FE); total $R^2 \approx 0.28$ for M1. The higher within-$R^2$ for M6--M7 reflects the inclusion of $\hat{R}^{NN}$, which is trained to predict the dependent variable. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.}''',
    'tab_bolton': r'''\caption{\textbf{Bolton Replication: Pooled OLS and Fama--MacBeth with Year-Month Fixed Effects.} Replication of the Bolton and Kacperczyk (2021) specification on the full CO$_2$ sample (2010--2025), reported one row per emission measure. Each cell gives the CO$_2$ coefficient ($\gamma$, monthly return $\times 100$) with its $t$-statistic in parentheses below. Columns report three estimators under the full Bolton control set (size, book-to-market, leverage, ROE, investment-to-assets, momentum, volatility, HHI, log(PPE), CAPM beta, sales growth, and EPS growth) augmented with closely-held ownership as elsewhere in the paper: pooled OLS with year-month fixed effects (OLS), the same with TRBC Industry Group fixed effects added (OLS + Ind FE), and Fama--MacBeth (FMB). Standard errors are double-clustered (firm $\times$ year) for OLS and Newey--West for FMB. The number of firm-month observations ranges from roughly 74,000 ($\Delta$CO$_2$) to 88,000 (total CO$_2$ and intensity). Coefficients are robust to excluding Sales Growth and EPS growth, which expands the sample without materially changing the estimates. Carbon intensity is measured as CO$_2$/revenue and enters on its raw scale, so its coefficient is not comparable in magnitude to the log-emission coefficients. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.}''',
    'tab_bootstrap': r'''\caption{\textbf{Bootstrap Hausman Test: Is the Absorption Statistically Significant?} Block bootstrap test of whether the CO$_2$ coefficient reduction from M1 to M7 is statistically significant. Resampling entire calendar months with replacement preserves cross-sectional dependence. $\Delta\gamma = \gamma_{\text{M1}} - \gamma_{\text{M7}}$: difference in CO$_2$ coefficients between M1 (Bolton controls only) and M7 (Bolton + neural prediction). Matched sample: 2018--2024, $N = 58{,}514$ firm-months. $p$-value: fraction of bootstrap replications with $\Delta\gamma \leq 0$.}''',
    'tab_cate': r'''\caption{\textbf{Causal Forest: Heterogeneous Treatment Effects of High CO$_2$.} CausalForestDML (econml; $B = 1{,}000$ trees, $cv = 5$). Treatment: top vs.\ bottom tercile of lagged carbon intensity. Outcome: annual mean neural residual ($\hat{\varepsilon}^{NN}$) or raw return; confounders are the 11 Bolton controls, all $z$-scored. $t$-statistics use CLT standard errors of the forest-based CATE estimates, cluster-adjusted by $\sqrt{7}$ for within-firm dependence. The analysis covers 3,933 firm-years (983 firms); feature importances are reported in the text. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.}''',
    'tab_conditional': r'''\caption{\textbf{Conditional Carbon Premium by Firm Characteristics.} Independent 3$\times$3 double sorts. Each month, firms sorted into terciles by the conditioning variable (columns) and terciles by $\log(\text{Total CO}_2)$. Cell values = CO$_2$ spread (High $-$ Low CO$_2$ tercile within each conditioning group, \%/month). H$-$L = spread-of-spreads (highest minus lowest conditioning group). Both panels use the matched sample (2018--2024, $T=77$ months). Panel A: raw returns $R_{i,t}$. Panel B: neural residuals $\hat{\varepsilon}^{NN}_{i,t} = R_{i,t} - \hat{R}^{NN}_{i,t}$ (rolling-window test periods). $t$-statistics: Newey--West (1987) HAC, 6 lags. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.}''',
    'tab_corr': r'''\caption{\textbf{Cross-Sectional Correlation Matrix.} CO$_2$ = $\log(\text{Total CO}_2)$; Size $= \log(\text{Market Cap})$; B/M is book-to-market; Lev is leverage; Vol is return volatility; CH is closely-held ownership; Illiq $= \log(\text{Amihud illiquidity})$. Pooled Pearson correlations, $N = 98,878$ firm-months with non-missing CO$_2$ and Amihud.}''',
    'tab_desc': r'''\caption{\textbf{Descriptive Statistics (Monthly Panel, 2018--2024).} Monthly firm-level observations over the LSTM out-of-sample period (2018--2024, $T = 77$ months) for the variables used in the asset-pricing models. Returns, firm characteristics, and carbon emissions are all obtained from LSEG. CO$_2$ = $\log(\text{Total CO}_2)$; Size = $\log(\text{Market Cap})$; carbon intensity is reported in units of $10^{-3}$. All variables are winsorised at the 1st/99th percentiles.}''',
    'tab_fold_r2': r'''\caption{\textbf{LSTM+CA Model Performance: Fold-by-Fold $R^2$.} LSTM with Cross-Attention (CA) architecture, $K = 5$ latent factors. 13 firm-level inputs: Size, Book-to-Market, Momentum, Volatility, Leverage, ROE, Investment-to-Assets, Sales-to-Price, log(PPE), HHI, closely-held ownership, CAPM Beta, and Market Excess Return. Rolling-window design: each fold expands the training set by one year, validates on the following year, and tests on the year after that. $R^2_{\text{val}}$: daily total (pooled) $R^2$ on validation data. $R^2_{\text{test}}$: daily total (pooled) $R^2$ on held-out test data. Positive $R^2$ indicates the model outperforms the historical mean forecast.}''',
    'tab_industry': r'''\caption{\textbf{Industry-Level Carbon Premium.} Monthly CO$_2$ High-minus-Low spread (equal-weighted) within each TRBC Business Sector (matched sample, 2018--2024). Each month, firms within an industry sorted into terciles by $\log(\text{Total CO}_2)$; the spread is the difference in mean returns between the top and bottom terciles. FF5 $\alpha$: intercept from time-series regression of the H-L spread on Fama--French five factors (MKT, SMB, HML, RMW, CMA). Business sectors with $\geq$20 firms per month and $\geq$30 months only. Returns in \%/month; Newey--West (1987) HAC $t$-statistics (6 lags) in parentheses, with significance stars on the point estimate. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.}''',
    'tab_industry_fe': r'''\caption{\textbf{Industry Fixed Effects Robustness.} Monthly Fama--MacBeth regressions on neural residuals ($\hat{\varepsilon}^{NN} = R - \hat{R}^{NN}$, matched sample 2018--2024, $T = 77$ months), with the full Bolton control set. Following Bolton and Kacperczyk (2023), the ``With Ind.~FE'' column demeans every model variable within each industry$\times$month cell (52 industries) before the Fama--MacBeth procedure. Entries are the $t$-statistic of each CO$_2$ measure; $\Delta t = t_{\text{IndFE}} - t_{\text{NoIndFE}}$. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.}''',
    'tab_neural_compare': r'''\caption{\textbf{Carbon Premium: Raw Returns vs.\ Neural Residuals.} Monthly Fama--MacBeth regressions (2018--2024, $T = 77$ cross-sections) with Newey--West (4 lag) HAC standard errors. ``Raw Returns'' uses $R_{it}$ as dependent variable; ``Neural Residual'' uses $\hat{\varepsilon}^{NN}_{it} = R_{it} - \hat{R}^{NN}_{it}$. Bolton controls included in all specifications: size, book-to-market, leverage, ROE, investment-to-assets, sales-to-price, log(PPE), momentum, volatility, HHI, and closely-held ownership. $R^2$ = average cross-sectional $R^2$. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.}''',
    'tab_neural_resid': r'''\caption{\textbf{Carbon Measures in Raw Returns vs.\ Neural Residuals.} Panel A: PanelOLS with year-month FE, double-clustered SE (firm $\times$ year); $R^2$ = within-$R^2$; $N$ = firm-month observations. Panel B: Fama--MacBeth on neural residuals ($\hat{\varepsilon}^{NN} = R - \hat{R}^{NN}$) with Newey--West (4 lag) SE; $R^2$ = average cross-sectional $R^2$; $T$ = number of monthly cross-sections. The two panels employ different estimation methods because neural predictions are available only for 2018--2024 (rolling window test periods), while Panel A uses the full sample. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.}''',
    'tab_np_quintile': r'''\caption{\textbf{Carbon Premium by Neural Prediction Quintile.} Independent $5 \times 3$ double sorts on the matched sample (2018--2024, $T = 77$ months). Each month, firms sorted into quintiles by LSTM neural predicted return $\hat{R}^{NN}$ (Q1 = lowest prediction, Q5 = highest) and into terciles by $\log(\text{Total CO}_2)$ within each quintile. CO$_2$ H-L = High $-$ Low CO$_2$ tercile equal-weighted monthly return (\%/month). FF5 $\alpha$: intercept from time-series regression of H-L spread on Fama--French five factors (MKT, SMB, HML, RMW, CMA). $t$-statistics: Newey--West (1987) HAC, 6 lags. Q5$-$Q1: Newey--West on paired difference series. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.}''',
    'tab_placebo': r'''\caption{\textbf{Placebo Test: Does neural prediction Absorb Known Risk Premiums?} PanelOLS with year-month FE and double-clustered SE (firm $\times$ year). Matched sample: 2018--2024, $N = 58{,}514$. ``Without Neural: $\gamma$ coefficient for each row-variable with Bolton controls only. ``With Neural: adds neural prediction ($\hat{R}^{NN}$) as an additional control. Absorption (\%) = $100 \times (1 - \gamma_{\text{with}} / \gamma_{\text{without}})$. The Size, Value, and Momentum premia are the SMB, HML, and momentum (UMD) proxies, respectively; CO$_2$ is the paper's test variable. Size, BM, and Momentum are LSTM inputs and should therefore be absorbed (a valid placebo), whereas CO$_2$ is not an LSTM input, so its absorption is informative rather than mechanical. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.}''',
    'tab_scope3_robustness': r'''\caption{\textbf{Scope 3 Robustness: Carbon Premium Absorption Across Emission Scopes.} PanelOLS with year-month FE, double-clustered SE (firm $\times$ year). Dependent variable: monthly return $\times$ 100. M1: Bolton controls (size, book-to-market, leverage, ROE, investment-to-assets, sales-to-price, log(PPE), momentum, volatility, HHI, and closely-held ownership). M7: Bolton controls + $\hat{R}^{NN}$ (LSTM neural predicted return). Scope 3 emissions in LSEG are predominantly vendor-estimated rather than self-reported; results should be interpreted with this measurement caveat. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.}''',
    'tab_spanning': r'''\caption{\textbf{Spanning Regression: CO$_2$ Long-Short Portfolio.} Dependent variable: monthly return of the High$-$Low CO$_2$ tercile portfolio (equal-weighted, matched sample 2018--2024, $T = 77$ months). Each row reports the CO$_2$ spread alpha against a different benchmark: the CAPM, the Fama--French five-factor model (FF5), the neural high-minus-low factor (Neural HmL), and FF5 augmented with the neural factor (FF5 + Neural). A significant $\alpha$ indicates the CO$_2$ spread is not spanned by the included factors; $\alpha \approx 0$ indicates full spanning. OLS with Newey--West (1987) HAC standard errors, 6 lags. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.}''',
    'tab_temporal': r'''\caption{\textbf{Temporal Subsample Analysis: Pre- vs.\ Post-COVID Carbon Premium.} PanelOLS with time fixed effects and double-clustered standard errors (firm $\times$ time). Split point: December 2020. Sample sizes: full sample $N=\valTempFullN$ (\valTempFullFirms{} firms, \valTempFullMonths{} months); pre-COVID $N=\valTempPreN$ (\valTempPreFirms{} firms, \valTempPreMonths{} months); post-COVID $N=\valTempPostN$ (\valTempPostFirms{} firms, \valTempPostMonths{} months). $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.}''',
    'tab_vif': r'''\caption{\textbf{Variance Inflation Factor (VIF) Diagnostics for CO$_2$ Across Model Specifications.} VIF of $\log(\text{CO}_2)$ in each model specification from Table~\ref{tab:absorption}. Max VIF: highest VIF among all regressors in that model. VIF $< 5$ indicates no multicollinearity concern. Condition numbers below 30 indicate a well-conditioned design matrix. All variables standardized before computation.}''',
}

# Tables that overflow the text block: shrink font + tighten row height so they fit one page.
FIT = {'tab_cate': '0.85', 'tab_industry': '0.95', 'tab_temporal': '0.92'}
# Tables whose first (label) column is wide: tighten inter-column spacing to fit the width.
TIGHT_COLSEP = {'tab_industry': '4pt'}

def _finalize(path):
    name = os.path.basename(path)[:-4]
    raw = open(path, encoding='utf-8').read()
    raw = raw.replace('\t'+'imes', r'\times').replace('\t'+'ext', r'\text')  # placebo TAB repair
    lines = raw.split('\n')
    out, skip = [], False
    for ln in lines:
        s = ln.strip()
        if s.startswith(r'\begin{tablenotes}'):
            skip = True; continue
        if skip:
            if s.startswith(r'\end{tablenotes}'):
                skip = False
            continue
        if s in (r'\begin{threeparttable}', r'\end{threeparttable}'):
            continue
        if name in FIT and s in (r'\small', r'\footnotesize'):
            continue
        if name in FIT and s.startswith(r'\renewcommand{\arraystretch}'):
            continue
        if name in FIT and s.startswith(r'\begin{tabular}'):
            indent = ln[:len(ln) - len(ln.lstrip())]
            out.append(indent + r'\footnotesize')
            out.append(indent + r'\renewcommand{\arraystretch}{%s}' % FIT[name])
            if name in TIGHT_COLSEP:
                out.append(indent + r'\setlength{\tabcolsep}{%s}' % TIGHT_COLSEP[name])
            out.append(ln)
            continue
        ln = ln.replace(r'\begin{table}[H]', r'\begin{table}[htbp]')
        if re.match(r'^\s*\\caption\{', ln) and name in CAPTIONS:
            indent = ln[:len(ln) - len(ln.lstrip())]
            out.append(indent + CAPTIONS[name])
            continue
        out.append(ln)
    open(path, 'w', encoding='utf-8').write('\n'.join(out))

def main():
    n = 0
    for path in sorted(glob.glob(os.path.join(TEX_DIR, 'tab_*.tex'))):
        _finalize(path); n += 1
        print('  finalized', os.path.basename(path))
    print(f'Done: {n} tables finalized.')

if __name__ == '__main__':
    main()
