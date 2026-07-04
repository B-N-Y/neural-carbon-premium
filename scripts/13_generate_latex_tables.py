r"""
13_generate_latex_tables.py  —  AUTO-GENERATE ALL MANUSCRIPT LATEX TABLES FROM CSVs
===================================================================================
Reads CSV result files and the clean panel to produce .tex table files and a
manuscript_values.tex file with \newcommand macros for every number cited in prose.

Run this script after ANY analysis script to regenerate tables + inline values.

Output:
    ../results/tables/tab_*.tex   — LaTeX tables for \input{}
    ../results/manuscript_values.tex — \newcommand macros for inline values
"""

import pandas as pd
import numpy as np
import os
import json

# ------------------------------------------------------------------
# PATHS
# ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR  = os.path.dirname(SCRIPT_DIR)
CSV_DIR    = os.path.join(PAPER_DIR, 'results', 'tables')   # CSVs + .tex in one place
TEX_DIR    = os.path.join(PAPER_DIR, 'results', 'tables')   # primary output
DATA_DIR   = os.path.join(PAPER_DIR, 'data_clean')
VALUES_PATH = os.path.join(PAPER_DIR, 'results', 'manuscript_values.tex')
os.makedirs(TEX_DIR, exist_ok=True)

# Collected values dictionary  – populated as tables are built
VALUES = {}


def latex_safe(s):
    """Sanitize string for LaTeX: escape underscores, replace Unicode."""
    s = str(s)
    s = s.replace('Δ', '$\\Delta$')
    s = s.replace('₂', '$_2$')
    
    # Custom variable name formatting for professional LaTeX output
    mappings = {
        'NEURAL_PRED': r'Neural Prediction ($\hat{R}^{NN}$)',
        'NEURAL_RESID': r'Neural Residual ($\hat{\varepsilon}^{NN}$)',
        'LOG_CO2_TOTAL': r'$\log(\text{Total CO}_2)$',
        'CARBON_INTENSITY': r'Carbon Intensity',
        'DELTA_CO2': r'Emission Growth ($\Delta\text{CO}_2$)',
        'LOG_SCOPE1': r'$\log(\text{Scope 1})$',
        'LOG_SCOPE2': r'$\log(\text{Scope 2})$',
        'LOG_SCOPE3': r'$\log(\text{Scope 3})$',
        'TOBINS_Q': r"Tobin's $q$",
        'LOG_PPE': r'$\log(\text{PPE})$',
        'D_REPORTED': r'$D^{\text{Reported}}$',
        'ENV_SCORE': r'Environmental Score',
        'SALESGR': r'Sales-to-Price',
        'FIRM_AGE': r'Firm Age',
        'ANALYSTS': r'Number of Analysts',
        'SIZE': r'Size ($\log(\text{ME})$)',
        'BM': r'Book-to-Market (BM)',
        'LEVERAGE': r'Leverage',
        'IO': r'Closely-held ownership',
        'BETA': r'CAPM Beta',
        'ROE': r'Return on Equity (ROE)',
        'INVEST_A': r'Investment-to-Assets',
        'VOLAT': r'Volatility',
        'MOM': r'Momentum',
        'Total CO2': r'Total CO$_2$',
        'Total CO_2': r'Total CO$_2$',
        'CO2/Revenue': r'Carbon Intensity',
        'CO_2/Revenue': r'Carbon Intensity',
        'CO$_2$/Revenue': r'Carbon intensity',
        'Delta CO2': r'$\Delta$CO$_2$',
        'Delta CO_2': r'$\Delta$CO$_2$',

    }
    
    # Exact matching first
    if s in mappings:
        return mappings[s]
        
    # Substring replaces or prefix checks
    for k, v in mappings.items():
        if k in s:
            s = s.replace(k, v)
            
    s = s.replace('_', '\\_')
    s = s.replace('nan', '--')
    # The blanket underscore escape above also escapes the LaTeX subscripts that
    # the Unicode/mapping steps introduced (e.g. $_2$, \text{CO}_2); restore them.
    s = s.replace(r'$\_2$', r'$_2$')
    s = s.replace(r'}\_2', r'}_2')
    s = s.replace(r'\_{', r'_{')
    return s


def stars(t):
    """Return significance stars for a t-statistic."""
    if pd.isna(t):
        return ''
    at = abs(float(t))
    if at >= 2.576:
        return '^{***}'
    elif at >= 1.960:
        return '^{**}'
    elif at >= 1.645:
        return '^{*}'
    return ''


def fmt_coef(val, dec=3):
    """Format coefficient."""
    if pd.isna(val):
        return ''
    return f'{float(val):.{dec}f}'


def fmt_t(val):
    """Format t-statistic in parentheses with stars."""
    if pd.isna(val):
        return ''
    return f'({float(val):.2f}){stars(val)}'


def fmt_pct(val, dec=2):
    """Format as percentage."""
    return f'{float(val)*100:+.{dec}f}'


# ==================================================================
#  TABLE 1: DESCRIPTIVE STATISTICS  (tab_desc.tex)
# ==================================================================
def gen_tab_desc():
    """Descriptive statistics from the clean monthly panel."""
    panel_path = os.path.join(DATA_DIR, 'final_monthly_panel_clean.csv')
    if not os.path.exists(panel_path):
        print('    tab_desc.tex SKIPPED (panel CSV not found)')
        return

    df = pd.read_csv(panel_path, usecols=[
        'MonthlyReturn', 'LOG_CO2_TOTAL', 'CARBON_INTENSITY',
        'SIZE', 'BM', 'MOM', 'VOLAT', 'BETA',
        'ROE', 'INVEST_A', 'SALESGR', 'LOG_PPE',
        'LEVERAGE', 'HHI', 'IO',
    ])

    rows_spec = [
        ('Monthly Return',    'MonthlyReturn',    3),
        ('Log Total CO$_2$',  'LOG_CO2_TOTAL',    2),
        ('Carbon Intensity ($\\times 10^{3}$)',  'CARBON_INTENSITY',  3),
        ('Ln(Market Cap)',    'SIZE',              2),
        ('Book-to-Market',   'BM',                2),
        ('Momentum',         'MOM',               3),
        ('Volatility',       'VOLAT',             3),
        ('CAPM Beta',        'BETA',              2),
        ('ROE',              'ROE',               3),
        ('Investment/A',     'INVEST_A',          3),
        ('Sales-to-Price',   'SALESGR',           3),
        ('Log PPE',          'LOG_PPE',           2),
        ('Leverage',         'LEVERAGE',          2),
        ('HHI',              'HHI',               3),
        ('Closely-held',     'IO',                2),
    ]

    tex = []
    tex.append(r'\begin{table}[H]')
    tex.append(r'\centering')
    tex.append(r'\caption{Descriptive Statistics (Monthly Panel, 2018--2024)}')
    tex.append(r'\label{tab:desc}')
    tex.append(r'\small')
    tex.append(r'\begin{tabular}{lcccccc}')
    tex.append(r'\toprule')
    tex.append(r' & Mean & SD & P25 & P50 & P75 & $N$ \\')
    tex.append(r'\midrule')

    for label, col, dec in rows_spec:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if col == 'CARBON_INTENSITY':
            s = s * 1000.0  # display in units of 10^{-3} for readability
        n = len(s)
        if n == 0:
            continue
        mean = s.mean()
        sd   = s.std()
        p25  = s.quantile(0.25)
        p50  = s.median()
        p75  = s.quantile(0.75)
        tex.append(
            f'{label} & ${mean:.{dec}f}$ & ${sd:.{dec}f}$ & '
            f'${p25:.{dec}f}$ & ${p50:.{dec}f}$ & ${p75:.{dec}f}$ & '
            f'{n:,} \\\\'
        )

    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(r'\begin{tablenotes}')
    tex.append(r'\small')
    tex.append(r'\item Monthly firm-level observations from the LSTM out-of-sample period (2018--2024, $T = 77$ months). Returns and firm characteristics from LSEG Datastream and Eikon; carbon emissions from LSEG ESG. CO$_2$ = $\log(\text{Total CO}_2)$; SIZE = $\log(\text{Market Cap})$; BM = Book-to-Market; NEURAL\_PRED = LSTM-predicted monthly return; NEURAL\_RESID = $R_{i,t} - \hat{R}^{NN}_{i,t}$. All variables winsorised at 1\%/99\%.')
    tex.append(r'\end{tablenotes}')
    tex.append(r'\end{table}')

    with open(os.path.join(TEX_DIR, 'tab_desc.tex'), 'w') as f:
        f.write('\n'.join(tex))
    print('   tab_desc.tex')


# ==================================================================
#  TABLE 2: BOLTON REPLICATION  (tab_bolton.tex)
# ==================================================================
def gen_tab_bolton():
    """Bolton replication: one row per emission measure, estimators as columns, full Bolton controls only.

    The Core (excl. SALESGR/EPSGR) variant is intentionally dropped from the manuscript table
    (near-identical coefficients, larger sample); it remains in the CSV for reference.
    """
    df = pd.read_csv(os.path.join(CSV_DIR, 'table2_bolton_replication.csv'))
    df = df[df['Method'].str.contains('Full Bolton')].copy()

    def spec_of(m):
        if 'Fama-MacBeth' in m: return 'FMB'
        if 'No IndFE' in m:     return 'OLS'
        return 'OLSFE'          # '+ IndFE'
    df['spec'] = df['Method'].map(spec_of)

    # (panel label, [(CSV CO2 name, display label, decimals)])
    panels = [
        (r'\textit{Panel A: Emission levels}', [
            ('Scope 1', 'Scope 1', 3), ('Scope 2', 'Scope 2', 3),
            ('Total CO2', r'Total CO$_2$', 3)]),
        (r'\textit{Panel B: Emission growth}', [
            ('ΔCO2', r'$\Delta$CO$_2$', 3)]),
        (r'\textit{Panel C: Carbon intensity}', [
            ('CO2/Revenue', 'Carbon intensity', 1)]),
    ]

    def cell(co2, spec, dec):
        r = df[(df['CO2'] == co2) & (df['spec'] == spec)]
        if r.empty:
            return '', ''
        coef, t = r.iloc[0]['coef'], r.iloc[0]['t_stat']
        return f'${fmt_coef(coef, dec)}{stars(t)}$', f'$({float(t):.2f})$'

    caption = (
        r'\caption{\textbf{Bolton Replication: Pooled OLS and Fama--MacBeth with Year-Month Fixed Effects.} '
        r'Replication of the Bolton and Kacperczyk (2021) specification on the full CO$_2$ sample (2010--2025), '
        r'reported one row per emission measure. Each cell gives the CO$_2$ coefficient ($\gamma$, monthly return '
        r'$\times 100$) with its $t$-statistic in parentheses below. Columns report three estimators under the full '
        r'Bolton control set (SIZE, BM, LEV, ROE, INVEST, MOM, VOLAT, HHI, log(PPE), SALESGR, EPSGR): pooled OLS with '
        r'year-month fixed effects (OLS), the same with TRBC Industry Group fixed effects added (OLS + Ind FE), and '
        r'Fama--MacBeth (FMB). Standard errors are double-clustered (firm $\times$ year) for OLS and Newey--West for '
        r'FMB. The number of firm-month observations ranges from roughly 74,000 ($\Delta$CO$_2$) to 88,000 (total '
        r'CO$_2$ and intensity). Coefficients are robust to excluding SALESGR and EPSGR, which expands the sample '
        r'without materially changing the estimates. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.}'
    )

    tex = []
    tex.append(r'\begin{table}[htbp]')
    tex.append(r'\centering')
    tex.append(caption)
    tex.append(r'\label{tab:bolton}')
    tex.append(r'\small')
    tex.append(r'\begin{tabular}{lccc}')
    tex.append(r'\toprule')
    tex.append(r'Emission measure & OLS & OLS + Ind FE & FMB \\')
    tex.append(r'\midrule')
    for pi, (panel_label, measures) in enumerate(panels):
        if pi > 0:
            tex.append(r'\addlinespace')
        tex.append(f'\\multicolumn{{4}}{{l}}{{{panel_label}}} \\\\')
        for co2, disp, dec in measures:
            c_ols, t_ols = cell(co2, 'OLS', dec)
            c_fe, t_fe   = cell(co2, 'OLSFE', dec)
            c_fmb, t_fmb = cell(co2, 'FMB', dec)
            tex.append(f'{disp} & {c_ols} & {c_fe} & {c_fmb} \\\\')
            tex.append(f'        & {t_ols} & {t_fe} & {t_fmb} \\\\')
    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(r'\end{table}')

    with open(os.path.join(TEX_DIR, 'tab_bolton.tex'), 'w') as f:
        f.write('\n'.join(tex))
    print('   tab_bolton.tex')


# ==================================================================
#  TABLE 3: ABSORPTION TEST M1-M7  (tab_absorption.tex)
# ==================================================================
def gen_tab_absorption():
    """Neural absorption: M1-M7 nested models."""
    df = pd.read_csv(os.path.join(CSV_DIR, 'table2b_neural_cross_sectional.csv'))
    flagship = df[df['Test'] == 'PanelOLS_Flagship'].copy()

    # ---- collect inline values for ALL models ----
    model_map = {
        'M1': 'Mone', 'M2': 'Mtwo', 'M3': 'Mthree', 'M4': 'Mfour',
        'M5': 'Mfive', 'M6': 'Msix', 'M7': 'Mseven',
    }
    for prefix, key in model_map.items():
        m = flagship[flagship['Model'].str.startswith(prefix)]
        if len(m):
            r = m.iloc[0]
            VALUES[f'{key}Coef'] = f'{float(r["coef"]):.3f}'
            VALUES[f'{key}T']    = f'{float(r["t"]):.2f}'

    # M1* matched sample
    m1star = flagship[flagship['Model'].str.contains('M1\\*', regex=True)]
    if len(m1star):
        r = m1star.iloc[0]
        VALUES['MoneStarCoef'] = f'{float(r["coef"]):.3f}'
        VALUES['MoneStarT']    = f'{float(r["t"]):.2f}'
        VALUES['MoneStarN']    = f'{int(r["n"]):,}'

    # M9 interaction coefficients (CO2xSIZE, CO2xBM) -- persisted, not hardcoded
    m9 = flagship[flagship['Model'].str.startswith('M9')]
    if len(m9):
        r = m9.iloc[0]
        for col, key in [('CO2xSIZE_coef', 'MnineSizeCoef'), ('CO2xSIZE_t', 'MnineSizeT'),
                         ('CO2xBM_coef', 'MnineBMCoef'), ('CO2xBM_t', 'MnineBMT')]:
            if col in r and pd.notna(r[col]):
                VALUES[key] = f'{float(r[col]):.2f}' if key.endswith('T') else f'{float(r[col]):.4f}'

    m1 = flagship[flagship['Model'].str.startswith('M1')]
    m7 = flagship[flagship['Model'].str.startswith('M7')]
    if len(m1) and len(m7):
        drop = (1 - float(m7.iloc[0]['coef']) / float(m1.iloc[0]['coef'])) * 100
        VALUES['AbsorptionPct'] = f'{drop:.0f}'

    # Matched-sample absorption: M1* → M7
    if len(m1star) and len(m7):
        drop_matched = (1 - float(m7.iloc[0]['coef']) / float(m1star.iloc[0]['coef'])) * 100
        VALUES['AbsorptionMatchedPct'] = f'{drop_matched:.0f}'

    # ---- MultiCO2 / neural residual t-stats ----
    multi = df[df['Test'].str.startswith('MultiCO2')]
    for _, row in multi.iterrows():
        import re as _re
        tag = row['Test'].replace('MultiCO2_', '').replace(' ', '')
        model_name = str(row['Model']).replace(' ', '').replace('/', '')
        raw_key = f'NR{model_name}{tag}'
        # Sanitize: only ASCII letters allowed in LaTeX \newcommand names
        vkey = _re.sub(r'[^A-Za-z]', '', raw_key)[:30]
        VALUES[vkey + 'T'] = f'{float(row["t"]):.2f}'

    # ---- Neural Residual FMB: NR1 (no controls), NR2 (Bolton controls) ----
    nr_fmb = df[df['Test'] == 'NeuralResid_FMB']
    nr1 = nr_fmb[nr_fmb['Model'].str.startswith('NR1')]
    nr2 = nr_fmb[nr_fmb['Model'].str.startswith('NR2')]
    if len(nr1):
        VALUES['NRnoCtrlT'] = f'{float(nr1.iloc[0]["t"]):.2f}'
    if len(nr2):
        VALUES['NRboltonCtrlT'] = f'{float(nr2.iloc[0]["t"]):.2f}'

    # ---- build table ----
    tex = []
    tex.append(r'\begin{table}[H]')
    tex.append(r'\centering')
    tex.append(r'\caption{Carbon Premium Absorption: Nested Model Specifications}')
    tex.append(r'\label{tab:absorption}')
    tex.append(r'\small')
    tex.append(r'\begin{tabular}{lrrrr}')
    tex.append(r'\toprule')
    tex.append(r'Model & $\gamma_{\text{CO}_2}$ & $t$ & $R^2$ & $N$ \\')
    tex.append(r'\midrule')

    for _, row in flagship.iterrows():
        model  = latex_safe(row['Model'])
        c      = fmt_coef(row['coef'], 4)
        t      = fmt_t(row['t'])
        r2_val = row['within_r2'] if pd.notna(row['within_r2']) else row.get('avg_r2', np.nan)
        r2     = f'{float(r2_val):.3f}' if pd.notna(r2_val) else '--'
        n      = f'{int(row["n"]):,}' if pd.notna(row['n']) else '--'
        tex.append(f'  {model:<35s} & ${c}$ & ${t}$ & {r2} & {n} \\\\')

    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(r'\begin{tablenotes}')
    tex.append(r'\small')
    tex.append(r'\item PanelOLS with year-month FE, double-clustered SE (firm $\times$ year). Dependent variable: monthly return $\times$ 100. CO$_2$ = $\log(\text{Total CO}_2)$. M1: Bolton controls (SIZE, BM, LEV, ROE, INVEST, SALESGR, log(PPE), MOM, VOLAT, HHI, IO). M2: ICA latent factor betas only. M3: Bolton + ICA LF betas. M4: Bolton + FF5 betas. M5: Bolton + ICA LF + FF5. M6: NEURAL\_PRED ($\hat{R}^{NN}$) only. M7: Bolton + NEURAL\_PRED. M1*: Bolton on the M7 matched sample. M8: CO$_2$ + Size only (lower bound). M9: Bolton + CO$_2 \times$ SIZE and CO$_2 \times$ BM interactions. Matched sample: 2018--2024, $N = 58{,}514$. $R^2$ = within-$R^2$ (net of time FE); total $R^2 \approx 0.28$ for M1. The higher within-$R^2$ for M6--M7 reflects the inclusion of $\hat{R}^{NN}$, which is trained to predict the dependent variable. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.')
    tex.append(r'\end{tablenotes}')
    tex.append(r'\end{table}')

    with open(os.path.join(TEX_DIR, 'tab_absorption.tex'), 'w') as f:
        f.write('\n'.join(tex))
    print('   tab_absorption.tex')


# ==================================================================
#  TABLE 4: NEURAL RESIDUAL FMB  (tab_neural_resid.tex)
# ==================================================================
def gen_tab_neural_resid():
    """Multi-CO2 in Raw Returns (PanelOLS) vs Neural Residuals (FMB)."""
    df = pd.read_csv(os.path.join(CSV_DIR, 'table2b_neural_cross_sectional.csv'))
    raw = df[df['Test'] == 'MultiCO2_M1_Bolton'].copy()
    nr  = df[df['Test'] == 'NeuralResid_MultiCO2_NR_Bolton'].copy()

    tex = []
    tex.append(r'\begin{table}[H]')
    tex.append(r'\centering')
    tex.append(r'\caption{Carbon Measures in Raw Returns vs.\ Neural Residuals}')
    tex.append(r'\label{tab:neural_resid}')
    tex.append(r'\small')
    tex.append(r'\begin{tabular}{llrrrr}')
    tex.append(r'\toprule')
    tex.append(r'CO$_2$ Measure & Dep.\ Var. & $\gamma$ & $t$ & $R^2$ & $N$/$T$ \\')
    tex.append(r'\midrule')

    # Panel A: Raw Returns (PanelOLS)
    tex.append(r'\multicolumn{6}{l}{\textit{Panel A: Raw Returns ($R_{i,t}$)}} \\')
    tex.append(r'\addlinespace')
    for _, row in raw.iterrows():
        co2_label = latex_safe(row['Model'])
        c  = fmt_coef(row['coef'], 4)
        t  = fmt_t(row['t'])
        r2 = f'{float(row["within_r2"]):.3f}' if pd.notna(row.get('within_r2')) else '--'
        n  = f'{int(row["n"]):,}' if pd.notna(row.get('n')) else '--'
        tex.append(f'  {co2_label:<20s} & M1: Bolton   & ${c}$ & ${t}$ & {r2} & {n} \\\\')

    tex.append(r'\addlinespace')
    # Panel B: Neural Residuals (FMB)
    tex.append(r'\multicolumn{6}{l}{\textit{Panel B: Neural Residuals ($\hat{\varepsilon}^{NN}_{i,t}$)}} \\')
    tex.append(r'\addlinespace')
    for _, row in nr.iterrows():
        co2_label = latex_safe(row['Model'])
        c  = fmt_coef(row['coef'], 4)
        t  = fmt_t(row['t'])
        r2_val = row.get('avg_r2', row.get('r2', np.nan))
        r2 = f'{float(r2_val):.3f}' if pd.notna(r2_val) else '--'
        T_val = row.get('T', np.nan)
        T_str = f'{int(T_val)}' if pd.notna(T_val) else '--'
        tex.append(f'  {co2_label:<20s} & NR: + Bolton & ${c}$ & ${t}$ & {r2} & {T_str} \\\\')

    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(r'\begin{tablenotes}')
    tex.append(r'\small')
    tex.append(r'\item Panel A: PanelOLS with year-month FE, double-clustered SE (firm $\times$ year); $R^2$ = within-$R^2$; $N$ = firm-month observations. Panel B: Fama--MacBeth on neural residuals ($\hat{\varepsilon}^{NN} = R - \hat{R}^{NN}$) with Newey--West (4 lag) SE; $R^2$ = average cross-sectional $R^2$; $T$ = number of monthly cross-sections. The two panels employ different estimation methods because neural predictions are available only for 2018--2024 (rolling window test periods), while Panel A uses the full sample. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.')
    tex.append(r'\end{tablenotes}')
    tex.append(r'\end{table}')

    with open(os.path.join(TEX_DIR, 'tab_neural_resid.tex'), 'w') as f:
        f.write('\n'.join(tex))
    print('  \u2705 tab_neural_resid.tex')


# ==================================================================
#  TABLE 5: NP QUINTILE MONOTONICITY  (tab_np_quintile.tex)
# ==================================================================
def gen_tab_np_quintile():
    """NP quintile carbon premium — reads Q5-Q1 from CSV if available."""
    df = pd.read_csv(os.path.join(CSV_DIR, 'conditional_np_quintile.csv'))

    # Separate quintile rows from SoS row (if saved by script 6)
    sos_row = df[df['Quintile'] == 'Q5-Q1']
    data_rows = df[~df['Quintile'].isin(['Q5-Q1'])]

    q1_row = data_rows[data_rows['Quintile'] == 'Q1'].iloc[0]
    q5_row = data_rows[data_rows['Quintile'] == 'Q5'].iloc[0]
    q1_spread = float(q1_row['spread'])
    q5_spread = float(q5_row['spread'])
    q1_t      = float(q1_row['t'])
    q5_t      = float(q5_row['t'])
    T         = int(q1_row['T'])

    sos_spread = q5_spread - q1_spread

    if len(sos_row):
        # Use the exact t-stat from the analysis script (NW on paired diff)
        t_sos = float(sos_row.iloc[0]['t'])
        sos_method = 'Newey--West on paired difference series'
    else:
        # Fallback: Welch approximation
        se_q1  = q1_spread / q1_t if q1_t != 0 else 0
        se_q5  = q5_spread / q5_t if q5_t != 0 else 0
        se_sos = np.sqrt(se_q1**2 + se_q5**2)
        t_sos  = sos_spread / se_sos if se_sos != 0 else 0
        sos_method = 'Welch approximation (re-run script 6 for exact NW)'

    # ---- collect inline values ----
    VALUES['NPqOnespread']  = f'{q1_spread*100:+.2f}'
    VALUES['NPqOnet']       = f'{q1_t:.2f}'
    VALUES['NPqFivespread'] = f'{q5_spread*100:+.2f}'
    VALUES['NPqFivet']      = f'{q5_t:.2f}'
    VALUES['SoSspread']     = f'{sos_spread*100:+.2f}'
    VALUES['SoSt']          = f'{t_sos:.2f}'

    tex = []
    tex.append(r'\begin{table}[H]')
    tex.append(r'\centering')
    tex.append(r'\caption{Carbon Premium by Neural Prediction Quintile}')
    tex.append(r'\label{tab:np_quintile}')
    tex.append(r'\begin{tabular}{lrrr}')
    tex.append(r'\toprule')
    tex.append(r'Neural Prediction ($\hat{R}^{NN}$) Quintile & CO$_2$ H-L Spread (\%/mo) & $t$(NW) & $T$ \\')
    tex.append(r'\midrule')

    for _, row in data_rows.iterrows():
        q      = row['Quintile']
        spread = float(row['spread']) * 100
        t      = float(row['t'])
        Tval   = int(row['T'])
        label  = f'{q} (Low predicted return)' if q == 'Q1' else (f'{q} (High predicted return)' if q == 'Q5' else q)
        tex.append(f'  {label:<30s} & ${spread:+.2f}$ & ${t:+.2f}{stars(t)}$ & {Tval} \\\\')

    tex.append(r'\midrule')
    tex.append(f'  Q5 $-$ Q1             & ${sos_spread*100:+.2f}$ & ${t_sos:+.2f}{stars(t_sos)}$ & {T} \\\\')
    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(r'\begin{tablenotes}')
    tex.append(r'\small')
    tex.append(r'\item Independent $5 \times 3$ double sorts on the matched sample (2018--2024, $T = 77$ months). Each month, firms sorted into quintiles by LSTM neural predicted return $\hat{R}^{NN}$ (Q1 = lowest prediction, Q5 = highest) and into terciles by $\log(\text{Total CO}_2)$ within each quintile. CO$_2$ H-L = High $-$ Low CO$_2$ tercile equal-weighted monthly return (\%/month). FF5 $\alpha$: intercept from time-series regression of H-L spread on Fama--French five factors (MKT, SMB, HML, RMW, CMA). $t$-statistics: Newey--West (1987) HAC, 6 lags. Q5$-$Q1: Newey--West on paired difference series. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.')
    tex.append(r'\end{tablenotes}')
    tex.append(r'\end{table}')

    with open(os.path.join(TEX_DIR, 'tab_np_quintile.tex'), 'w') as f:
        f.write('\n'.join(tex))
    print('   tab_np_quintile.tex')


# ==================================================================
#  TABLE 6: CONDITIONAL SORTS  (tab_conditional.tex)
# ==================================================================
def gen_tab_conditional():
    """Conditional CO2 premium: dual-panel landscape (raw + neural residual)."""
    # Try dual CSV first; fall back to single
    dual_path = os.path.join(CSV_DIR, 'conditional_carbon_premium_dual.csv')
    single_path = os.path.join(CSV_DIR, 'conditional_carbon_premium.csv')

    if os.path.exists(dual_path):
        df = pd.read_csv(dual_path)
        has_nr = True
    else:
        df = pd.read_csv(single_path)
        has_nr = False

    # Use Raw panel for variable ordering
    df_raw = df[df['Dep'] == 'Raw'] if has_nr else df
    all_vars = list(dict.fromkeys(df_raw['Variable'].tolist()))

    # Collect SIZE and BM SoS values for manuscript macros
    for var_key in ['SIZE', 'BM']:
        sos_row = df_raw[(df_raw['Variable'] == var_key) & (df_raw['Group'] == 'H-L')]
        h_row   = df_raw[(df_raw['Variable'] == var_key) & (df_raw['Group'] == 'High')]
        if len(sos_row):
            VALUES[f'Cond{var_key}sos']  = f'{float(sos_row["spread"].values[0])*100:+.2f}'
            VALUES[f'Cond{var_key}tsos'] = f'{float(sos_row["t"].values[0]):.2f}'
        if len(h_row):
            VALUES[f'Cond{var_key}Ht']   = f'{float(h_row["t"].values[0]):.2f}'

    col_labels = {
        'SIZE': 'Size', 'BM': 'BM', 'LEVERAGE': 'Lev.', 'IO': 'IO',
        'ANALYSTS': 'Analysts', 'LOG_AMIHUD': 'Amihud', 'ENV_SCORE': 'ESG',
        'CARBON_INTENSITY': r'CO$_2$/Rev', 'DELTA_CO2': r'$\Delta$CO$_2$',
        'VOLAT': 'Volatility',
    }

    groups = ['Low', 'Med', 'High', 'H-L']
    n_vars = len(all_vars)

    tex = []
    tex.append(r'% AUTO-GENERATED by 13_generate_latex_tables.py')
    tex.append(r'\begin{landscape}')
    tex.append(r'\begin{table}[H]')
    tex.append(r'\centering')
    tex.append(r'\caption{Conditional Carbon Premium by Firm Characteristics}')
    tex.append(r'\label{tab:conditional}')
    tex.append(r'\small')

    col_spec = 'l' + 'r' * n_vars
    tex.append(r'\begin{tabular}{' + col_spec + '}')
    tex.append(r'\toprule')

    # Header
    header = 'Group'
    for var in all_vars:
        header += ' & ' + col_labels.get(var, var)
    header += r' \\'
    tex.append(header)
    tex.append(r'\midrule')

    def emit_panel(dep_filter, panel_label):
        src = df[df['Dep'] == dep_filter] if has_nr else df
        tex.append(r'\multicolumn{' + str(n_vars + 1) + r'}{l}{\textit{' + panel_label + r'}} \\')
        tex.append(r'\addlinespace[2pt]')
        for grp in groups:
            if grp == 'H-L':
                tex.append(r'\addlinespace')
                row_label = r'H $-$ L'
            else:
                row_label = grp
            cells = [row_label]
            for var in all_vars:
                sub = src[(src['Variable'] == var) & (src['Group'] == grp)]
                if len(sub) == 0:
                    cells.append('--')
                else:
                    spread = float(sub['spread'].values[0]) * 100
                    t_val  = float(sub['t'].values[0])
                    star   = stars(t_val)
                    cells.append(f'${spread:+.2f}{star}$')
            tex.append(' & '.join(cells) + r' \\')

    # Panel A: Raw Returns
    emit_panel('Raw', 'Panel A: Raw Returns')

    if has_nr:
        tex.append(r'\addlinespace[6pt]')
        tex.append(r'\midrule')
        emit_panel('NR', r'Panel B: Neural Residuals ($\hat{\varepsilon}^{NN}$)')

    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(r'\begin{tablenotes}')
    tex.append(r'\small')
    note = (r'\item Independent 3$\times$3 double sorts. Each month, firms sorted '
            r'into terciles by the conditioning variable (columns) and terciles by '
            r'$\log(\text{Total CO}_2)$. Cell values = CO$_2$ spread (High $-$ Low CO$_2$ tercile '
            r'within each conditioning group, \%/month). '
            r'H$-$L = spread-of-spreads (highest minus lowest conditioning group). ')
    if has_nr:
        note += (r'Both panels use the matched sample (2018--2024, $T=77$ months). '
                 r'Panel A: raw returns $R_{i,t}$. '
                 r'Panel B: neural residuals $\hat{\varepsilon}^{NN}_{i,t} = R_{i,t} - \hat{R}^{NN}_{i,t}$ '
                 r'(rolling-window test periods). ')
    note += r'$t$-statistics: Newey--West (1987) HAC, 6 lags. '
    note += r'$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.'
    tex.append(note)
    tex.append(r'\end{tablenotes}')
    tex.append(r'\end{table}')
    tex.append(r'\end{landscape}')

    with open(os.path.join(TEX_DIR, 'tab_conditional.tex'), 'w') as f:
        f.write('\n'.join(tex))
    print('   tab_conditional.tex')


# ==================================================================
#  TABLE 7: INDUSTRY HETEROGENEITY  (tab_industry.tex)
def gen_tab_industry():
    """Industry-level carbon premium."""
    df = pd.read_csv(os.path.join(CSV_DIR, 'industry_carbon_premium.csv'))
    df = df.sort_values('ff5_t', ascending=False)

    tex = []
    tex.append(r'\begin{table}[H]')
    tex.append(r'\centering')
    tex.append(r'\caption{Industry-Level Carbon Premium}')
    tex.append(r'\label{tab:industry}')
    tex.append(r'\small')
    tex.append(r'\begin{tabular}{lrrrr}')
    tex.append(r'\toprule')
    tex.append(r'Industry & $N_{\text{firms}}$ & Med CO$_2$ & Raw H-L ($t$) & FF5 $\alpha$ ($t$) \\')
    tex.append(r'\midrule')

    for _, row in df.iterrows():
        ind  = row['Industry'].replace('&', '\\&')
        if len(ind) > 38:
            ind = ind[:36] + '..'
        nf    = int(row['n_firms'])
        mco2  = float(row['med_co2'])
        raw   = float(row['raw_spread']) * 100
        raw_t = float(row['raw_t'])
        ff5a  = float(row['ff5_alpha']) * 100
        ff5t  = float(row['ff5_t'])
        tex.append(
            f'  {ind:<35s} & {nf:>3d} & {mco2:.1f} & '
            f'${raw:+.2f}{stars(raw_t)}$ (${raw_t:+.2f}$) & '
            f'${ff5a:+.2f}{stars(ff5t)}$ (${ff5t:+.2f}$) \\\\'
        )

    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(r'\begin{tablenotes}')
    tex.append(r'\small')
    tex.append(r'\item Monthly CO$_2$ High-minus-Low spread (equal-weighted) within each TRBC Business Sector (matched sample, 2018--2024). Each month, firms within an industry sorted into terciles by $\log(\text{Total CO}_2)$; the spread is the difference in mean returns between the top and bottom terciles. FF5 $\alpha$: intercept from time-series regression of the H-L spread on Fama--French five factors (MKT, SMB, HML, RMW, CMA). Business sectors with $\geq$20 firms per month and $\geq$30 months only. Returns in \%/month. $t$-statistics: Newey--West (1987) HAC, 6 lags. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.')
    tex.append(r'\end{tablenotes}')
    tex.append(r'\end{table}')

    # ---- collect top-3 industry values ----
    top3 = df.head(3)
    ordinal = {1: 'One', 2: 'Two', 3: 'Three'}
    for i, (_, row) in enumerate(top3.iterrows(), 1):
        o = ordinal[i]
        VALUES[f'IndTop{o}Name']  = row['Industry']
        VALUES[f'IndTop{o}Alpha'] = f'{float(row["ff5_alpha"])*100:+.2f}'
        VALUES[f'IndTop{o}T']     = f'{float(row["ff5_t"]):.2f}'

    with open(os.path.join(TEX_DIR, 'tab_industry.tex'), 'w') as f:
        f.write('\n'.join(tex))
    print('   tab_industry.tex')


# ==================================================================
#  TABLE 8: SPANNING TEST  (tab_spanning.tex)
# ==================================================================
def gen_tab_spanning():
    """Spanning regression of CO2 L/S portfolio."""
    df = pd.read_csv(os.path.join(CSV_DIR, 'spanning_test_results.csv'))

    tex = []
    tex.append(r'\begin{table}[H]')
    tex.append(r'\centering')
    tex.append(r'\caption{Spanning Regression: CO$_2$ Long-Short Portfolio}')
    tex.append(r'\label{tab:spanning}')
    tex.append(r'\begin{tabular}{lrrrr}')
    tex.append(r'\toprule')
    tex.append(r'Benchmark & $\alpha$ (\%/mo) & $t_\alpha$ & $R^2$ & $T$ \\')
    tex.append(r'\midrule')

    for _, row in df.iterrows():
        model     = row['Model']
        alpha_pct = float(row['alpha_pct'])
        t_a       = float(row['t_alpha'])
        r2        = float(row['r2'])
        T         = int(row['T'])
        tex.append(
            f'  {model:<20s} & ${alpha_pct:+.2f}$ & '
            f'${t_a:+.2f}{stars(t_a)}$ & {r2:.3f} & {T} \\\\'
        )

    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(r'\begin{tablenotes}')
    tex.append(r'\small')
    tex.append(r'\item Dependent variable: monthly return of the High$-$Low CO$_2$ tercile portfolio (equal-weighted, matched sample 2018--2024, $T = 77$ months). Each column adds standard risk factors: CAPM ($\alpha$ + MKT), FF3 (+ SMB, HML), FF5 (+ RMW, CMA). A significant $\alpha$ indicates the CO$_2$ spread is not spanned by the included factors; $\alpha \approx 0$ indicates full spanning. OLS with Newey--West (1987) HAC standard errors, 6 lags. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.')
    tex.append(r'\end{tablenotes}')
    tex.append(r'\end{table}')

    with open(os.path.join(TEX_DIR, 'tab_spanning.tex'), 'w') as f:
        f.write('\n'.join(tex))
    print('   tab_spanning.tex')

# ==================================================================
#  TEMPORAL SUBSAMPLE TABLE
# ==================================================================
def gen_tab_temporal():
    """Generate temporal subsample table from temporal_subsample.csv."""
    path = os.path.join(CSV_DIR, 'temporal_subsample.csv')
    if not os.path.exists(path):
        print('    temporal_subsample.csv not found — skipping')
        return

    df = pd.read_csv(path)

    tex = []
    tex.append(r'\begin{table}[H]')
    tex.append(r'\centering')
    tex.append(r'\caption{Temporal Subsample Analysis: Pre- vs.\ Post-COVID Carbon Premium}')
    tex.append(r'\label{tab:temporal}')
    tex.append(r'\begin{threeparttable}')
    tex.append(r'\begin{tabular}{llrrrr}')
    tex.append(r'\toprule')
    tex.append(r'Period & Specification & $\hat{\gamma}$ & $t$-stat & '
               r'$R^2_w$ & $N$ \\')
    tex.append(r'\midrule')

    for period in ['Full Sample', 'Pre-COVID', 'Post-COVID']:
        rows = df[df['Period'] == period]
        first = True
        for _, row in rows.iterrows():
            p_label = period if first else ''
            first = False
            spec = row['Model'].replace('M1: Bolton', 'M1: Bolton controls')
            spec = spec.replace('M7: Bolton+Neural', 'M7: Bolton + Neural Prediction ($\\hat{R}^{NN}$)')
            gamma = float(row['gamma'])
            t_val = float(row['t'])
            stars = _stars(t_val)
            r2 = float(row['within_r2'])
            n = int(row['n'])

            tex.append(f'{p_label} & {spec} & {gamma:+.3f} & '
                       f'{t_val:.2f}{stars} & {r2:.3f} & {n:,d} \\\\')
        tex.append(r'\midrule')

    tex.pop()  # remove last \midrule
    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(r'\begin{tablenotes}')
    tex.append(r'\small')
    tex.append(r'\item PanelOLS with time fixed effects and double-clustered '
               r'standard errors (firm $\times$ time). Split point: December 2020. '
               r'$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.')
    tex.append(r'\end{tablenotes}')
    tex.append(r'\end{threeparttable}')
    tex.append(r'\end{table}')

    out = os.path.join(TEX_DIR, 'tab_temporal.tex')
    with open(out, 'w') as f:
        f.write('\n'.join(tex) + '\n')
    print(f'   tab_temporal.tex')

    # ---- Collect inline values ----
    for _, row in df.iterrows():
        period_key = row['Period'].replace(' ', '').replace('-', '')
        model_key = 'Mone' if 'M1' in row['Model'] else 'Mseven'
        VALUES[f'Temp{period_key}{model_key}Coef'] = f'{float(row["gamma"]):.3f}'
        VALUES[f'Temp{period_key}{model_key}T'] = f'{float(row["t"]):.2f}'

    for period, key in [('Full Sample', 'Full'), ('Pre-COVID', 'Pre'), ('Post-COVID', 'Post')]:
        pr = df[df['Period'] == period]
        if len(pr):
            r0 = pr.iloc[0]
            VALUES[f'Temp{key}N']      = f'{int(r0["n"]):,}'.replace(',', '{,}')
            VALUES[f'Temp{key}Firms']  = f'{int(r0["firms"]):,}'.replace(',', '{,}')
            VALUES[f'Temp{key}Months'] = str(int(r0['months']))

    # Absorption percentages
    for period in ['Pre-COVID', 'Post-COVID']:
        m1 = df[(df['Period'] == period) & (df['Model'].str.contains('M1'))]
        m7 = df[(df['Period'] == period) & (df['Model'].str.contains('M7'))]
        if len(m1) and len(m7):
            g1 = abs(float(m1.iloc[0]['gamma']))
            g7 = abs(float(m7.iloc[0]['gamma']))
            if g1 > 1e-10:
                pct = (1 - g7 / g1) * 100
                pk = period.replace(' ', '').replace('-', '')
                VALUES[f'Temp{pk}AbsPct'] = f'{pct:.0f}'


def _stars(t):
    """Significance stars for LaTeX."""
    at = abs(t)
    if at >= 2.576:
        return '$^{***}$'
    elif at >= 1.960:
        return '$^{**}$'
    elif at >= 1.645:
        return '$^{*}$'
    return ''


# ==================================================================
#  TABLE 10: CAUSAL FOREST CATE  (tab_cate.tex)
# ==================================================================
def gen_tab_cate():
    """Causal Forest: CATE by firm characteristics (from 10_causal_forest_cate.py)."""
    cate_path = os.path.join(CSV_DIR, 'causal_forest_cate_by_characteristic.csv')
    summary_path = os.path.join(CSV_DIR, 'causal_forest_results.csv')
    if not os.path.exists(cate_path):
        print('    causal_forest_cate_by_characteristic.csv not found — skipping')
        return

    cate = pd.read_csv(cate_path)
    summary = pd.read_csv(summary_path) if os.path.exists(summary_path) else None

    # ---- Collect inline VALUES ----
    if summary is not None and len(summary):
        r = summary.iloc[0]
        VALUES['CATEateNeural'] = f'{float(r["ATE_neural"]):.3f}'
        ate_p = float(r['ATE_neural_pval'])
        VALUES['CATEateNeuralP'] = f'{ate_p:.3f}'
        VALUES['CATEateRaw'] = f'{float(r["ATE_raw"]):.3f}'
        VALUES['CATEnFirmYears'] = f'{int(r["N_firm_years"]):,}'
        VALUES['CATEnFirms'] = f'{int(r["N_tickers"]):,}'

        # Feature importances
        fi_cols = [c for c in r.index if c.startswith('FI_')]
        fi_sorted = sorted(fi_cols, key=lambda c: float(r[c]), reverse=True)
        word_nums = {1: 'One', 2: 'Two', 3: 'Three'}
        for i, col in enumerate(fi_sorted[:3]):
            var_name = col.replace('FI_', '')
            w = word_nums[i+1]
            VALUES[f'CATEfi{w}Name'] = var_name.replace('_', ' ')
            VALUES[f'CATEfi{w}Pct'] = f'{float(r[col])*100:.1f}'


    # ---- Pretty labels for modifiers ----
    modifier_labels = {
        'SIZE_L1': r'Size ($\log(\text{ME})$)',
        'BM_L1': r'Book-to-Market (BM)',
        'LEVERAGE': r'Leverage',
        'VOLAT': r'Volatility',
        'IO': r'Closely-held ownership',
    }

    modifiers = cate['Modifier'].unique()

    tex = []
    tex.append(r'% AUTO-GENERATED by 13_generate_latex_tables.py — DO NOT EDIT MANUALLY')
    tex.append(r'\begin{table}[H]')
    tex.append(r'\centering')
    tex.append(r'\caption{Causal Forest: Heterogeneous Treatment Effects of High CO$_2$}')
    tex.append(r'\label{tab:cate}')
    tex.append(r'\begin{threeparttable}')
    tex.append(r'\small')
    tex.append(r'\begin{tabular}{llcccc}')
    tex.append(r'\toprule')
    tex.append(r' & & \multicolumn{2}{c}{Neural Residual} & \multicolumn{2}{c}{Raw Returns} \\')
    tex.append(r'\cmidrule(lr){3-4} \cmidrule(lr){5-6}')
    tex.append(r'Effect Modifier & Tercile & CATE & $t$ & CATE & $t$ \\')
    tex.append(r'\midrule')

    for mod in modifiers:
        sub = cate[cate['Modifier'] == mod]
        label = modifier_labels.get(mod, mod.replace('_', r'\_'))
        tex.append(f'\\textit{{{label}}} & & & & & \\\\')
        for _, row in sub.iterrows():
            terc = row['Tercile']
            n_cate = float(row['CATE_neural'])
            n_t = float(row['t_neural'])
            r_cate = float(row['CATE_raw'])
            r_t = float(row['t_raw'])
            tex.append(
                f'  & {terc} & ${n_cate:+.3f}$ & ${n_t:.2f}{stars(n_t)}$ '
                f'& ${r_cate:+.3f}$ & ${r_t:.2f}{stars(r_t)}$ \\\\'
            )

        # H - L difference (point estimate only, no separate t-stat)
        h_row = sub[sub['Tercile'] == 'High']
        l_row = sub[sub['Tercile'] == 'Low']
        if len(h_row) and len(l_row):
            h_n = float(h_row['CATE_neural'].values[0])
            l_n = float(l_row['CATE_neural'].values[0])
            diff_n = h_n - l_n
            tex.append(
                f'  & H $-$ L & ${diff_n:+.3f}$ & '
                f'& & \\\\'
            )

            # Collect SIZE and BM CATE H-L for inline values
            if mod in ['SIZE_L1', 'BM_L1']:
                var_key = 'SIZE' if 'SIZE' in mod else 'BM'
                VALUES[f'CATEcond{var_key}diff'] = f'{diff_n:.3f}'
                # Use Low/High t-stats directly (already cluster-adjusted)
                VALUES[f'CATEcond{var_key}t'] = f'{float(h_row["t_neural"].values[0]):.2f}'

        tex.append(r'\addlinespace[2pt]')

    # Feature importance is reported in the text (macros \valCATEfi*Pct set above),
    # so it is intentionally omitted from the table to keep it on a single page.

    # ATE + sample size
    tex.append(r'\midrule')
    if summary is not None and len(summary):
        r = summary.iloc[0]
        ate_n = float(r['ATE_neural'])
        ate_n_p = float(r['ATE_neural_pval'])
        ate_r = float(r['ATE_raw'])
        tex.append(
            f'  ATE & & ${ate_n:+.3f}$ & $(p={ate_n_p:.3f})$ '
            f'& ${ate_r:+.3f}$ & $(p={float(r["ATE_raw_pval"]):.3f})$ \\\\'
        )
        # N (firm-years) and Firms moved to the caption to keep the table on one page.

    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(r'\begin{tablenotes}')
    tex.append(r'\small')
    tex.append(r'\item CausalForestDML (econml) with gradient-boosted nuisance models, '
               r'$B = 1{,}000$ trees, $cv = 5$. Treatment: $D^{\text{HIGH\_CO}_2}$ '
               r'(top vs.\ bottom tercile of lagged carbon intensity). '
               r'Outcome: annual mean neural residual ($\hat{\varepsilon}^{NN}$) '
               r'or raw return. Confounders (W): 11 Bolton controls. '
               r'All continuous variables z-scored. $t$-statistics: CLT standard errors '
               r'of individual forest-based CATE estimates, cluster-adjusted by $\sqrt{7}$ '
               r'for within-firm panel dependence. '
               r'$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.')
    tex.append(r'\end{tablenotes}')
    tex.append(r'\end{threeparttable}')
    tex.append(r'\end{table}')

    out = os.path.join(TEX_DIR, 'tab_cate.tex')
    with open(out, 'w') as f:
        f.write('\n'.join(tex) + '\n')
    print('   tab_cate.tex')


# ==================================================================
#  EXTRA VALUES from other CSVs
# ==================================================================
def collect_extra_values():
    """Read remaining CSVs for inline-cited numbers."""

    # Clark-West — use paired_t (proper t-statistic), NOT cw_stat (cumulative sum)
    cw_path = os.path.join(CSV_DIR, 'clark_west_results.csv')
    if os.path.exists(cw_path):
        cw = pd.read_csv(cw_path)
        if len(cw):
            r = cw.iloc[0]
            # paired_t is the correct significance measure for nested model comparison
            VALUES['CWstat']   = f'{float(r["paired_t"]):.2f}'
            VALUES['CWdeltaR'] = f'{float(r["delta_r2"])*100:.2f}'
            # mean_r2_a is the main 13-input model (no CO2) daily factor R²
            VALUES['DailyRsq'] = f'{float(r["mean_r2_a"])*100:.1f}'

    # Portfolio double sort (single sorts)
    ds_path = os.path.join(CSV_DIR, 'portfolio_double_sort.csv')
    if os.path.exists(ds_path):
        ds = pd.read_csv(ds_path)
        if len(ds):
            total_spread = ds['CO2_spread'].mean()
            VALUES['UnconditionalCOtwospread'] = f'{total_spread*100:+.2f}'
            # FF5 alphas for double-sort terciles
            for _, r in ds.iterrows():
                label = r['NP_Tercile'].replace(' ', '')
                VALUES[f'DS{label}Alpha'] = f'{r.get("ff5_alpha", 0)*100:+.3f}'
                VALUES[f'DS{label}AlphaT'] = f'{r.get("ff5_t", 0):.2f}'
                VALUES[f'DS{label}SpreadEW'] = f'{r["CO2_spread"]*100:+.2f}'
                if 'CO2_spread_VW' in r and pd.notna(r['CO2_spread_VW']):
                    VALUES[f'DS{label}SpreadVW']  = f'{r["CO2_spread_VW"]*100:+.2f}'
                    VALUES[f'DS{label}SpreadVWT'] = f'{r["t_VW"]:.2f}'
                if 'ff5_alpha_VW' in r and pd.notna(r['ff5_alpha_VW']):
                    VALUES[f'DS{label}AlphaVW']  = f'{r["ff5_alpha_VW"]*100:+.2f}'
                    VALUES[f'DS{label}AlphaVWT'] = f'{r["ff5_t_VW"]:.2f}'

    # Single sort results (CO2 H-L and NEURAL_PRED H-L)
    ss_path = os.path.join(CSV_DIR, 'single_sort_results.csv')
    if os.path.exists(ss_path):
        ss = pd.read_csv(ss_path)
        # Handle both old format (no Weighting column) and new (EW/VW)
        has_w = 'Weighting' in ss.columns
        for w_label, w_suffix in [('EW', ''), ('VW', 'VW')]:
            if has_w:
                sub = ss[ss['Weighting'] == w_label]
            else:
                sub = ss if w_label == 'EW' else pd.DataFrame()
            co2_row = sub[sub['Sort'] == 'CO2'] if len(sub) else pd.DataFrame()
            np_row  = sub[sub['Sort'] == 'NEURAL_PRED'] if len(sub) else pd.DataFrame()
            if len(co2_row):
                VALUES[f'SingleCOtwoSpread{w_suffix}'] = f'{float(co2_row.iloc[0]["HmL_mean"])*100:+.2f}'
                VALUES[f'SingleCOtwoT{w_suffix}']      = f'{float(co2_row.iloc[0]["HmL_t"]):.2f}'
            if len(np_row):
                VALUES[f'SingleNPspread{w_suffix}']    = f'{float(np_row.iloc[0]["HmL_mean"])*100:+.2f}'
                VALUES[f'SingleNPt{w_suffix}']         = f'{float(np_row.iloc[0]["HmL_t"]):.2f}'

    # Bootstrap SE ratios
    bs_path = os.path.join(CSV_DIR, 'bootstrap_vs_nw.csv')
    if os.path.exists(bs_path):
        bs = pd.read_csv(bs_path)
        if 'se_ratio' in bs.columns or 'SE_ratio' in bs.columns:
            col = 'se_ratio' if 'se_ratio' in bs.columns else 'SE_ratio'
            VALUES['BootstrapSEratioMin'] = f'{bs[col].min():.2f}'
            VALUES['BootstrapSEratioMax'] = f'{bs[col].max():.2f}'

    hb_path = os.path.join(CSV_DIR, 'hausman_bootstrap.csv')
    if os.path.exists(hb_path):
        hb = pd.read_csv(hb_path)
        if 'delta_gamma' in hb.columns and len(hb):
            dg = hb['delta_gamma'].dropna().values
            m, se = float(np.mean(dg)), float(np.std(dg))
            VALUES['BootDeltaMean'] = f'{m:+.3f}'
            VALUES['BootDeltaSE']   = f'{se:.3f}'
            VALUES['BootDeltaCIlo'] = f'{np.percentile(dg, 2.5):+.3f}'
            VALUES['BootDeltaCIhi'] = f'{np.percentile(dg, 97.5):+.3f}'
            VALUES['BootDeltaT']    = f'{m/se:.2f}'
            VALUES['BootDeltaP']    = f'{float(np.mean(dg <= 0)):.3f}'

    rg_path = os.path.join(CSV_DIR, 'rolling_co2_gamma.csv')
    if os.path.exists(rg_path):
        rg = pd.read_csv(rg_path)
        if len(rg):
            pk = rg.loc[rg['rolling_gamma'].idxmax()]
            tr = rg.loc[rg['rolling_gamma'].idxmin()]
            VALUES['RollPeakGamma'] = f'{float(pk["rolling_gamma"]):+.2f}'
            VALUES['RollPeakT']     = f'{float(pk["rolling_t"]):.2f}'
            VALUES['RollTroughGamma'] = f'{float(tr["rolling_gamma"]):+.2f}'
            VALUES['RollTroughT']   = f'{float(tr["rolling_t"]):.2f}'

    re_path = os.path.join(CSV_DIR, 'reported_vs_estimated.csv')
    if os.path.exists(re_path):
        rv = pd.read_csv(re_path).set_index('measure')
        if 'Estimated (LOG_EST_CO2)' in rv.index:
            VALUES['AswaniEstGamma'] = f'{float(rv.loc["Estimated (LOG_EST_CO2)","gamma"]):+.3f}'
            VALUES['AswaniEstT']     = f'{float(rv.loc["Estimated (LOG_EST_CO2)","t"]):.2f}'
        if 'reported_share_pct' in rv.index:
            VALUES['AswaniRepShare'] = f'{float(rv.loc["reported_share_pct","gamma"]):.0f}'


# ==================================================================
#  GENERATE manuscript_values.tex
# ==================================================================
def gen_values_file():
    r"""Write \newcommand macros for every inline number."""
    lines = [
        r'%% AUTO-GENERATED by 13_generate_latex_tables.py — DO NOT EDIT',
        r'%% Re-run the script to update all inline values.',
        r'%%',
    ]
    for key in sorted(VALUES.keys()):
        val = VALUES[key]
        # Sanitise: replace % and & for LaTeX safety
        val_safe = val.replace('%', r'\%').replace('&', r'\&')
        lines.append(f'\\newcommand{{\\val{key}}}{{{val_safe}}}')

    with open(VALUES_PATH, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'   manuscript_values.tex  ({len(VALUES)} values)')


# ==================================================================
#  TABLE: FOLD-BY-FOLD R² (tab_fold_r2.tex)
# ==================================================================
def gen_tab_fold_r2():
    """Generate fold-by-fold R² table from TRUBA results."""
    results_dir = os.path.join(PAPER_DIR, 'results', 'paper2_rolling_k5')
    if not os.path.isdir(results_dir):
        print('   No TRUBA results directory found, skipping fold R²')
        return

    folds = sorted([d for d in os.listdir(results_dir) if d.startswith('fold_test')])
    rows = []
    for fold_dir in folds:
        metrics_file = os.path.join(results_dir, fold_dir, 'final_metrics.json')
        if not os.path.exists(metrics_file):
            continue
        with open(metrics_file) as f:
            m = json.load(f)
        rows.append({
            'test_year': m.get('test_start', fold_dir.replace('fold_test', '')),
            'train_end': m.get('train_end', ''),
            'val_end': m.get('val_end', ''),
            'best_epoch': m.get('best_epoch', ''),
            'val_r2': m.get('best_val_r2', np.nan),
            'test_r2': m.get('final_test_r2', np.nan),
        })

    if not rows:
        print('   No fold metrics found')
        return

    # Compute averages
    val_r2s = [r['val_r2'] for r in rows if not np.isnan(r['val_r2'])]
    test_r2s = [r['test_r2'] for r in rows if not np.isnan(r['test_r2'])]
    avg_val = np.mean(val_r2s) if val_r2s else np.nan
    avg_test = np.mean(test_r2s) if test_r2s else np.nan

    # Store values
    VALUES['NfoldsFoldR'] = str(len(rows))
    VALUES['AvgTestR'] = f'{avg_test*100:.1f}'
    VALUES['AvgValR'] = f'{avg_val*100:.1f}'

    tex = []
    tex.append(r'\begin{table}[H]')
    tex.append(r'\centering')
    tex.append(r'\caption{LSTM+CA Model Performance: Fold-by-Fold $R^2$}')
    tex.append(r'\label{tab:fold_r2}')
    tex.append(r'\small')
    tex.append(r'\begin{tabular}{ccccc}')
    tex.append(r'\toprule')
    tex.append(r'Test Year & Training End & Best Epoch & Validation $R^2$ & Test $R^2$ \\')
    tex.append(r'\midrule')

    for r in rows:
        tex.append(f"  {r['test_year']} & {r['train_end']} & {r['best_epoch']} & "
                   f"{r['val_r2']*100:.1f}\\% & {r['test_r2']*100:.1f}\\% \\\\")

    tex.append(r'\midrule')
    tex.append(f"  Average & & & {avg_val*100:.1f}\\% & {avg_test*100:.1f}\\% \\\\")
    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(r'\begin{tablenotes}')
    tex.append(r'\small')
    tex.append(r'\item LSTM with Cross-Attention (CA) architecture, $K = 5$ latent factors. 13 firm-level inputs: Size, Book-to-Market, Momentum, Volatility, Leverage, ROE, Investment-to-Assets, Sales-to-Price, log(PPE), HHI, closely-held ownership, CAPM Beta, and Market Excess Return. Rolling-window design: each fold expands the training set by one year, validates on the following year, and tests on the year after that. $R^2_{\text{val}}$: daily total (pooled) $R^2$ on validation data. $R^2_{\text{test}}$: daily total (pooled) $R^2$ on held-out test data. Positive $R^2$ indicates the model outperforms the historical mean forecast.')
    tex.append(r'\end{tablenotes}')
    tex.append(r'\end{table}')

    with open(os.path.join(TEX_DIR, 'tab_fold_r2.tex'), 'w') as f:
        f.write('\n'.join(tex))
    print('   tab_fold_r2.tex')


# ==================================================================
#  TABLE: PLACEBO TEST (tab_placebo.tex)
# ==================================================================
def gen_tab_placebo():
    """Placebo test: does NEURAL_PRED absorb known risk premiums?"""
    path = os.path.join(CSV_DIR, 'placebo_test.csv')
    if not os.path.exists(path):
        print('   placebo_test.csv not found')
        return

    df = pd.read_csv(path)

    tex = []
    tex.append(r'\begin{table}[H]')
    tex.append(r'\centering')
    tex.append(r'\caption{Placebo Test: Does NEURAL\_PRED Absorb Known Risk Premiums?}')
    tex.append(r'\label{tab:placebo}')
    tex.append(r'\small')
    tex.append(r'\begin{tabular}{lcccccc}')
    tex.append(r'\toprule')
    tex.append(r' & \multicolumn{2}{c}{Without Neural} & \multicolumn{2}{c}{With Neural} & & LSTM \\')
    tex.append(r'\cmidrule(lr){2-3} \cmidrule(lr){4-5}')
    tex.append(r'Premium & $\gamma$ & $t$ & $\gamma$ & $t$ & Absorption & Input? \\')
    tex.append(r'\midrule')

    lstm_inputs = {'SIZE': 'Yes', 'BM': 'Yes', 'MOM': 'Yes', 'LOG_CO2_TOTAL': 'No'}

    for _, row in df.iterrows():
        var = row['Variable']
        label = str(row['Label']).split('(')[0].strip().replace('₂', '$_2$').replace('&', r'\&')
        c_w = row['coef_without']
        t_w = row['t_without']
        c_n = row['coef_with']
        t_n = row['t_with']
        absp = row['absorption_pct']
        is_input = lstm_inputs.get(var, '?')

        tex.append(f"  {label} & {c_w:+.3f} & ${t_w:.2f}{stars(t_w)}$ & "
                   f"{c_n:+.3f} & ${t_n:.2f}{stars(t_n)}$ & {absp:.0f}\\% & {is_input} \\\\")

    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(r'\begin{tablenotes}')
    tex.append(r'\small')
    tex.append(r'\item PanelOLS with year-month FE and double-clustered SE (firm $\times$ year). Matched sample: 2018--2024, $N = 58{,}514$. ``Without Neural'': $\gamma$ coefficient for each row-variable with Bolton controls only. ``With Neural'': adds NEURAL\_PRED ($\hat{R}^{NN}$) as an additional control. Absorption (\%) = $100 \times (1 - \gamma_{\text{with}} / \gamma_{\text{without}})$. Size, BM, and Momentum are among the 13 LSTM inputs, so their absorption is mechanically expected. CO$_2$ is \emph{not} an LSTM input; its absorption indicates that the neural model captures carbon-relevant pricing information through nonlinear characteristic interactions. $^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.')
    tex.append(r'\end{tablenotes}')
    tex.append(r'\end{table}')

    with open(os.path.join(TEX_DIR, 'tab_placebo.tex'), 'w') as f:
        f.write('\n'.join(tex))
    print('   tab_placebo.tex')


# ==================================================================
#  TABLE: BOOTSTRAP HAUSMAN (tab_bootstrap.tex)
# ==================================================================
def gen_tab_bootstrap():
    """Bootstrap Hausman test results."""
    path = os.path.join(CSV_DIR, 'hausman_bootstrap.csv')
    if not os.path.exists(path):
        print('   hausman_bootstrap.csv not found')
        return

    df = pd.read_csv(path, header=None, names=['delta_gamma'])
    if len(df) == 0:
        print('   hausman_bootstrap.csv is empty')
        return

    deltas = pd.to_numeric(df['delta_gamma'], errors='coerce').dropna().values
    n_boot = len(deltas)
    mean_dg = np.mean(deltas)
    se_dg = np.std(deltas, ddof=1)
    t_stat = mean_dg / se_dg if se_dg > 0 else 0
    ci_lo = np.percentile(deltas, 2.5)
    ci_hi = np.percentile(deltas, 97.5)
    prob_pos = np.mean(deltas > 0)

    tex = []
    tex.append(r'\begin{table}[H]')
    tex.append(r'\centering')
    tex.append(r'\caption{Bootstrap Hausman Test: Is the Absorption Statistically Significant?}')
    tex.append(r'\label{tab:bootstrap}')
    tex.append(r'\small')
    tex.append(r'\begin{tabular}{lc}')
    tex.append(r'\toprule')
    tex.append(r'Statistic & Value \\')
    tex.append(r'\midrule')
    tex.append(f"  $\\Delta\\gamma = \\gamma_{{M1}} - \\gamma_{{M7}}$ & {mean_dg:+.3f} \\\\")
    tex.append(f"  Bootstrap SE & {se_dg:.3f} \\\\")
    tex.append(f"  $t$-statistic & {t_stat:.2f}{stars(t_stat)} \\\\")
    tex.append(f"  95\\% CI & [{ci_lo:+.3f}, {ci_hi:+.3f}] \\\\")
    tex.append(f"  $P(\\Delta\\gamma > 0)$ & {prob_pos*100:.1f}\\% \\\\")
    tex.append(f"  Bootstrap iterations & {n_boot:,} \\\\")
    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(r'\begin{tablenotes}')
    tex.append(r'\small')
    tex.append(r'\item Block bootstrap test of whether the CO$_2$ coefficient reduction from M1 to M7 is statistically significant. Resampling entire calendar months with replacement preserves cross-sectional dependence. $\Delta\gamma = \gamma_{\text{M1}} - \gamma_{\text{M7}}$: difference in CO$_2$ coefficients between M1 (Bolton controls only) and M7 (Bolton + NEURAL\_PRED). Matched sample: 2018--2024, $N = 58{,}514$ firm-months. $p$-value: fraction of bootstrap replications with $\Delta\gamma \leq 0$.')
    tex.append(r'\end{tablenotes}')
    tex.append(r'\end{table}')

    with open(os.path.join(TEX_DIR, 'tab_bootstrap.tex'), 'w') as f:
        f.write('\n'.join(tex))
    print('   tab_bootstrap.tex')

    # Store values
    VALUES['BootstrapDeltaGamma'] = f'{mean_dg:+.3f}'
    VALUES['BootstrapT'] = f'{t_stat:.2f}'
    VALUES['BootstrapProbPos'] = f'{prob_pos*100:.1f}'
    VALUES['BootstrapN'] = str(n_boot)


# ==================================================================
#  TABLE: CORRELATION MATRIX  (tab_corr.tex)
# ==================================================================
def gen_tab_corr():
    """Generate correlation matrix from panel data."""
    panel_path = os.path.join(DATA_DIR, 'final_monthly_panel_clean.csv')
    if not os.path.exists(panel_path):
        print('   tab_corr: panel not found, skipping')
        return

    df = pd.read_csv(panel_path)

    # Variable mapping: internal -> display
    var_map = {
        'LOG_CO2_TOTAL': r'CO$_2$',
        'SIZE': 'SIZE',
        'BM': 'BM',
        'LEVERAGE': 'LEV',
        'VOLAT': 'VOLAT',
        'IO': 'CH',
        'LOG_AMIHUD': 'Illiq',
    }
    cols = list(var_map.keys())
    cols_avail = [c for c in cols if c in df.columns]

    sub = df[cols_avail].dropna()
    N = len(sub)
    corr = sub.corr()

    # Build lower-triangle LaTeX
    labels = [var_map[c] for c in cols_avail]
    ncols = len(labels)
    col_spec = 'l' + 'r' * ncols

    tex = []
    tex.append(r'% AUTO-GENERATED — DO NOT EDIT MANUALLY')
    tex.append(r'\begin{table}[H]')
    tex.append(r'\centering')
    tex.append(r'\caption{Cross-Sectional Correlation Matrix}')
    tex.append(r'\label{tab:corr}')
    tex.append(r'\begin{threeparttable}')
    tex.append(r'\begin{tabular}{' + col_spec + '}')
    tex.append(r'\toprule')
    tex.append(' & ' + ' & '.join(labels) + r' \\')
    tex.append(r'\midrule')

    for i, row_var in enumerate(cols_avail):
        row_label = var_map[row_var]
        cells = []
        for j, col_var in enumerate(cols_avail):
            if j > i:
                cells.append('')
            else:
                val = corr.loc[row_var, col_var]
                if val < 0:
                    cells.append(f'$-${abs(val):.3f}')
                else:
                    cells.append(f'{val:.3f}')
        tex.append(f'{row_label:8s} & ' + ' & '.join(cells) + r' \\')

    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(r'\begin{tablenotes}[flushleft]')
    tex.append(r'\small')
    tex.append(
        r'\item CO$_2$ = $\log(\text{Total CO}_2)$; SIZE = $\log(\text{Market Cap})$; '
        r'BM = Book-to-Market; LEV = Leverage; VOLAT = Return Volatility; '
        r'IO = Institutional Ownership; ILLIQ = $\log(\text{Amihud Illiquidity})$. '
        f'Pooled Pearson correlations, $N = {N:,d}$ firm-months with non-missing CO$_2$ and Amihud. '
        r'The high $|\rho|$ between SIZE and ILLIQ '
        f'(${corr.loc["SIZE", "LOG_AMIHUD"]:.3f}$) '
        r'reflects that illiquidity is largely a size phenomenon.'
    )
    tex.append(r'\end{tablenotes}')
    tex.append(r'\end{threeparttable}')
    tex.append(r'\end{table}')

    with open(os.path.join(TEX_DIR, 'tab_corr.tex'), 'w') as f:
        f.write('\n'.join(tex) + '\n')
    print('   tab_corr.tex')


# ==================================================================
#  TABLE: NEURAL RESIDUAL FMB  (tab_neural_compare.tex)
# ==================================================================
def gen_tab_neural_compare():
    """Neural Residual FMB results from table2b CSV."""
    csv_path = os.path.join(CSV_DIR, 'table2b_neural_cross_sectional.csv')
    if not os.path.exists(csv_path):
        print('   tab_neural_compare: CSV not found, skipping')
        return

    df = pd.read_csv(csv_path)
    nr = df[df['Test'] == 'NeuralResid_FMB'].copy()
    if nr.empty:
        print('   tab_neural_compare: no NeuralResid_FMB rows')
        return

    def stars(t):
        t = abs(t)
        if t >= 2.576: return '^{***}'
        if t >= 1.960: return '^{**}'
        if t >= 1.645: return '^{*}'
        return ''

    tex = []
    tex.append(r'% AUTO-GENERATED by 13_generate_latex_tables.py')
    tex.append(r'\begin{table}[H]')
    tex.append(r'\centering')
    tex.append(r'\caption{Carbon Premium: Raw Returns vs.\ Neural Residuals}')
    tex.append(r'\label{tab:neural_co2}')
    tex.append(r'\begin{threeparttable}')
    tex.append(r'\begin{tabular}{lccccc}')
    tex.append(r'\toprule')
    tex.append(r'Model & $\gamma_{CO_2}$ & $t$ & $R^2$ & $N$ & $T$ \\')
    tex.append(r'\midrule')

    for _, row in nr.iterrows():
        model = row['Model']
        coef = row['coef']
        t_val = row['t']
        r2 = row['avg_r2']
        T = int(row['T']) if pd.notna(row['T']) else ''
        s = stars(t_val)
        tex.append(
            f'{model} & ${coef:.3f}$ & ${t_val:.2f}{s}$ & '
            f'${r2:.3f}$ &  & {T} \\\\'
        )

    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(r'\begin{tablenotes}')
    tex.append(r'\small')
    tex.append(
        r'\item Monthly Fama--MacBeth regressions (2018--2024, $T = 77$ cross-sections) with '
        r'Newey--West (4 lag) HAC standard errors. '
        r"``Raw Returns'' uses $R_{it}$ as dependent variable; "
        r"``Neural Residual'' uses $\hat{\varepsilon}^{NN}_{it} = R_{it} - \hat{R}^{NN}_{it}$. "
        r'Bolton controls included in all specifications: SIZE, BM, LEV, ROE, INVEST, SALESGR, '
        r'log(PPE), MOM, VOLAT, HHI, IO. $R^2$ = average cross-sectional $R^2$. '
        r'$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.'
    )
    tex.append(r'\end{tablenotes}')
    tex.append(r'\end{threeparttable}')
    tex.append(r'\end{table}')

    with open(os.path.join(TEX_DIR, 'tab_neural_compare.tex'), 'w') as f:
        f.write('\n'.join(tex) + '\n')
    print('   tab_neural_compare.tex')


# ==================================================================
#  TABLE: INDUSTRY FE ROBUSTNESS  (tab_industry_fe.tex)
# ==================================================================
def gen_tab_industry_fe():
    """Industry FE robustness from industry_fe_bolton.csv."""
    csv_path = os.path.join(CSV_DIR, 'industry_fe_bolton.csv')
    if not os.path.exists(csv_path):
        print('   tab_industry_fe: CSV not found, skipping')
        return

    df = pd.read_csv(csv_path)

    # Readable variable names
    # Row labels are already LaTeX-ready in the CSV (Total CO$_2$ / Scope 1 / CO$_2$ Intensity).
    var_labels = {}

    def stars(t):
        t = abs(t)
        if t >= 2.576: return '^{***}'
        if t >= 1.960: return '^{**}'
        if t >= 1.645: return '^{*}'
        return ''

    tex = []
    tex.append(r'% AUTO-GENERATED by 13_generate_latex_tables.py')
    tex.append(r'\begin{table}[H]')
    tex.append(r'\centering')
    tex.append(r'\caption{Industry Fixed Effects Robustness}')
    tex.append(r'\label{tab:indfe}')
    tex.append(r'\begin{threeparttable}')
    tex.append(r'\begin{tabular}{lccc}')
    tex.append(r'\toprule')
    tex.append(r'Variable & No Ind.~FE ($t$) & With Ind.~FE ($t$) & $\Delta t$ \\')
    tex.append(r'\midrule')

    for _, row in df.iterrows():
        vname = var_labels.get(row['Variable'], row['Variable'])
        t_no = row['NoIndFE_t']
        t_ind = row['IndFE_t']
        dt = row['Delta_T']
        s_no = stars(t_no)
        s_ind = stars(t_ind)
        tex.append(
            f'{vname} & ${t_no:.2f}{s_no}$ & ${t_ind:.2f}{s_ind}$ & ${dt:.2f}$ \\\\'
        )

    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(r'\begin{tablenotes}')
    tex.append(r'\small')
    tex.append(
        r'\item Monthly Fama--MacBeth regressions on neural residuals '
        r'($\hat{\varepsilon}^{NN} = R - \hat{R}^{NN}$, matched sample 2018--2024, $T = 77$ months). '
        r'Following \citet{bolton2023global}, ``With Ind.\ FE'' demeans every model variable '
        r'within each industry$\times$month cell before the Fama--MacBeth procedure (52 industries). '
        r'Reported values are the $t$-statistic of the CO$_2$ measure with the full Bolton controls; '
        r'$\Delta t = t_{\text{IndFE}} - t_{\text{NoIndFE}}$. '
        r'$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.'
    )
    tex.append(r'\end{tablenotes}')
    tex.append(r'\end{threeparttable}')
    tex.append(r'\end{table}')

    with open(os.path.join(TEX_DIR, 'tab_industry_fe.tex'), 'w') as f:
        f.write('\n'.join(tex) + '\n')
    print('   tab_industry_fe.tex')


# ==================================================================
#  BH-FDR CORRECTION ACROSS ALL HYPOTHESIS TESTS
# ==================================================================
def gen_fdr_correction():
    """
    Collect p-values from all major hypothesis tests, apply
    Benjamini-Hochberg FDR correction at q=0.05, and report
    how many survive.
    """
    from scipy import stats as sp_stats

    tests = []

    # 1. Absorption models (M1-M7)
    abs_path = os.path.join(CSV_DIR, 'absorption_confidence_intervals.csv')
    if os.path.exists(abs_path):
        ab = pd.read_csv(abs_path)
        for _, r in ab.iterrows():
            tests.append((f"Absorption: {r['Model']}", r['pval']))

    # 2. Industry spreads
    ind_path = os.path.join(CSV_DIR, 'industry_carbon_premium.csv')
    if os.path.exists(ind_path):
        ind = pd.read_csv(ind_path)
        for _, r in ind.iterrows():
            t_val = r['raw_t']
            T_months = r['T']
            p_val = 2 * (1 - sp_stats.t.cdf(abs(t_val), df=T_months - 1))
            tests.append((f"Industry: {r['Industry'][:25]}", p_val))

    # 3. Conditional spreads
    cond_path = os.path.join(CSV_DIR, 'conditional_carbon_premium.csv')
    if os.path.exists(cond_path):
        cond = pd.read_csv(cond_path)
        for _, r in cond.iterrows():
            t_val = r['t']
            T_months = r['T']
            p_val = 2 * (1 - sp_stats.t.cdf(abs(t_val), df=T_months - 1))
            tests.append((f"Conditional: {r['Variable']}-{r['Group']}", p_val))

    # 4. Spanning test alphas
    span_path = os.path.join(CSV_DIR, 'spanning_test_results.csv')
    if os.path.exists(span_path):
        span = pd.read_csv(span_path)
        for _, r in span.iterrows():
            t_val = r['t_alpha']
            T_months = r['T']
            p_val = 2 * (1 - sp_stats.t.cdf(abs(t_val), df=T_months - 1))
            tests.append((f"Spanning: {r['Model']}", p_val))

    if not tests:
        print('   No hypothesis tests found for FDR correction')
        return

    # Sort by p-value (BH procedure)
    tests.sort(key=lambda x: x[1])
    m = len(tests)
    q = 0.05

    # BH critical values
    bh_results = []
    max_reject_k = -1
    for k, (name, pval) in enumerate(tests, 1):
        bh_threshold = (k / m) * q
        reject = pval <= bh_threshold
        if reject:
            max_reject_k = k
        bh_results.append((name, pval, bh_threshold, reject))

    # All tests with rank <= max_reject_k are rejected
    n_rejected = 0
    for k, (name, pval, bh_thr, _) in enumerate(bh_results):
        final_reject = (k + 1 <= max_reject_k) if max_reject_k >= 0 else False
        bh_results[k] = (name, pval, bh_thr, final_reject)
        if final_reject:
            n_rejected += 1

    n_total = len(bh_results)
    n_survive = n_rejected
    n_orig_sig = sum(1 for _, p, _, _ in bh_results if p < 0.05)

    print(f'   BH-FDR correction: {n_orig_sig} tests at p<0.05, '
          f'{n_survive} survive FDR q=0.05 (of {n_total} total)')

    # Store values
    VALUES['FDRtotalTests'] = str(n_total)
    VALUES['FDRorigSig'] = str(n_orig_sig)
    VALUES['FDRsurvive'] = str(n_survive)

    # Print details for significant ones
    for name, pval, bh_thr, reject in bh_results:
        if pval < 0.05 or reject:
            status = ' survives FDR' if reject else ' does not survive'
            print(f'    {name:45s}  p={pval:.3f}  BH_thr={bh_thr:.3f}  {status}')


# ==================================================================
#  MAIN
# ==================================================================
def main():
    print('=' * 60)
    print('  AUTO-GENERATING LATEX TABLES + VALUES FROM CSV RESULTS')
    print('=' * 60)
    print(f'  Source CSVs : {CSV_DIR}')
    print(f'  Source data : {DATA_DIR}')
    print(f'  Output      : {TEX_DIR}')
    print()

    gen_tab_desc()
    gen_tab_bolton()
    gen_tab_absorption()
    gen_tab_neural_resid()
    gen_tab_np_quintile()
    gen_tab_conditional()
    gen_tab_industry()
    gen_tab_spanning()
    gen_tab_temporal()
    gen_tab_cate()
    gen_tab_fold_r2()
    gen_tab_placebo()
    gen_tab_bootstrap()
    gen_tab_corr()
    gen_tab_neural_compare()
    gen_tab_industry_fe()
    gen_fdr_correction()
    collect_extra_values()
    gen_values_file()

    tex_count = len([f for f in os.listdir(TEX_DIR) if f.endswith('.tex')])

    print()
    print(' All tables + values generated successfully!')
    print(f'   Tables  : {TEX_DIR}  ({tex_count} .tex files)')
    print(f'   Values  : {VALUES_PATH}')


if __name__ == '__main__':
    main()
