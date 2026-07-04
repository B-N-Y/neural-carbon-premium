"""
28_manuscript_figures.py
========================
Generate ALL manuscript figures from CSV results (K=5).
Output directory: 2nd_paper/manuscript/figures/  (PDF + PNG)

Figures (taxonomy-free):
  fig1_waterfall            – Carbon Premium Absorption (M1→M7)
  fig2_conditional_premium  – Conditional Carbon Premium by Firm Characteristic
  fig3_industry             – Industry Heterogeneity of Carbon Premium
  fig_mechanism_map         – Methodological Framework Diagram
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── paths ──────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TBL    = os.path.join(BASE, 'manuscript', 'tables')
FIG    = os.path.join(BASE, 'manuscript', 'figures')
DATA   = os.path.join(BASE, 'results')
os.makedirs(FIG, exist_ok=True)

def save(fig, name):
    """Save as PDF + PNG to manuscript/figures/"""
    fig.savefig(os.path.join(FIG, f'{name}.pdf'), dpi=300,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    fig.savefig(os.path.join(FIG, f'{name}.png'), dpi=200,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  ✅ {name}.pdf / .png')


# ======================================================================
# FIG 1 — Carbon Premium Absorption Waterfall
# ======================================================================
def fig1_waterfall():
    """
    Waterfall showing CO₂ coefficient across nested models:
    M1 (Bolton) → M3 (+ ICA LF) → M4 (+ FF5) → M6 (Neural only) → M7 (Bolton + Neural)
    Key insight: Linear controls (M3, M4) barely reduce the premium,
    but neural prediction (M6) reverses the sign entirely.
    """
    abs_path = os.path.join(TBL, 'absorption_confidence_intervals.csv')
    if not os.path.exists(abs_path):
        print('  ⚠️ absorption_confidence_intervals.csv not found — skipping')
        return

    ab = pd.read_csv(abs_path)

    # Select key models: M1, M3, M4, M6, M7
    key_models = {
        'M1': 'M1: Bolton\nOnly',
        'M3': 'M3: Bolton +\nICA LF',
        'M4': 'M4: Bolton +\nFF5 Betas',
        'M6': 'M6: Neural\nPred Only',
        'M7': 'M7: Bolton +\nNeural Pred'
    }

    rows = []
    for spec, label in key_models.items():
        match = ab[ab['Model'].str.startswith(spec, na=False)]
        if len(match) > 0:
            rows.append({
                'label': label,
                'coef': match.iloc[0]['coef'],
                'ci_lo': match.iloc[0]['ci_lo'],
                'ci_hi': match.iloc[0]['ci_hi'],
                't': match.iloc[0]['t_stat']
            })

    if len(rows) < 4:
        print('  ⚠️ Not enough absorption models — skipping waterfall')
        return

    fig, ax = plt.subplots(figsize=(13, 6))

    x = np.arange(len(rows))
    colors = []
    for r in rows:
        if abs(r['t']) >= 2.576:
            colors.append('#27ae60' if r['coef'] > 0 else '#c0392b')  # green/red for significant
        elif abs(r['t']) >= 1.96:
            colors.append('#2ecc71' if r['coef'] > 0 else '#e74c3c')
        elif abs(r['t']) >= 1.645:
            colors.append('#82e0aa' if r['coef'] > 0 else '#f1948a')  # lighter for marginal
        else:
            colors.append('#bdc3c7')  # gray for insignificant

    bars = ax.bar(x, [r['coef'] for r in rows], width=0.5,
                  color=colors, edgecolor='white', linewidth=1.5)

    # Error bars
    for i, r in enumerate(rows):
        ax.plot([i, i], [r['ci_lo'], r['ci_hi']], color='black', lw=1.5)
        ax.plot([i-0.08, i+0.08], [r['ci_lo'], r['ci_lo']], color='black', lw=1.5)
        ax.plot([i-0.08, i+0.08], [r['ci_hi'], r['ci_hi']], color='black', lw=1.5)

    # t-stat labels
    for i, r in enumerate(rows):
        sig = '***' if abs(r['t']) >= 2.576 else '**' if abs(r['t']) >= 1.96 else '*' if abs(r['t']) >= 1.645 else 'n.s.'
        y_pos = r['ci_hi'] + 0.005 if r['coef'] >= 0 else r['ci_lo'] - 0.005
        va = 'bottom' if r['coef'] >= 0 else 'top'
        ax.text(i, y_pos, f"t={r['t']:.2f}{sig}",
                ha='center', va=va, fontsize=10, fontweight='bold')

    # Arrow from M4 to M6 showing sign reversal (indices 2 → 3 in 5-bar layout)
    if len(rows) >= 5:
        from matplotlib.patches import FancyArrowPatch
        arrow = FancyArrowPatch(
            (2.35, -0.03),
            (2.65, -0.08),
            arrowstyle='->', mutation_scale=20,
            color='#e74c3c', lw=3, connectionstyle='arc3,rad=0.25'
        )
        ax.add_patch(arrow)
        ax.text(2.5, -0.02, 'SIGN\nREVERSAL',
                fontsize=11, color='#e74c3c', fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#e74c3c', alpha=0.9))

    # Bracket over M1-M4 labeling "Linear Controls"
    if len(rows) >= 5:
        bracket_y = max(r['ci_hi'] for r in rows[:3]) + 0.04
        ax.annotate('', xy=(0, bracket_y), xytext=(2, bracket_y),
                    arrowprops=dict(arrowstyle='-', color='#555', lw=1.2))
        ax.plot([0, 0], [bracket_y - 0.005, bracket_y + 0.005], color='#555', lw=1.2)
        ax.plot([2, 2], [bracket_y - 0.005, bracket_y + 0.005], color='#555', lw=1.2)
        ax.text(1, bracket_y + 0.012, 'Linear Controls\n(premium persists)',
                ha='center', va='bottom', fontsize=9, fontstyle='italic', color='#555')

    ax.set_xticks(x)
    ax.set_xticklabels([r['label'] for r in rows], fontsize=10)
    ax.set_ylabel('CO₂ Coefficient (γ)', fontsize=13)
    ax.axhline(0, color='gray', ls='-', lw=0.8)
    ax.set_title('Carbon Premium Absorption:\nLinear Controls vs. Neural Prediction',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    save(fig, 'fig1_waterfall')



# ======================================================================
# FIG 2 — Conditional Carbon Premium by Firm Characteristic
# ======================================================================
def fig2_conditional_premium():
    """
    Heatmap or grouped bar chart showing CO₂ H-L spread conditional
    on firm characteristics (Size, BM, Leverage, IO, etc.)
    """
    cond_path = os.path.join(TBL, 'conditional_carbon_premium.csv')
    if not os.path.exists(cond_path):
        print('  ⚠️ conditional_carbon_premium.csv not found — skipping')
        return

    cond = pd.read_csv(cond_path)

    # Pivot: Variable × Group → spread
    variables = cond['Variable'].unique()
    groups = ['Low', 'Med', 'High']

    fig, ax = plt.subplots(figsize=(12, 6))

    n_vars = len(variables)
    x = np.arange(n_vars)
    width = 0.25
    colors_grp = {'Low': '#3498db', 'Med': '#f39c12', 'High': '#e74c3c'}

    for gi, grp in enumerate(groups):
        grp_data = cond[cond['Group'] == grp]
        spreads = []
        t_stats = []
        for var in variables:
            row = grp_data[grp_data['Variable'] == var]
            spreads.append(row['spread'].values[0] if len(row) > 0 else 0)
            t_stats.append(row['t'].values[0] if len(row) > 0 else 0)

        bars = ax.bar(x + (gi - 1) * width, spreads, width,
                      label=f'{grp} tercile', color=colors_grp[grp],
                      edgecolor='white', alpha=0.85)

        # Significance markers
        for i, (sp, t) in enumerate(zip(spreads, t_stats)):
            if abs(t) >= 1.96:
                marker = '**' if abs(t) >= 2.576 else '*'
                ax.text(x[i] + (gi - 1) * width, sp + 0.001 * np.sign(sp),
                        marker, ha='center', va='bottom' if sp > 0 else 'top',
                        fontsize=10, fontweight='bold', color='black')

    ax.set_xticks(x)
    ax.set_xticklabels(variables, fontsize=10, rotation=30, ha='right')
    ax.set_ylabel('CO₂ H-L Spread (monthly)', fontsize=12)
    ax.axhline(0, color='black', ls='-', lw=0.5)
    ax.set_title('Conditional Carbon Premium by Firm Characteristic\n(CO₂ High−Low spread within each tercile)',
                 fontsize=13, fontweight='bold')
    ax.legend(title='Conditioning Tercile', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    save(fig, 'fig2_conditional_premium')


# ======================================================================
# FIG 3 — Industry Heterogeneity
# ======================================================================
def fig3_industry():
    """
    Horizontal bar chart showing industry-level CO₂ premium,
    color-coded by significance.
    """
    ind_path = os.path.join(TBL, 'industry_carbon_premium.csv')
    if not os.path.exists(ind_path):
        print('  ⚠️ industry_carbon_premium.csv not found — skipping')
        return

    ind = pd.read_csv(ind_path)
    ind = ind.sort_values('raw_spread', ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(ind) * 0.35)))

    colors = []
    for _, row in ind.iterrows():
        if abs(row['raw_t']) >= 1.96:
            colors.append('#2ecc71')  # significant
        elif abs(row['raw_t']) >= 1.645:
            colors.append('#f39c12')  # marginal
        else:
            colors.append('#bdc3c7')  # insignificant

    y_pos = np.arange(len(ind))
    bars = ax.barh(y_pos, ind['raw_spread'], color=colors, edgecolor='white', height=0.7)

    # Labels with industry names
    industry_labels = []
    for _, row in ind.iterrows():
        name = row['Industry']
        if len(name) > 35:
            name = name[:32] + '...'
        industry_labels.append(name)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(industry_labels, fontsize=8)
    ax.axvline(0, color='black', ls='-', lw=0.5)
    ax.set_xlabel('CO₂ H-L Spread (monthly return)', fontsize=11)
    ax.set_title('Industry-Level Carbon Premium\n(CO₂ High−Low portfolio spread)',
                 fontsize=13, fontweight='bold')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#2ecc71', lw=8, label='Significant (p<0.05)'),
        Line2D([0], [0], color='#f39c12', lw=8, label='Marginal (p<0.10)'),
        Line2D([0], [0], color='#bdc3c7', lw=8, label='Insignificant'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    ax.grid(axis='x', alpha=0.3)

    save(fig, 'fig3_industry')


# ======================================================================
# FIG MECHANISM MAP — Methodological Framework
# ======================================================================
def fig_mechanism_map():
    """
    Methodological framework diagram.
    Redesigned to show the parallel hybrid architecture (LSTM || CA) of the neural model:
    - Stage 1 (Input Data): Characteristics ($X_{i,t}$) & Carbon Metrics ($CO_2$)
    - Stage 2 (Parallel Branches): LSTM Branch (Temporal Encoding) & CA Branch (Cross-Sectional Factor Pricing)
    - Stage 3 (Fusion & Prediction): Concatenation and Fusion MLP
    - Stage 4 (Residual Decomposition): Isolating neural residuals
    - Stage 5 (Testing Lenses): Lens A (Aggregate Pricing) & Lens B (Local Pricing)
    - Stage 6 (Empirical Resolution): Dual-lens synthesis
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Color system
    C_TEAL = '#008080'      # Stage 1 Characteristics
    C_GREEN = '#2e7d32'     # Stage 1 Carbon
    C_INDIGO = '#3f51b5'    # Branch 1 LSTM
    C_PURPLE = '#6a1b9a'    # Branch 2 CA
    C_BLUE = '#1565c0'      # Stage 3 Fusion
    C_RED = '#c62828'       # Stage 5 Lens A
    C_ORANGE = '#ef6c00'    # Stage 5 Lens B
    C_SLATE = '#37474f'     # Stage 4 Residuals & Stage 6 Resolution
    C_BG = '#f8f9fa'

    # Helper function for drawing elegant boxes
    def elegant_box(x, y, w, h, title, fill_color, border_color):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                              facecolor=fill_color, edgecolor=border_color, lw=1.5, alpha=0.9)
        ax.add_patch(rect)
        if '\n' in title:
            ax.text(x + w/2, y + h - 0.26, title, ha='center', va='center',
                    fontsize=8.2, fontweight='bold', color=border_color)
            ax.plot([x + 0.05, x + w - 0.05], [y + h - 0.52, y + h - 0.52], color=border_color, lw=0.6, alpha=0.4)
        else:
            ax.text(x + w/2, y + h - 0.22, title, ha='center', va='center',
                    fontsize=9.2, fontweight='bold', color=border_color)
            ax.plot([x + 0.05, x + w - 0.05], [y + h - 0.38, y + h - 0.38], color=border_color, lw=0.6, alpha=0.4)

    def elegant_arrow(x1, y1, x2, y2, color='#7f8c8d', connectionstyle=None, ls='-'):
        arrowprops = dict(arrowstyle='-|>', color=color, lw=1.2, ls=ls,
                          mutation_scale=10, patchA=None, patchB=None)
        if connectionstyle:
            arrowprops['connectionstyle'] = connectionstyle
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=arrowprops)

    # Column Separators
    for x_sep in [2.4, 4.7, 7.0, 9.3, 11.925]:
        ax.plot([x_sep, x_sep], [0.4, 7.0], color='#b0bec5', lw=0.8, ls='--')

    ax.text(1.25, 7.3, 'STAGE 1: INPUTS', ha='center', va='center', fontsize=9.5, fontweight='bold', color='#2c3e50')
    ax.text(3.55, 7.3, 'STAGE 2: MODEL\nBRANCHES', ha='center', va='center', fontsize=9.5, fontweight='bold', color='#2c3e50')
    ax.text(5.85, 7.3, 'STAGE 3: FUSION', ha='center', va='center', fontsize=9.5, fontweight='bold', color='#2c3e50')
    ax.text(8.15, 7.3, 'STAGE 4: RESIDUALS', ha='center', va='center', fontsize=9.5, fontweight='bold', color='#2c3e50')
    ax.text(10.6, 7.3, 'STAGE 5: LENS TESTS', ha='center', va='center', fontsize=9.5, fontweight='bold', color='#2c3e50')
    ax.text(12.975, 7.3, 'STAGE 6: RESOLUTION', ha='center', va='center', fontsize=9.5, fontweight='bold', color='#2c3e50')

    # ==========================================
    # COLUMN 1: INPUT DATA
    # ==========================================
    # 1. Firm Characteristics Grid
    elegant_box(0.3, 4.3, 1.9, 2.4, '1. Characteristics ($X_{i,t}$)', C_BG, C_TEAL)
    for i in range(13):
        r, c = divmod(i, 5)
        cx = 0.44 + c * 0.32
        cy = 4.95 + r * 0.30
        rect = mpatches.Rectangle((cx, cy), 0.24, 0.20, facecolor='#e0f2f1', edgecolor=C_TEAL, lw=0.6)
        ax.add_patch(rect)
    ax.text(1.25, 4.55, '13 Inputs (Size, B/M, beta...)\nLagged & z-scored',
            ha='center', va='center', fontsize=6.8, color='#37474f')

    # Carbon Indicators Stack
    elegant_box(0.3, 1.0, 1.9, 2.0, 'Carbon Metrics ($CO_2$)', C_BG, C_GREEN)
    metrics = ['Level ($CO_{2, i, t}$)', r'Growth ($\Delta CO_{2, i, t}$)', 'Intensity ($CI_{i, t}$)']
    for i, m in enumerate(metrics):
        cy = 2.45 - i * 0.45
        circle = mpatches.Circle((0.6, cy), 0.13, facecolor='#e8f5e9', edgecolor=C_GREEN, lw=0.8)
        ax.add_patch(circle)
        ax.text(0.6, cy, str(i+1), ha='center', va='center', fontsize=7.2, fontweight='bold', color=C_GREEN)
        ax.text(0.85, cy, m, ha='left', va='center', fontsize=7.2, fontweight='semibold', color='#37474f')
    ax.text(1.25, 1.15, '[ISOLATED TEST FEED]\nBypasses Neural Pipeline', ha='center', va='center', fontsize=6.8, fontweight='bold', color=C_GREEN)

    # ==========================================
    # COLUMN 2: PARALLEL BRANCHES
    # ==========================================
    # 2a. Temporal Encoding (LSTM)
    elegant_box(2.6, 4.3, 1.9, 2.4, '2a. LSTM (Temporal)', C_BG, C_INDIGO)
    t_labels = ['t-30', 't-29', '...', 't']
    for t_step in range(4):
        cx = 2.85 + t_step * 0.45
        cy = 5.75
        circle = mpatches.Circle((cx, cy), 0.11, facecolor='#e8eaf6', edgecolor=C_INDIGO, lw=0.8)
        ax.add_patch(circle)
        ax.text(cx, cy, t_labels[t_step], ha='center', va='center', fontsize=6.5, color=C_INDIGO)
        if t_step < 3:
            elegant_arrow(cx + 0.11, cy, cx + 0.34, cy, color=C_INDIGO)
    ax.text(3.55, 4.65, 'Processes 30-day paths\nOutputs temporal asset\nembeddings ($h_{i,t}$)',
            ha='center', va='center', fontsize=6.8, color='#37474f')

    # 2b. Cross-Sectional Pricing (CA)
    elegant_box(2.6, 1.0, 1.9, 2.0, '2b. CA\n(Cross-Sectional)', C_BG, C_PURPLE)
    y_in = [2.3, 2.0, 1.7]
    y_lat = [2.4, 2.1, 1.8, 1.5]
    y_out = [2.3, 2.0, 1.7]
    for yi in y_in:
        for yl in y_lat:
            ax.plot([2.9, 3.55], [yi, yl], color=C_PURPLE, lw=0.4, alpha=0.3)
    for yl in y_lat:
        for yo in y_out:
            ax.plot([3.55, 4.2], [yl, yo], color=C_PURPLE, lw=0.4, alpha=0.3)
    for yi in y_in:
        ax.add_patch(mpatches.Circle((2.9, yi), 0.07, facecolor='#f3e5f5', edgecolor=C_PURPLE, lw=0.5))
    for yl in y_lat:
        ax.add_patch(mpatches.Circle((3.55, yl), 0.06, facecolor='#6a1b9a', edgecolor='#4a148c', lw=0.5))
    for yo in y_out:
        ax.add_patch(mpatches.Circle((4.2, yo), 0.07, facecolor='#f3e5f5', edgecolor=C_PURPLE, lw=0.5))
    ax.text(3.55, 1.18, 'Extracts $K=5$ factors $f_t$\nand asset risk loadings ($\\beta_{i,t}$)\n$\\hat{R}^{CA}_{i,t} = \\beta_{i,t}\' f_t$',
            ha='center', va='center', fontsize=6.8, color='#37474f')

    # ==========================================
    # COLUMN 3: FUSION & PREDICTION
    # ==========================================
    # 3. Fusion & Prediction
    elegant_box(4.9, 2.9, 1.9, 2.4, '3. Fusion & Prediction', C_BG, C_BLUE)
    # Concatenation bar
    rect_cat = mpatches.Rectangle((5.1, 4.4), 1.5, 0.3, facecolor='#e3f2fd', edgecolor=C_BLUE, lw=0.8)
    ax.add_patch(rect_cat)
    ax.text(5.85, 4.55, r'$h_{i,t} \oplus \beta_{i,t} \oplus \hat{R}^{CA}_{i,t}$', ha='center', va='center', fontsize=8.0, color='#0d47a1', fontweight='bold')
    # Arrow from concat to MLP
    elegant_arrow(5.85, 4.4, 5.85, 4.05, color=C_BLUE)
    # MLP block
    rect_mlp = mpatches.Rectangle((5.3, 3.5), 1.1, 0.5, facecolor='#bbdefb', edgecolor='#0d47a1', lw=1.0)
    ax.add_patch(rect_mlp)
    ax.text(5.85, 3.75, 'Fusion MLP', ha='center', va='center', fontsize=8.0, color='#0d47a1', fontweight='bold')
    # Arrow from MLP to output text
    elegant_arrow(5.85, 3.5, 5.85, 3.3, color=C_BLUE)
    # Output text at bottom
    ax.text(5.85, 3.12, r'Prediction $\hat{R}^{NN}_{i,t}$', ha='center', va='center', fontsize=8.5, fontweight='bold', color='#1565c0')

    # ==========================================
    # COLUMN 4: RESIDUAL DECOMPOSITION
    # ==========================================
    # 4. Residual Decomposition
    elegant_box(7.2, 2.9, 1.9, 2.4, '4. Residual Decomp.', C_BG, C_SLATE)
    ax.text(8.15, 4.25, r'$\hat{\varepsilon}^{NN}_{i,t} = R_{i,t} - \hat{R}^{NN}_{i,t}$',
            ha='center', va='center', fontsize=8.8, fontweight='bold', color='#263238',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='#eceff1', edgecolor='#cfd8dc', lw=0.8))
    ax.text(8.15, 3.35, 'Isolates returns from\nnonlinear characteristic\nfactor risk',
            ha='center', va='center', fontsize=6.8, color='#37474f')

    # 5a. Lens A: Aggregate
    # Summary values (kept in sync with authoritative results tables):
    #   Linear      = M1 Bolton          -> absorption_confidence_intervals.csv (0.144***, t=3.10)
    #   Linear + NN = M7 Bolton+Neural    -> absorption_confidence_intervals.csv (-0.019, t=-0.58)
    #   NN Residual = Total CO2 Panel B   -> tab_neural_resid.tex (-0.128, t=-2.32**)
    elegant_box(9.35, 3.9, 2.45, 2.5, '5a. Lens A: Aggregate', C_BG, C_RED)
    ax.text(9.45, 5.65, 'Model', ha='left', va='center', fontsize=6.8, fontweight='bold', color='#78909c')
    ax.text(10.60, 5.65, 'Coef', ha='center', va='center', fontsize=6.8, fontweight='bold', color='#78909c')
    ax.text(11.42, 5.65, 'Outcome', ha='center', va='center', fontsize=6.8, fontweight='bold', color='#78909c')
    ax.plot([9.45, 11.70], [5.52, 5.52], color='#cfd8dc', lw=0.6)

    # Row 1: Linear Baseline
    ax.text(9.45, 5.10, 'Linear', ha='left', va='center', fontsize=6.8, color='#37474f')
    ax.text(10.60, 5.10, '0.144***', ha='center', va='center', fontsize=6.8, color='#37474f')
    ax.text(11.42, 5.10, 'Premium', ha='center', va='center', fontsize=6.5, color='#2e7d32', fontweight='bold', bbox=dict(boxstyle="square,pad=0.1", facecolor='#e8f5e9', edgecolor='#2e7d32', lw=0.4))

    # Row 2: Combined Model
    ax.text(9.45, 4.65, 'Linear + NN', ha='left', va='center', fontsize=6.8, color='#37474f')
    ax.text(10.60, 4.65, '-0.019', ha='center', va='center', fontsize=6.8, color='#37474f')
    ax.text(11.42, 4.65, 'No Premium', ha='center', va='center', fontsize=6.2, color='#78909c', bbox=dict(boxstyle="square,pad=0.1", facecolor='#eceff1', edgecolor='#cfd8dc', lw=0.4))

    # Row 3: Neural Residual
    ax.text(9.45, 4.20, 'NN Residual', ha='left', va='center', fontsize=6.8, color='#37474f')
    ax.text(10.60, 4.20, '-0.128**', ha='center', va='center', fontsize=6.8, color='#37474f')
    ax.text(11.42, 4.20, 'Discount', ha='center', va='center', fontsize=6.5, color=C_RED, fontweight='bold', bbox=dict(boxstyle="square,pad=0.1", facecolor='#ffebee', edgecolor=C_RED, lw=0.4))


    # 5b. Lens B: Conditional & CATE
    elegant_box(9.35, 1.0, 2.45, 2.5, '5b. Lens B:\nConditional & CATE', C_BG, C_ORANGE)
    ax.text(9.40, 2.75, 'Group', ha='left', va='center', fontsize=6.8, fontweight='bold', color='#78909c')
    ax.text(10.25, 2.75, 'Spread', ha='center', va='center', fontsize=6.8, fontweight='bold', color='#78909c')
    ax.text(10.95, 2.75, 'CATE', ha='center', va='center', fontsize=6.8, fontweight='bold', color='#78909c')
    ax.text(11.58, 2.75, 'Outcome', ha='center', va='center', fontsize=6.8, fontweight='bold', color='#78909c')
    ax.plot([9.40, 11.70], [2.62, 2.62], color='#cfd8dc', lw=0.6)

    # Row 1: Size Small
    ax.text(9.40, 2.25, 'Small', ha='left', va='center', fontsize=6.5, color='#37474f')
    ax.text(10.25, 2.25, '+0.48%', ha='center', va='center', fontsize=6.5, color='#37474f')
    ax.text(10.95, 2.25, '+0.80%***', ha='center', va='center', fontsize=6.5, color='#37474f')
    ax.text(11.58, 2.25, 'Premium', ha='center', va='center', fontsize=6.2, color='#2e7d32', fontweight='bold', bbox=dict(boxstyle="square,pad=0.1", facecolor='#e8f5e9', edgecolor='#2e7d32', lw=0.4))

    # Row 2: Size Large
    ax.text(9.40, 1.95, 'Large', ha='left', va='center', fontsize=6.5, color='#37474f')
    ax.text(10.25, 1.95, '-0.70%*', ha='center', va='center', fontsize=6.5, color='#37474f')
    ax.text(10.95, 1.95, '-0.31%***', ha='center', va='center', fontsize=6.5, color='#37474f')
    ax.text(11.58, 1.95, 'Discount', ha='center', va='center', fontsize=6.2, color=C_RED, fontweight='bold', bbox=dict(boxstyle="square,pad=0.1", facecolor='#ffebee', edgecolor=C_RED, lw=0.4))

    # Row 3: BM Value
    ax.text(9.40, 1.65, 'Value', ha='left', va='center', fontsize=6.5, color='#37474f')
    ax.text(10.25, 1.65, '+0.56%*', ha='center', va='center', fontsize=6.5, color='#37474f')
    ax.text(10.95, 1.65, '+0.67%***', ha='center', va='center', fontsize=6.5, color='#37474f')
    ax.text(11.58, 1.65, 'Premium', ha='center', va='center', fontsize=6.2, color='#2e7d32', fontweight='bold', bbox=dict(boxstyle="square,pad=0.1", facecolor='#e8f5e9', edgecolor='#2e7d32', lw=0.4))

    # Row 4: BM Growth
    ax.text(9.40, 1.35, 'Growth', ha='left', va='center', fontsize=6.5, color='#37474f')
    ax.text(10.25, 1.35, '-0.25%', ha='center', va='center', fontsize=6.5, color='#37474f')
    ax.text(10.95, 1.35, '-0.26%***', ha='center', va='center', fontsize=6.5, color='#37474f')
    ax.text(11.58, 1.35, 'Discount', ha='center', va='center', fontsize=6.2, color=C_RED, fontweight='bold', bbox=dict(boxstyle="square,pad=0.1", facecolor='#ffebee', edgecolor=C_RED, lw=0.4))


    # ==========================================
    # COLUMN 6: RESOLUTION SYNTHESIS
    # ==========================================
    elegant_box(12.05, 0.8, 1.85, 5.9, '6. Resolution', C_BG, C_SLATE)
    
    synthesis_points = [
        ("AGGREGATE LEVEL", C_RED),
        ("Carbon premium is a proxy\neffect, fully absorbed by\nnonlinear characteristic\ninteractions (t = -0.58).", '#2c3e50'),
        ("", '#2c3e50'),
        ("LOCAL LEVEL", C_ORANGE),
        ("Pricing is a localized\nfriction, concentrating\nin small-cap & value\nfirms. Causal Forest\nconfirms (Table 9).", '#2c3e50'),
        ("", '#2c3e50'),
        ("RESOLUTION", C_SLATE),
        ("Carbon risk is a localized\nfriction driven by pricing\ndifficulty, not a systemic\nmarket-wide risk factor.", '#2c3e50')

    ]
    
    curr_y = 5.95
    for title, text_color in synthesis_points:
        if title == "AGGREGATE LEVEL" or title == "LOCAL LEVEL" or title == "RESOLUTION":
            ax.text(12.15, curr_y, title, ha='left', va='center', fontsize=7.2, fontweight='bold', color=text_color)
            curr_y -= 0.18
        elif title == "":
            curr_y -= 0.10
        else:
            ax.text(12.15, curr_y, title, ha='left', va='top', fontsize=6.6, color=text_color, wrap=True)
            lines = title.count('\n') + 1
            curr_y -= (lines * 0.138 + 0.05)

    # ==========================================
    # CAUSAL FLOWS (PARALLEL & ROUTED)
    # ==========================================
    # Stage 1 Characteristics splits to LSTM (top) and CA (bottom)
    elegant_arrow(2.2, 5.5, 2.6, 5.5, color=C_TEAL)       # Characteristics -> LSTM
    elegant_arrow(2.2, 5.2, 2.6, 2.0, color=C_TEAL)       # Characteristics -> CA (diagonal)

    # LSTM and CA branches merge into Stage 3 Fusion
    elegant_arrow(4.5, 5.5, 4.9, 4.55, color=C_INDIGO)    # LSTM -> Fusion (diagonal down)
    elegant_arrow(4.5, 2.0, 4.9, 3.75, color=C_PURPLE)    # CA -> Fusion (diagonal up)

    # Fusion -> Residual Decomp
    elegant_arrow(6.8, 4.1, 7.2, 4.1, color=C_SLATE)

    # Residuals flow to Stage 5 Lenses
    elegant_arrow(9.1, 5.15, 9.35, 5.15, color=C_SLATE)    # Residuals -> Lens A
    elegant_arrow(9.1, 2.85, 9.35, 2.25, color=C_SLATE)    # Residuals -> Lens B (diagonal down to Row 1)

    # Segmented Routing for Carbon Metrics (bypasses deep learning, routed under Col 2-4 at y=0.45)
    ax.plot([2.2, 2.4, 2.4, 9.15, 9.15], [1.3, 1.3, 0.45, 0.45, 1.65], color=C_GREEN, ls='--', lw=1.2)
    # Arrow to Lens B:
    elegant_arrow(9.15, 1.65, 9.35, 1.65, color=C_GREEN, ls='--')
    # Vertical rise and arrow to Lens A:
    ax.plot([9.15, 9.15], [1.65, 4.65], color=C_GREEN, ls='--', lw=1.2)
    elegant_arrow(9.15, 4.65, 9.35, 4.65, color=C_GREEN, ls='--')

    # Add label for the bypassed feed
    ax.text(5.8, 0.55, 'CO2 Evaluation Feed', ha='center', va='bottom', fontsize=6.8, color=C_GREEN, fontweight='semibold')

    # Column 5 Lenses -> Column 6 Synthesis
    elegant_arrow(11.8, 5.15, 12.05, 5.0, color=C_RED, connectionstyle="arc3,rad=-0.08")
    elegant_arrow(11.8, 2.25, 12.05, 3.6, color=C_ORANGE, connectionstyle="arc3,rad=0.08")



    save(fig, 'fig_mechanism_map')


# ======================================================================
# FIG 4 -- Cumulative CO2 H-L Spread Returns (EW vs VW)
# ======================================================================
def fig4_cumulative_returns():
    """
    Cumulative return of CO2 High-Low portfolio spread (EW vs VW).
    Shows the time-series evolution of the carbon premium.
    """
    panel_path = os.path.join(BASE, 'results', 'analysis_panel_monthly.csv')
    if not os.path.exists(panel_path):
        print('  ⚠️ analysis_panel_monthly.csv not found -- skipping cumulative returns')
        return

    df = pd.read_csv(panel_path)
    co2_col = 'LOG_CO2_TOTAL_L1'
    ret_col = 'Actual'

    if co2_col not in df.columns or ret_col not in df.columns:
        print(f'  ⚠️ Required columns ({co2_col}, {ret_col}) missing -- skipping')
        return

    # Drop rows with missing CO2 or return
    sub = df.dropna(subset=[co2_col, ret_col]).copy()
    sub['YM'] = sub['YearMonth']

    # Monthly tercile assignment (transform keeps all columns incl. YM;
    # robust across pandas versions where groupby-apply may drop the key)
    def qcut_safe(x):
        try:
            return pd.qcut(x, 3, labels=['Low', 'Med', 'High'])
        except ValueError:
            return pd.Series(np.nan, index=x.index)

    sub['tercile'] = sub.groupby('YM')[co2_col].transform(qcut_safe)
    sub = sub.dropna(subset=['tercile'])

    # EW returns by tercile-month
    ew = sub.groupby(['YM', 'tercile'])[ret_col].mean().unstack('tercile')
    ew_hl = ew['High'] - ew['Low']

    # VW returns (use exp(SIZE_L1) as MCAP proxy)
    if 'SIZE_L1' in sub.columns:
        sub['MCAP_proxy'] = np.exp(sub['SIZE_L1'])

        def vw_return(group):
            w = group['MCAP_proxy'] / group['MCAP_proxy'].sum()
            return (w * group[ret_col]).sum()

        vw = sub.groupby(['YM', 'tercile']).apply(vw_return).unstack('tercile')
        vw_hl = vw['High'] - vw['Low']
    else:
        vw_hl = None

    # Convert to datetime for plotting
    ew_dates = pd.to_datetime(ew_hl.index, format='%Y-%m')
    if vw_hl is not None:
        vw_dates = pd.to_datetime(vw_hl.index, format='%Y-%m')

    # Cumulative returns
    cum_ew = (1 + ew_hl.values).cumprod() - 1
    cum_vw = (1 + vw_hl.values).cumprod() - 1 if vw_hl is not None else None

    fig, ax = plt.subplots(figsize=(12, 5.5))

    t_ew = ew_hl.mean() / ew_hl.std() * np.sqrt(len(ew_hl))
    ax.plot(ew_dates, cum_ew * 100, color='#2980b9', lw=2.0,
            label=f'EW H-L (mean={ew_hl.mean()*100:.3f}%/mo, t={t_ew:.2f})')

    if cum_vw is not None:
        t_vw = vw_hl.mean() / vw_hl.std() * np.sqrt(len(vw_hl))
        ax.plot(vw_dates, cum_vw * 100, color='#e74c3c', lw=2.0, ls='--',
                label=f'VW H-L (mean={vw_hl.mean()*100:.3f}%/mo, t={t_vw:.2f})')

    ax.axhline(0, color='gray', ls='-', lw=0.5)
    ax.fill_between(ew_dates, 0, cum_ew * 100, alpha=0.08, color='#2980b9')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax.set_title('Cumulative CO$_2$ High-Minus-Low Spread Returns\n(Equal-Weighted vs Value-Weighted)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(axis='both', alpha=0.3)

    # Mark COVID period
    import datetime
    covid_start = datetime.datetime(2020, 2, 1)
    covid_end = datetime.datetime(2020, 6, 1)
    if ew_dates.min() < covid_start < ew_dates.max():
        ax.axvspan(covid_start, covid_end, alpha=0.1, color='red', label='_nolegend_')
        ax.text(covid_start, ax.get_ylim()[1] * 0.9, 'COVID', fontsize=8, color='red', alpha=0.7)

    save(fig, 'fig4_cumulative_returns')


# ======================================================================
# MAIN
# ======================================================================
def fig5_rolling_co2_gamma():
    """
    Rolling 24-month Fama-MacBeth CO2 gamma with 95% CI band.
    Shows time-varying carbon premium: negative during COVID, positive post-ESG,
    fading toward end of sample.
    """
    rolling_path = os.path.join(TBL, 'rolling_co2_gamma.csv')
    if not os.path.exists(rolling_path):
        print('  rolling_co2_gamma.csv not found -- skipping')
        return

    df = pd.read_csv(rolling_path)
    df['YearMonth'] = pd.to_datetime(df['YearMonth'])

    fig, ax = plt.subplots(figsize=(10, 5))

    # CI band
    ax.fill_between(df['YearMonth'], df['rolling_ci_lo'], df['rolling_ci_hi'],
                     alpha=0.2, color='steelblue', label='95% CI')
    # Gamma line
    ax.plot(df['YearMonth'], df['rolling_gamma'], color='steelblue', lw=2,
            label='Rolling 24-month $\\hat{\\gamma}_{CO_2}$')
    # Zero line
    ax.axhline(0, color='black', ls='-', lw=0.8)

    # COVID shading
    covid_start = pd.Timestamp('2020-03-01')
    covid_end = pd.Timestamp('2020-12-31')
    ymin, ymax = ax.get_ylim()
    ax.axvspan(covid_start, covid_end, alpha=0.08, color='red', label='COVID-19')

    # Paris/ESG marker
    ax.axvline(pd.Timestamp('2021-01-01'), color='grey', ls='--', lw=0.8, alpha=0.6)
    ax.text(pd.Timestamp('2021-02-01'), ax.get_ylim()[1] * 0.9, 'Post-COVID\nESG regime',
            fontsize=8, color='grey', ha='left', va='top')

    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('$\\hat{\\gamma}_{CO_2}$ (FMB coefficient)', fontsize=11)
    ax.set_title('Rolling 24-Month Fama-MacBeth CO$_2$ Gamma', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    out = os.path.join(FIG, 'fig5_rolling_co2_gamma.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f'  fig5_rolling_co2_gamma -> {out}')


if __name__ == '__main__':
    print('='*60)
    print('  GENERATING ALL MANUSCRIPT FIGURES (K=5, taxonomy-free)')
    print(f'  Output: {FIG}')
    print('='*60)

    fig1_waterfall()
    fig2_conditional_premium()
    fig3_industry()
    fig_mechanism_map()
    fig4_cumulative_returns()
    fig5_rolling_co2_gamma()

    # No copying to manuscript/ root needed; LaTeX uses relative paths pointing directly to figures/ subfolder.

    print(f'\nAll figures saved to {FIG}')
