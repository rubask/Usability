import pandas as pd
import numpy as np
import semopy
from semopy import Model
import matplotlib.pyplot as plt
import networkx as nx

# ============================================================================
# LOAD DATA
# ============================================================================
df = pd.read_csv(r"C:\Users\rubas\Usability\Information_Assurance_Clustered.csv")  # Replace with your actual file path
print(f"Original dataset: {df.shape[0]} rows")

# Conditional columns (excluded)
conditional_cols = ['Educational_Other', 'Employment_Other']

# Survey items
survey_items = ['CON1', 'CON2', 'CON3', 'CON4', 'INT1', 'INT2', 'INT3',
                'AVL1', 'AVL2', 'AVL3', 'AT1', 'AT2', 'AT3', 'AT4',
                'NR1', 'NR2', 'NR3', 'NR4', 'IA1', 'IA2', 'IA3', 'IA4', 'IA5']

for col in survey_items:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Encode categorical
df['Gender_numeric'] = pd.factorize(df['Gender'])[0]

age_order = {'Less than 18': 0, '18-24 years old': 1, '25-34 years old': 2, 
             '35-44 years old': 3, '45-54 years old': 4, '55-64 years old': 5, '65+ years old': 6}
df['Age_numeric'] = df['Age_Group'].map(age_order)
if df['Age_numeric'].isnull().any():
    df['Age_numeric'] = df['Age_numeric'].fillna(pd.factorize(df['Age_Group'])[0])

usage_order = {'Less than 1 hour': 1, '1-2 hours': 2, '2-3 hours': 3,
               '3-4 hours': 4, '4-5 hours': 5, 'More than 5 hours': 6}
df['Usage_numeric'] = df['Daily_App_Usage'].map(usage_order)
if df['Usage_numeric'].isnull().any():
    df['Usage_numeric'] = df['Usage_numeric'].fillna(pd.factorize(df['Daily_App_Usage'])[0])

# Clean dataset
analysis_cols = survey_items + ['Gender_numeric', 'Age_numeric', 'Usage_numeric']
df_clean = df[analysis_cols].dropna()

print(f"Clean dataset: {len(df_clean)} rows")
print("\n" + "="*80)
print("FITTING SEM MODEL")
print("="*80)

# ============================================================================
# FIT SEM MODEL
# ============================================================================

model_sem = """
# Measurement models
CON =~ CON1 + CON2 + CON3 + CON4
INT =~ INT1 + INT2 + INT3
AVL =~ AVL1 + AVL2 + AVL3
AT =~ AT1 + AT2 + AT3 + AT4
NR =~ NR1 + NR2 + NR3 + NR4
IA =~ IA1 + IA2 + IA3 + IA4 + IA5

# Structural model
IA ~ CON + INT + AVL + AT + NR + Age_numeric + Gender_numeric + Usage_numeric

# Correlations
CON ~~ INT
CON ~~ AVL
INT ~~ AVL
AT ~~ NR
"""

model = Model(model_sem)
result = model.fit(df_clean, solver='SLSQP')
print("✓ Model fitted successfully!\n")

# Get estimates
estimates = model.inspect()
std_estimates = model.inspect(std_est=True)

# Save
estimates.to_csv('sem_estimates.csv', index=False)
std_estimates.to_csv('sem_std_estimates.csv', index=False)

# ============================================================================
# FIT STATISTICS (FIXED)
# ============================================================================

print("="*80)
print("FIT INDICES")
print("="*80)

fit_stats = semopy.calc_stats(model)
print(fit_stats)

# Extract specific indices
try:
    chi2 = float(fit_stats.loc['chi-square', 'Value'])
    df_model = float(fit_stats.loc['DoF', 'Value'])
    chi2_df = chi2 / df_model
    
    cfi = float(fit_stats.loc['CFI', 'Value']) if 'CFI' in fit_stats.index else None
    tli = float(fit_stats.loc['TLI', 'Value']) if 'TLI' in fit_stats.index else None
    rmsea = float(fit_stats.loc['RMSEA', 'Value']) if 'RMSEA' in fit_stats.index else None
    srmr = float(fit_stats.loc['SRMR', 'Value']) if 'SRMR' in fit_stats.index else None
    
    print("\n" + "="*80)
    print("FIT INTERPRETATION")
    print("="*80)
    print(f"χ² = {chi2:.2f}, df = {df_model:.0f}")
    print(f"χ²/df = {chi2_df:.3f} {'✓ Good' if chi2_df < 3 else '✗ Poor'} (threshold < 3)")
    
    if cfi:
        print(f"CFI = {cfi:.3f} {'✓ Good' if cfi >= 0.95 else '~ Acceptable' if cfi >= 0.90 else '✗ Poor'} (threshold ≥ 0.95)")
    if tli:
        print(f"TLI = {tli:.3f} {'✓ Good' if tli >= 0.95 else '~ Acceptable' if tli >= 0.90 else '✗ Poor'} (threshold ≥ 0.95)")
    if rmsea:
        print(f"RMSEA = {rmsea:.3f} {'✓ Good' if rmsea < 0.06 else '~ Acceptable' if rmsea < 0.08 else '✗ Poor'} (threshold < 0.06)")
    if srmr:
        print(f"SRMR = {srmr:.3f} {'✓ Good' if srmr < 0.08 else '✗ Poor'} (threshold < 0.08)")
        
except Exception as e:
    print(f"Note: Some fit indices could not be extracted: {e}")

# ============================================================================
# FACTOR LOADINGS (FIXED)
# ============================================================================

print("\n" + "="*80)
print("FACTOR LOADINGS (Standardized)")
print("="*80)

loadings = std_estimates[std_estimates['op'] == '=~'].copy()
# Only keep items loading onto constructs (exclude reference indicators)
loadings_display = loadings[loadings['lval'].isin(['CON1', 'CON2', 'CON3', 'CON4',
                                                     'INT1', 'INT2', 'INT3',
                                                     'AVL1', 'AVL2', 'AVL3',
                                                     'AT1', 'AT2', 'AT3', 'AT4',
                                                     'NR1', 'NR2', 'NR3', 'NR4',
                                                     'IA1', 'IA2', 'IA3', 'IA4', 'IA5'])].copy()

# Rename columns for clarity
loadings_display.columns = ['Item', 'op', 'Construct', 'Loading (Unst.)', 'Loading (Std.)', 'SE', 'z', 'p-value']
loadings_display = loadings_display[['Construct', 'Item', 'Loading (Std.)', 'SE', 'p-value']]

# Add significance stars
def add_stars(pval):
    # Handle string p-values
    if isinstance(pval, str):
        if pval == '-':
            return '-'
        try:
            pval = float(pval)
        except:
            return ''
    
    if pd.isna(pval):
        return ''
    if pval < 0.001:
        return '***'
    elif pval < 0.01:
        return '**'
    elif pval < 0.05:
        return '*'
    else:
        return 'ns'

loadings_display['Sig.'] = loadings_display['p-value'].apply(add_stars)

print(loadings_display.to_string(index=False))
loadings_display.to_csv('factor_loadings.csv', index=False)

# ============================================================================
# STRUCTURAL PATHS (FIXED)
# ============================================================================

print("\n" + "="*80)
print("STRUCTURAL PATHS (Standardized)")
print("="*80)

structural = std_estimates[std_estimates['op'] == '~'].copy()
# Keep only IA ~ predictors (not measurement model)
structural = structural[structural['lval'] == 'IA'].copy()

structural['Path'] = 'IA ← ' + structural['rval']
structural['β'] = structural['Est. Std'].round(3)  # Use standardized estimate
structural['SE'] = structural['Std. Err'].round(3)

# Apply add_stars function
structural['Sig.'] = structural['p-value'].apply(add_stars)

structural_table = structural[['Path', 'β', 'SE', 'p-value', 'Sig.']].copy()
print(structural_table.to_string(index=False))
structural_table.to_csv('structural_paths.csv', index=False)

# ============================================================================
# INTERPRETATION SUMMARY
# ============================================================================

print("\n" + "="*80)
print("KEY FINDINGS SUMMARY")
print("="*80)

print("\nStructural Paths (IA Pillars → General IA):")
for idx, row in structural_table.iterrows():
    predictor = row['Path'].replace('IA ← ', '')
    beta = row['β']
    sig = row['Sig.']
    if sig in ['*', '**', '***']:
        effect = "positive" if beta > 0 else "negative"
        print(f"  {predictor:15s}: β = {beta:6.3f} {sig:3s} ({effect} effect)")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("CREATING PATH DIAGRAM")
print("="*80)

# Get significant paths only
sig_structural = structural_table[structural_table['Sig.'].isin(['*', '**', '***'])].copy()

G = nx.DiGraph()

# Positions
pos = {
    'CON': (0, 5), 'INT': (0, 4), 'AVL': (0, 3),
    'AT': (0, 2), 'NR': (0, 1), 'IA': (5, 3),
    'Age_numeric': (2.5, 0), 'Gender_numeric': (5, 0), 'Usage_numeric': (7.5, 0)
}

# Add edges
for idx, row in sig_structural.iterrows():
    predictor = row['Path'].replace('IA ← ', '')
    G.add_edge(predictor, 'IA', weight=row['β'], pval=row['p-value'], sig=row['Sig.'])

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))

# Node colors
node_colors = []
for node in G.nodes():
    if node in ['CON', 'INT', 'AVL', 'AT', 'NR']:
        node_colors.append('#3498db')
    elif node == 'IA':
        node_colors.append('#e74c3c')
    else:
        node_colors.append('#95a5a6')

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=5000, alpha=0.9, ax=ax)

edges = G.edges()
weights = [abs(G[u][v]['weight']) * 5 for u, v in edges]
nx.draw_networkx_edges(G, pos, width=weights, alpha=0.6, edge_color='black',
                       arrows=True, arrowsize=30, arrowstyle='->',
                       connectionstyle='arc3,rad=0.1', ax=ax)

labels = {node: node.replace('_numeric', '') for node in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, font_size=14, font_weight='bold', ax=ax)

# Edge labels
edge_labels = {}
for u, v, d in G.edges(data=True):
    edge_labels[(u, v)] = f"β={d['weight']:.2f}{d['sig']}"

nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=11,
                             font_color='darkred', font_weight='bold', ax=ax)

plt.title(f'SEM: IA Pillars → General IA Perception\nN = {len(df_clean)}, χ²/df = {chi2_df:.2f}',
          fontsize=16, fontweight='bold', pad=20)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#3498db', label='IA Pillars'),
    Patch(facecolor='#e74c3c', label='General IA'),
    Patch(facecolor='#95a5a6', label='Controls'),
]
plt.legend(handles=legend_elements, loc='lower left', fontsize=12)

plt.text(0.02, 0.98, '*** p < .001, ** p < .01, * p < .05',
         transform=ax.transAxes, fontsize=10, verticalalignment='top', style='italic')

plt.axis('off')
plt.tight_layout()

plt.savefig('sem_path_diagram.pdf', dpi=300, bbox_inches='tight')
plt.savefig('sem_path_diagram.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Path diagram saved!")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("Files created:")
print("  - sem_estimates.csv")
print("  - sem_std_estimates.csv")
print("  - factor_loadings.csv")
print("  - structural_paths.csv")
print("  - sem_path_diagram.pdf")
print("  - sem_path_diagram.png")