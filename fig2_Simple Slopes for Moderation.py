# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 03:59:58 2025

@author: Ruba
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Define your regression parameters (replace with actual values)
# For INT -> TR moderated by DL
intercept_INT = 1.0  # Replace with your intercept
beta_INT = 0.5       # Main effect of INT on TR
beta_DL = 0.3        # Main effect of DL on TR
beta_INT_DL = 0.2    # Interaction effect

# Define DL levels (mean ± 1 SD)
DL_low = 3.91 - 0.64    # M - 1SD
DL_med = 3.91           # M
DL_high = 3.91 + 0.64   # M + 1SD

# Create range of INT values
INT_range = np.linspace(1, 5, 100)

# Calculate TR for each DL level
TR_low = intercept_INT + beta_INT * INT_range + beta_DL * DL_low + beta_INT_DL * INT_range * DL_low
TR_med = intercept_INT + beta_INT * INT_range + beta_DL * DL_med + beta_INT_DL * INT_range * DL_med
TR_high = intercept_INT + beta_INT * INT_range + beta_DL * DL_high + beta_INT_DL * INT_range * DL_high

# Plot
plt.figure(figsize=(10, 6))
plt.plot(INT_range, TR_low, label=f'Low DL (M-1SD = {DL_low:.2f})', 
         linewidth=2.5, linestyle='--', color='#e74c3c')
plt.plot(INT_range, TR_med, label=f'Medium DL (M = {DL_med:.2f})', 
         linewidth=2.5, linestyle='-', color='#3498db')
plt.plot(INT_range, TR_high, label=f'High DL (M+1SD = {DL_high:.2f})', 
         linewidth=2.5, linestyle='-.', color='#2ecc71')

plt.xlabel('Integrity (INT)', fontsize=14, fontweight='bold')
plt.ylabel('Trust (TR)', fontsize=14, fontweight='bold')
plt.title('Moderation Effect of Digital Literacy on INT → TR Relationship', 
          fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save
plt.savefig('simple_slopes_INT.pdf', dpi=300, bbox_inches='tight')
plt.savefig('simple_slopes_INT.png', dpi=300, bbox_inches='tight')
plt.show()

# Repeat for AT -> TR if needed
# Just change the parameters and create a second plot