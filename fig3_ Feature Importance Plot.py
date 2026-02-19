# -*- coding: utf-8 -*-
"""
Created on Sat Oct 11 04:00:24 2025

@author: Ruba
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Example: Get feature importances from your logistic regression model
# If using sklearn LogisticRegression, coefficients represent importance
# from sklearn.linear_model import LogisticRegression
# model = LogisticRegression(penalty='l2', C=0.316)
# model.fit(X_train, y_train)
# coefs = np.abs(model.coef_).mean(axis=0)  # For multi-class, average across classes

# REPLACE THIS with your actual feature importances
feature_names = [
    'INT_mean', 'AT_mean', 'NR_mean', 'AVL_mean', 'CON_mean',
    'INT_SD', 'AT_SD', 'DL_mean', 'IA_mean', 'NR_high_count',
    'INT_high_count', 'AT_high_count', 'AVL_SD', 'Satisfaction_rate',
    'IA_CON_ratio', 'INT_range', 'Global_consistency', 'AT_range',
    'NR_SD', 'CON_SD'
]

# Example importance scores (replace with actual values)
importances = np.array([
    0.45, 0.42, 0.38, 0.35, 0.32,
    0.28, 0.25, 0.23, 0.21, 0.19,
    0.18, 0.17, 0.16, 0.15, 0.14,
    0.13, 0.12, 0.11, 0.10, 0.09
])

# Create DataFrame and sort
df_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=True)

# Take top 20
df_top = df_importance.tail(20)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_top)))

bars = ax.barh(df_top['Feature'], df_top['Importance'], color=colors, edgecolor='black')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, df_top['Importance'])):
    ax.text(val + 0.01, i, f'{val:.3f}', 
            va='center', fontsize=10, fontweight='bold')

ax.set_xlabel('Absolute Coefficient (Importance)', fontsize=14, fontweight='bold')
ax.set_ylabel('Feature', fontsize=14, fontweight='bold')
ax.set_title('Top 20 Features for IA Cluster Prediction\n(L2-Regularized Logistic Regression)', 
             fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('feature_importance.pdf', dpi=300, bbox_inches='tight')
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()