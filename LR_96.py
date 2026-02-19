import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

# ================================
# 1. Load Dataset
# ================================
df = pd.read_csv("Information_Assurance_Database_with_Clusters.csv")
feature_cols = df.columns[:-3].tolist()
X = df[feature_cols].copy()
y = df["Cluster_Number"].copy()

# Encode categorical variables
categorical_cols = ['Gender', 'Age_Group', 'Marital_Status', 'Educational_Level',
                    'Educational_Other', 'Employment_Status', 'Employment_Other', 'Daily_App_Usage']

for col in categorical_cols:
    if col in X.columns:
        X[col] = X[col].fillna("Unknown")
        mapping = {val: i for i, val in enumerate(sorted(X[col].unique()))}
        X[col] = X[col].map(mapping)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================================
# 2. Feature Engineering (Same as Optimizer)
# ================================
def create_engineered_features(X_train, X_test):
    X_train_eng = X_train.copy()
    X_test_eng = X_test.copy()

    survey_categories = {
        'IA': [c for c in X_train.columns if c.startswith('IA')],
        'CON': [c for c in X_train.columns if c.startswith('CON')],
        'INT': [c for c in X_train.columns if c.startswith('INT')],
        'AVL': [c for c in X_train.columns if c.startswith('AVL')],
        'AT': [c for c in X_train.columns if c.startswith('AT')],
        'NR': [c for c in X_train.columns if c.startswith('NR')],
        'DL': [c for c in X_train.columns if c.startswith('DL')]
    }

    for category, cols in survey_categories.items():
        if cols:
            for X_eng in [X_train_eng, X_test_eng]:
                X_eng[f'{category}_mean'] = X_eng[cols].mean(axis=1)
                X_eng[f'{category}_sum'] = X_eng[cols].sum(axis=1)
                X_eng[f'{category}_std'] = X_eng[cols].std(axis=1).fillna(0)
                X_eng[f'{category}_max'] = X_eng[cols].max(axis=1)
                X_eng[f'{category}_min'] = X_eng[cols].min(axis=1)
                X_eng[f'{category}_range'] = X_eng[f'{category}_max'] - X_eng[f'{category}_min']
                X_eng[f'{category}_high_count'] = (X_eng[cols] >= 4).sum(axis=1)
                X_eng[f'{category}_perfect_count'] = (X_eng[cols] == 5).sum(axis=1)
                X_eng[f'{category}_low_count'] = (X_eng[cols] <= 2).sum(axis=1)

    pairs = [('IA_mean', 'DL_mean'), ('IA_mean', 'CON_mean'),
             ('DL_mean', 'AT_mean'), ('CON_mean', 'INT_mean'),
             ('AVL_mean', 'NR_mean')]

    for cat1, cat2 in pairs:
        for X_eng in [X_train_eng, X_test_eng]:
            if cat1 in X_eng.columns and cat2 in X_eng.columns:
                X_eng[f'{cat1}_{cat2}_ratio'] = X_eng[cat1] / (X_eng[cat2] + 0.001)
                X_eng[f'{cat1}_{cat2}_diff'] = X_eng[cat1] - X_eng[cat2]

    survey_cols = [c for c in X_train.columns if any(c.startswith(cat) for cat in survey_categories)]
    for X_eng in [X_train_eng, X_test_eng]:
        X_eng["overall_mean"] = X_eng[survey_cols].mean(axis=1)
        X_eng["overall_satisfaction"] = (X_eng[survey_cols] >= 4).sum(axis=1) / len(survey_cols)
        X_eng["overall_consistency"] = 1 / (X_eng[survey_cols].std(axis=1) + 0.001)

    X_train_eng = X_train_eng.replace([np.inf, -np.inf], 0).fillna(0)
    X_test_eng = X_test_eng.replace([np.inf, -np.inf], 0).fillna(0)

    return X_train_eng, X_test_eng

X_train_eng, X_test_eng = create_engineered_features(X_train, X_test)

# ================================
# 3. Add Feature Selection + Best Logistic Regression
# ================================
best_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('select', SelectKBest(score_func=f_classif, k=50)),  # <-- CRITICAL FIX
    ('classifier', LogisticRegression(
        penalty='l1',
        C=1.1145319132082434,
        solver='liblinear',
        max_iter=3449,
        random_state=42
    ))
])

best_lr.fit(X_train_eng, y_train)

# ================================
# 4. Evaluate Model
# ================================
y_pred = best_lr.predict(X_test_eng)
accuracy = accuracy_score(y_test, y_pred)

print("\n🏆 BEST LOGISTIC REGRESSION RESULTS")
print("====================================")
print(f"✅ Test Accuracy: {accuracy:.6f} ({accuracy*100:.2f}%)")

# Save trained model
joblib.dump(best_lr, "best_logistic_regression_engineered.pkl")
print("💾 Model saved as best_logistic_regression_engineered.pkl")
