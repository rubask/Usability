# -*- coding: utf-8 -*-
"""
Enhanced multi-model trainer with rich metrics and timing.

- Computes accuracy, balanced accuracy, precision (macro/weighted),
  recall (macro/weighted), F1 (macro/weighted), log loss, ROC-AUC (OvR & OvO, weighted).
- Captures training time and prediction latency.
- Uses cross_validate to compute multiple CV metrics.
- Produces a tidy comparison DataFrame and saves it to CSV.
- Keeps your original preprocessing (ordinal encodings + one-hot for nominal).
- Works for multi-class targets in 'Cluster_Number'.

Author: (based on original by Ruba)
Date: 2025-09-06
"""
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    log_loss, roc_auc_score, classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class EnhancedIAClusterPredictor:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {
            'random_forest': RandomForestClassifier(random_state=random_state, n_estimators=300),
            'gradient_boosting': GradientBoostingClassifier(random_state=random_state),
            'logistic_regression': LogisticRegression(
                penalty="l2", solver="lbfgs", max_iter=3000, multi_class='auto', random_state=random_state
            ),
            'svm': SVC(random_state=random_state, probability=True, gamma="scale"),
            'neural_network': MLPClassifier(random_state=random_state, max_iter=600)
        }
        self.scaler = None

    # -------------------------------
    # Data Loading & Preprocessing
    # -------------------------------
    def load_and_preprocess_data(self, filepath: str):
        print("Loading data...")
        df = pd.read_csv(filepath)
        print(f"Data shape: {df.shape}")

        # Remove columns that are not features or are redundant
        columns_to_remove = [
            'Educational_Other', 'Employment_Other', 'Cluster_Label',
            'Total_IA_Score', 'Predicted_Cluster'
        ]
        df_clean = df.drop(columns=[c for c in columns_to_remove if c in df.columns], errors="ignore")

        # Define feature types
        categorical_features = ['Gender', 'Age_Group', 'Marital_Status',
                                'Educational_Level', 'Employment_Status', 'Daily_App_Usage']

        # Ordinal encodings
        age_order = ['Less than 18', '18-24 years old', '25-34 years old', '35-44 years old',
                     '45-54 years old', '55-64 years old', '65+ years old']

        education_order = ['High school', 'Diploma', "Bachelor's degree", "Master's degree",
                           "Doctorate degree", 'Other (please specify)']

        app_usage_order = ['Less than 1 hour', '1-2 hours', '2-3 hours', '3-4 hours',
                           '4-5 hours', 'More than 5 hours']

        # Apply ordinal encoding if present
        if 'Age_Group' in df_clean.columns:
            age_encoder = OrdinalEncoder(categories=[age_order], handle_unknown='use_encoded_value', unknown_value=-1)
            df_clean['Age_Group_Ordinal'] = age_encoder.fit_transform(df_clean[['Age_Group']]).ravel()

        if 'Educational_Level' in df_clean.columns:
            edu_encoder = OrdinalEncoder(categories=[education_order], handle_unknown='use_encoded_value', unknown_value=-1)
            df_clean['Educational_Level_Ordinal'] = edu_encoder.fit_transform(df_clean[['Educational_Level']]).ravel()

        if 'Daily_App_Usage' in df_clean.columns:
            usage_encoder = OrdinalEncoder(categories=[app_usage_order], handle_unknown='use_encoded_value', unknown_value=-1)
            df_clean['Daily_App_Usage_Ordinal'] = usage_encoder.fit_transform(df_clean[['Daily_App_Usage']]).ravel()

        # Nominal categorical (to one-hot)
        categorical_features_nominal = ['Gender', 'Marital_Status', 'Employment_Status']

        # Numerical features (questionnaire style prefixes)
        numerical_features = [col for col in df_clean.columns
                              if col.startswith(('IA', 'CON', 'INT', 'AVL', 'AT', 'NR', 'DL'))
                              and col not in categorical_features]

        # Add ordinal encodings
        ordinal_features = ['Age_Group_Ordinal', 'Educational_Level_Ordinal', 'Daily_App_Usage_Ordinal']
        numerical_features.extend([c for c in ordinal_features if c in df_clean.columns])

        print(f"Categorical features (nominal): {categorical_features_nominal}")
        if 'Cluster_Number' in df_clean.columns:
            print("Target distribution:")
            print(df_clean['Cluster_Number'].value_counts().sort_index())
        else:
            raise ValueError("Expected target column 'Cluster_Number' not found.")

        return df_clean, categorical_features_nominal, numerical_features

    def prepare_features(self, df: pd.DataFrame, categorical_features_nominal, numerical_features):
        X_num = df[numerical_features].copy()

        # one-hot on nominal categoricals
        X_cat = pd.get_dummies(df[categorical_features_nominal], drop_first=True)
        X = pd.concat([X_num, X_cat], axis=1)
        y = df['Cluster_Number']

        print(f"Final feature matrix shape: {X.shape}")
        return X, y

    # -------------------------------
    # Training & Evaluation
    # -------------------------------
    def _compute_metrics(self, y_true, y_pred, y_proba=None):
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_acc': balanced_accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        if y_proba is not None:
            # log loss and AUC for multiclass
            try:
                metrics['log_loss'] = log_loss(y_true, y_proba)
            except Exception:
                metrics['log_loss'] = np.nan
            try:
                metrics['roc_auc_ovr_weighted'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
            except Exception:
                metrics['roc_auc_ovr_weighted'] = np.nan
            try:
                metrics['roc_auc_ovo_weighted'] = roc_auc_score(y_true, y_proba, multi_class='ovo', average='weighted')
            except Exception:
                metrics['roc_auc_ovo_weighted'] = np.nan
        else:
            metrics['log_loss'] = np.nan
            metrics['roc_auc_ovr_weighted'] = np.nan
            metrics['roc_auc_ovo_weighted'] = np.nan
        return metrics

    def train_and_evaluate_models(self, X, y, test_size=0.2, cv_splits=5):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # scale numeric features (all features are numeric after get_dummies)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        results = {}
        rows_for_df = []

        print("\nTraining and evaluating models...")
        print("=" * 60)

        scoring = {
            'accuracy': 'accuracy',
            'balanced_acc': 'balanced_accuracy',
            'f1_macro': 'f1_macro',
            'f1_weighted': 'f1_weighted'
        }
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)

        for name, model in self.models.items():
            print(f"\n▶ Training {name}")

            # timing: fit
            t0 = time.time()
            model.fit(X_train_scaled, y_train)
            train_time_s = time.time() - t0

            # timing: predict
            t1 = time.time()
            y_pred = model.predict(X_test_scaled)
            predict_time_s = time.time() - t1
            predict_latency_ms_per_sample = (predict_time_s / len(y_test)) * 1000.0

            # probabilities (for AUC/log loss)
            y_proba = model.predict_proba(X_test_scaled) if hasattr(model, "predict_proba") else None

            # compute metrics on holdout
            holdout_metrics = self._compute_metrics(y_test, y_pred, y_proba=y_proba)

            # cross-validated metrics on train set
            cv_res = cross_validate(model, X_train_scaled, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
            cv_summary = {f"cv_{k}_mean": float(np.mean(v)) for k, v in cv_res.items() if k.startswith("test_") for _ in [None]}
            # Map keys to nicer names
            cv_summary = {
                'cv_accuracy_mean': float(np.mean(cv_res['test_accuracy'])),
                'cv_balanced_acc_mean': float(np.mean(cv_res['test_balanced_acc'])),
                'cv_f1_macro_mean': float(np.mean(cv_res['test_f1_macro'])),
                'cv_f1_weighted_mean': float(np.mean(cv_res['test_f1_weighted'])),
            }

            # Save per-model results
            results[name] = {
                'model': model,
                'train_time_s': train_time_s,
                'predict_latency_ms_per_sample': predict_latency_ms_per_sample,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_proba': y_proba,
                'holdout_metrics': holdout_metrics,
                'cv_summary': cv_summary,
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }

            # row for summary table
            row = {
                'model': name,
                'train_time_s': train_time_s,
                'predict_latency_ms_per_sample': predict_latency_ms_per_sample,
                'test_accuracy': holdout_metrics['accuracy'],
                'test_balanced_acc': holdout_metrics['balanced_acc'],
                'test_precision_macro': holdout_metrics['precision_macro'],
                'test_precision_weighted': holdout_metrics['precision_weighted'],
                'test_recall_macro': holdout_metrics['recall_macro'],
                'test_recall_weighted': holdout_metrics['recall_weighted'],
                'test_f1_macro': holdout_metrics['f1_macro'],
                'test_f1_weighted': holdout_metrics['f1_weighted'],
                'test_log_loss': holdout_metrics['log_loss'],
                'test_roc_auc_ovr_weighted': holdout_metrics['roc_auc_ovr_weighted'],
                'test_roc_auc_ovo_weighted': holdout_metrics['roc_auc_ovo_weighted'],
                **cv_summary
            }
            rows_for_df.append(row)

            # brief printout
            print(f"  train_time: {train_time_s:.3f}s | latency: {predict_latency_ms_per_sample:.3f} ms/sample")
            print(f"  test_acc: {row['test_accuracy']:.4f} | test_f1_w: {row['test_f1_weighted']:.4f} | test_recall_w: {row['test_recall_weighted']:.4f}")
            if not np.isnan(row['test_roc_auc_ovr_weighted']):
                print(f"  AUC (OvR,w): {row['test_roc_auc_ovr_weighted']:.4f}")

        # Summary DataFrame
        summary_df = pd.DataFrame(rows_for_df).sort_values(by=['test_f1_weighted', 'test_accuracy'], ascending=False).reset_index(drop=True)
        print("\n=== Final Model Comparison (sorted by test_f1_weighted then test_accuracy) ===")
        print(summary_df[['model','test_accuracy','test_f1_weighted','test_balanced_acc','test_roc_auc_ovr_weighted','train_time_s','predict_latency_ms_per_sample','cv_accuracy_mean','cv_f1_weighted_mean']])

        return results, summary_df, (X_test, y_test)

    # -------------------------------
    # Hyperparameter Tuning (optional)
    # -------------------------------
    def hyperparameter_tuning(self, X, y, model_name='logistic_regression'):
        print(f"\nHyperparameter tuning: {model_name}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=self.random_state, stratify=y)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if model_name == 'random_forest':
            param_grid = {
                'n_estimators': [200, 300, 500],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=self.random_state)
        elif model_name == 'gradient_boosting':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
            model = GradientBoostingClassifier(random_state=self.random_state)
        elif model_name == 'svm':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.01, 0.1],
                'kernel': ['rbf']
            }
            model = SVC(random_state=self.random_state, probability=True)
        else:
            # logistic_regression default
            param_grid = [
                {
                    'solver': ['lbfgs'],
                    'multi_class': ['auto', 'multinomial'],
                    'penalty': ['l2'],
                    'C': [0.05, 0.1, 0.2, 0.3162, 0.5, 1, 1.5, 2, 3],
                    'max_iter': [3000],
                    'class_weight': [None, 'balanced']
                },
                {
                    'solver': ['saga'],
                    'multi_class': ['auto', 'multinomial'],
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'l1_ratio': [0, 0.25, 0.5, 0.75, 1],
                    'C': [0.05, 0.1, 0.2, 0.3162, 0.5, 1, 1.5, 2, 3],
                    'max_iter': [3000],
                    'class_weight': [None, 'balanced']
                }
            ]
            model = LogisticRegression(random_state=self.random_state)

        grid = GridSearchCV(
            model, param_grid, cv=5,
            scoring='f1_weighted', n_jobs=-1, verbose=0
        )
        grid.fit(X_train_scaled, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test_scaled)
        y_proba = best_model.predict_proba(X_test_scaled) if hasattr(best_model, "predict_proba") else None

        holdout = self._compute_metrics(y_test, y_pred, y_proba)
        print("Best params:", grid.best_params_)
        print("Best CV f1_weighted:", f"{grid.best_score_:.4f}")
        print("Holdout weighted F1:", f"{holdout['f1_weighted']:.4f}", "| Accuracy:", f"{holdout['accuracy']:.4f}")

        return best_model, scaler, grid.best_params_

    # -------------------------------
    # Persistence
    # -------------------------------
    def save_model(self, model, scaler, filepath_prefix: str = "ia_cluster_predictor"):
        import joblib
        joblib.dump(model, f"{filepath_prefix}_model.pkl")
        joblib.dump(scaler, f"{filepath_prefix}_scaler.pkl")
        print(f"Saved: {filepath_prefix}_model.pkl and {filepath_prefix}_scaler.pkl")

# -------------------------------
# Script runner
# -------------------------------
if __name__ == "__main__":
    predictor = EnhancedIAClusterPredictor()

    # Adjust the CSV path if needed
    csv_path = "Information_Assurance_Clustered.csv"

    # Load & prep
    df, cat_nominal, num_feats = predictor.load_and_preprocess_data(csv_path)
    X, y = predictor.prepare_features(df, cat_nominal, num_feats)

    # Train & evaluate
    results, summary_df, (X_test, y_test) = predictor.train_and_evaluate_models(X, y, test_size=0.2, cv_splits=5)

    # Save the summary to CSV
    out_csv = "model_comparison_metrics.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"\nSaved summary table to: {out_csv}")

    # Optional: tune the best (by weighted F1) model name
    candidate = summary_df.loc[0, 'model']
    print(f"\nCandidate for tuning (best by test_f1_weighted): {candidate}")
    tuned_model, tuned_scaler, best_params = predictor.hyperparameter_tuning(X, y, model_name=candidate)

    # Persist tuned model
    predictor.save_model(tuned_model, tuned_scaler, filepath_prefix="ia_cluster_predictor")
