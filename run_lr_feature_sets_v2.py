# -*- coding: utf-8 -*-
"""
Augmented reporting for per-feature-set Logistic Regression.
Now prints: Accuracy, F1-macro, Precision-macro, Recall-macro, AUC (OvR weighted),
and still saves the full table to CSV.

Usage (example):
  python run_lr_feature_sets_v2.py --features_csv Features_Excel.csv --data_csv Information_Assurance_Clustered.csv
"""
#!/usr/bin/env python3
import re
import time
import json
import argparse
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


def load_features_map(features_csv: str) -> Dict[str, List[str]]:
    df = pd.read_csv(features_csv)
    code_pattern = re.compile(r"^[A-Z]{2,3}\d{1,2}$")  # IA1, CON3, DL10, etc.
    feature_sets = {}
    for col in df.columns:
        if str(col).lower().startswith("unnamed"):
            continue
        col_values = []
        for v in df[col].astype(str).tolist():
            v_strip = str(v).strip()
            if v_strip in ("", "nan", "NaN"):
                continue
            if code_pattern.match(v_strip):
                col_values.append(v_strip)
        if col_values:
            feature_sets[col] = col_values
    return feature_sets


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("lr", LogisticRegression(
                C=0.31622776601683794,
                penalty="l2",
                solver="lbfgs",
                max_iter=2000,
                random_state=42,
            )),
        ]
    )


def prepare_dataset(
    data_csv: str,
    target_col: str = "Cluster_Number",
    rename_map: Dict[str, str] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(data_csv)
    if rename_map:
        valid_map = {k: v for k, v in rename_map.items() if k in df.columns}
        if valid_map:
            df = df.rename(columns=valid_map)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {data_csv}.")
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=1, how="all")
    y = df[target_col]
    X = df.drop(columns=[target_col])

    likert_map = {
        "Strongly Agree": 5, "Agree": 4, "Neutral": 3, "Disagree": 2, "Strongly Disagree": 1,
    }
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = X[c].replace(likert_map)
            X[c] = pd.to_numeric(X[c], errors="ignore")

    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = X[c].astype("category").cat.codes

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X, y


def evaluate_feature_set(
    X: pd.DataFrame,
    y: pd.Series,
    features: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict:
    features_present = [f for f in features if f in X.columns]
    missing = list(sorted(set(features) - set(features_present)))

    if not features_present:
        return {
            "used_features": 0,
            "missing_features": len(missing),
            "missing_list": missing,
            "acc": np.nan,
            "precision_macro": np.nan,
            "recall_macro": np.nan,
            "f1_macro": np.nan,
            "auc_ovr_weighted": np.nan,
            "auc_ovo_weighted": np.nan,
            "train_time_s": np.nan,
            "notes": "No requested features were found in dataset",
        }

    X_sub = X[features_present].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X_sub, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = build_model()

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    t1 = time.perf_counter()

    y_pred = model.predict(X_test)

    # probabilities for AUC
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            y_proba = None

    auc_ovr_w = np.nan
    auc_ovo_w = np.nan
    if y_proba is not None:
        try:
            auc_ovr_w = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted"))
        except Exception:
            pass
        try:
            auc_ovo_w = float(roc_auc_score(y_test, y_proba, multi_class="ovo", average="weighted"))
        except Exception:
            pass

    metrics = {
        "used_features": len(features_present),
        "missing_features": len(missing),
        "missing_list": missing,
        "acc": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "auc_ovr_weighted": auc_ovr_w,
        "auc_ovo_weighted": auc_ovo_w,
        "train_time_s": float(t1 - t0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "notes": "",
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate tuned LR per features column.")
    parser.add_argument("--features_csv", type=str, required=True,
                        help="Path to Features_Excel.csv (exported as CSV).")
    parser.add_argument("--data_csv", type=str, required=True,
                        help="Path to modeling dataset with target column.")
    parser.add_argument("--target_col", type=str, default="Cluster_Number",
                        help="Target/label column name.")
    parser.add_argument("--rename_json", type=str, default="",
                        help="Optional JSON mapping of raw column names to codes.")
    args = parser.parse_args()

    feature_sets = load_features_map(args.features_csv)

    rename_map = None
    if args.rename_json:
        with open(args.rename_json, "r", encoding="utf-8") as f:
            rename_map = json.load(f)

    X, y = prepare_dataset(args.data_csv, target_col=args.target_col, rename_map=rename_map)

    results = []
    for col_name, feats in feature_sets.items():
        metrics = evaluate_feature_set(X, y, feats)
        row = {"feature_column": col_name, **metrics}
        results.append(row)

        print("=" * 80)
        print(f"Feature Column: {col_name}")
        print(f"Used features: {metrics['used_features']}  |  Missing: {metrics['missing_features']}")
        if metrics['missing_features']:
            print(f"Missing list (first 15): {metrics['missing_list'][:15]}")

        # Updated reporting line:
        print(
            "Accuracy: {:.4f}  |  F1-macro: {:.4f}  |  Precision-macro: {:.4f}  |  Recall-macro: {:.4f}  |  AUC(OvR,w): {}  |  Train time: {:.4f}s"
            .format(
                metrics['acc'],
                metrics['f1_macro'],
                metrics['precision_macro'],
                metrics['recall_macro'],
                ("{:.4f}".format(metrics['auc_ovr_weighted']) if not np.isnan(metrics['auc_ovr_weighted']) else "NA"),
                metrics['train_time_s']
            )
        )
        if isinstance(metrics.get("confusion_matrix"), list):
            print("Confusion Matrix:")
            for row in metrics["confusion_matrix"]:
                print(row)

    out_path = "lr_feature_set_results_v2.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print("\nSaved results to:", out_path)


if __name__ == "__main__":
    main()
