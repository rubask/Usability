# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 07:25:48 2025

@author: Ruba
"""

#!/usr/bin/env python3
import re
import time
import json
import math
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
)

warnings.filterwarnings("ignore")


def load_features_map(features_csv: str) -> Dict[str, List[str]]:
    """
    Read the features matrix (export from Features_Excel.xlsx as CSV).
    Each non-empty cell that looks like a code (e.g., IA1, CON3) is treated as a feature.
    Returns: { 'Cuckoo': ['IA1','IA3',...], 'Wolf': [...], ... }
    """
    df = pd.read_csv(features_csv)
    # Heuristics: ignore the first row if it's counts; keep code-like tokens only
    code_pattern = re.compile(r"^[A-Z]{2,3}\d{1,2}$")  # IA1, CON3, DL10, etc.
    feature_sets = {}

    for col in df.columns:
        # Skip likely index column names
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
    """
    Tuned LR (aligned with your tuned LR hyperparameters).
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("lr", LogisticRegression(
                C=0.31622776601683794,
                penalty="l2",
                solver="lbfgs",
                max_iter=2000,
                random_state=42,
                n_jobs=None
            )),
        ]
    )


def prepare_dataset(
    data_csv: str,
    target_col: str = "Cluster_Number",
    rename_map: Dict[str, str] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load dataset, optionally rename columns (to IA1/CON1/... codes), and return X, y.
    """
    df = pd.read_csv(data_csv)

    # Optional renaming to match code-style headers (if your CSV has long question texts)
    if rename_map:
        # Only rename keys that are present to avoid KeyErrors
        valid_map = {k: v for k, v in rename_map.items() if k in df.columns}
        if valid_map:
            df = df.rename(columns=valid_map)

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in {data_csv}.\n"
            "Please ensure your dataset includes the label/target with this name, or pass --target_col."
        )

    # Basic cleaning: drop fully empty columns, trim whitespace in column names
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=1, how="all")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Try to coerce Likert-like text to numeric (Strongly Agree -> 5 ... Strongly Disagree -> 1)
    likert_map = {
        "Strongly Agree": 5,
        "Agree": 4,
        "Neutral": 3,
        "Disagree": 2,
        "Strongly Disagree": 1,
    }
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = X[c].replace(likert_map)
            # Convert any remaining numeric-looking strings
            X[c] = pd.to_numeric(X[c], errors="ignore")

    # After replacements, coerce remaining non-numeric to categorical codes
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = X[c].astype("category").cat.codes

    # Fill remaining NaNs
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X, y


def evaluate_feature_set(
    X: pd.DataFrame,
    y: pd.Series,
    features: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict:
    """
    Train/test a tuned LR on the provided feature subset.
    Returns metrics + training time.
    """
    # Keep only features that exist
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

    metrics = {
        "used_features": len(features_present),
        "missing_features": len(missing),
        "missing_list": missing,
        "acc": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
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
                        help="Name of the target/label column (default: Cluster_Number).")
    parser.add_argument("--rename_json", type=str, default="",
                        help="Optional path to JSON mapping of raw column names to codes (IA1, CON1, ...).")
    args = parser.parse_args()

    # Load features mapping
    feature_sets = load_features_map(args.features_csv)

    # Optional rename map
    rename_map = None
    if args.rename_json:
        with open(args.rename_json, "r", encoding="utf-8") as f:
            rename_map = json.load(f)

    # Prepare dataset
    X, y = prepare_dataset(args.data_csv, target_col=args.target_col, rename_map=rename_map)

    results = []
    for col_name, feats in feature_sets.items():
        metrics = evaluate_feature_set(X, y, feats)
        row = {"feature_column": col_name, **metrics}
        results.append(row)

        # Print a brief per-set summary + confusion matrix to stdout
        print("=" * 80)
        print(f"Feature Column: {col_name}")
        print(f"Used features: {metrics['used_features']}  |  Missing: {metrics['missing_features']}")
        if metrics['missing_features']:
            print(f"Missing list (truncated to 15): {metrics['missing_list'][:15]}")
        print(f"Accuracy: {metrics['acc']:.4f}  |  F1-macro: {metrics['f1_macro']:.4f}  |  Train time: {metrics['train_time_s']:.4f}s")
        if isinstance(metrics.get("confusion_matrix"), list):
            print("Confusion Matrix:")
            for row in metrics["confusion_matrix"]:
                print(row)

    # Save table
    out_path = "lr_feature_set_results2.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print("\nSaved results to:", out_path)


if __name__ == "__main__":
    main()
    

