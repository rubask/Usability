
import argparse
import pandas as pd
import numpy as np

NUM_COLS = [
    "train_accuracy","test_accuracy","balanced_accuracy",
    "precision_macro","recall_macro","f1_macro",
    "auc_macro_ovr","log_loss","cohen_kappa","mcc",
    "cv_mean","cv_std","fit_time_s"
]

RENAME = {
    "model": "Model",
    "train_accuracy": "Train Acc",
    "test_accuracy": "Test Acc",
    "balanced_accuracy": "Balanced Acc",
    "precision_macro": "Precision (macro)",
    "recall_macro": "Recall (macro)",
    "f1_macro": "F1 (macro)",
    "auc_macro_ovr": "ROC-AUC (macro OvR)",
    "log_loss": "Log Loss",
    "cohen_kappa": "Cohen’s κ",
    "mcc": "MCC",
    "cv_mean": "CV Acc (mean)",
    "cv_std": "CV Acc (std)",
    "fit_time_s": "Fit Time (s)",
}

ORDER = [
    "Model","Test Acc","F1 (macro)","Recall (macro)","Precision (macro)",
    "ROC-AUC (macro OvR)","Balanced Acc","Cohen’s κ","MCC","Log Loss",
    "CV Acc (mean)","CV Acc (std)","Fit Time (s)"
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_html", default="/mnt/data/metrics_report.html")
    ap.add_argument("--out_csv", default="/mnt/data/metrics_report.csv")
    ap.add_argument("--out_tex", default="/mnt/data/metrics_table.tex")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)

    # Ensure columns exist and round nicely
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Reorder / rename
    df = df.rename(columns=RENAME)
    present = [c for c in ORDER if c in df.columns]
    df = df[present].copy()

    # Sort by F1 (macro) then Test Acc if present
    sort_cols = [c for c in ["F1 (macro)", "Test Acc"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=[False]*len(sort_cols))

    # Round metrics
    round_map = {c: 4 for c in df.columns if c != "Model"}
    df = df.round(round_map)

    # Save a clean CSV
    df.to_csv(args.out_csv, index=False)

    # Style for HTML
    sty = (df.style
            .set_caption("Model Performance Summary")
            .format({c: "{:.4f}" for c in df.columns if c != "Model"})
            .background_gradient(
                subset=[c for c in df.columns if c not in ["Model","Log Loss","Fit Time (s)"]],
                cmap="Greens"
            )
            .background_gradient(
                subset=["Log Loss"] if "Log Loss" in df.columns else [],
                cmap="Reds"
            )
            .bar(subset=["Fit Time (s)"] if "Fit Time (s)" in df.columns else [], align="mid")
            .set_table_styles([
                {"selector":"th","props":[("text-align","center"),("font-weight","600")]},
                {"selector":"caption","props":[("caption-side","top"),("font-size","1.1em"),("font-weight","600")]},
            ])
            .hide(axis="index")
          )

    html = sty.to_html()
    with open(args.out_html, "w", encoding="utf-8") as f:
        f.write(html)

    # LaTeX table for papers (booktabs style)
    try:
        tex = df.to_latex(index=False, escape=False, bold_rows=False, longtable=False)
        with open(args.out_tex, "w", encoding="utf-8") as f:
            f.write(tex)
    except Exception:
        pass

    print("Saved:")
    print(" - HTML:", args.out_html)
    print(" - CSV :", args.out_csv)
    print(" - TeX :", args.out_tex)

if __name__ == "__main__":
    main()
