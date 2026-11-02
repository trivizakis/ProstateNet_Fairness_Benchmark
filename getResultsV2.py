#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUROC Analysis & Global Results Aggregator for ProstateNet Radiomics Fairness Benchmark

Reads from the structured results hierarchy:
    chkp/<experiment>/<classifier_family>/<variant>/

Generates:
1. AUROC_comparison.png inside each classifier folder (comparing Biased [Red], Fair: Reweighing [Green], BGMMOCE [Blue], etc.)
2. metrics_comparison.csv inside each classifier folder
3. benchmark_summary_[experiment].csv inside each experiment folder
4. Global benchmark_summary_global.csv aggregating ALL experiments, classifiers, and variants.

@author: Eleftherios Trivizakis & Google DeepMind Antigravity
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Style dictionary for comparable variants (Triplets)
VARIANT_STYLES = {
    "biased":                    {"color": "#d62728", "style": "-",  "label": "Biased (Baseline)"},
    "BGMMOCE":                   {"color": "#1f77b4", "style": "--", "label": "BGMMOCE"},
    "reweighing":                {"color": "#2ca02c", "style": "-",  "label": "Fair: Reweighing"},
    "disparate_impact_remover":  {"color": "#9467bd", "style": "-.", "label": "Fair: Disparate Impact Remover"},
    "reductions":                {"color": "#ff7f0e", "style": ":",  "label": "Fair: Reduction (EO/DP)"},
    "inprocessing":              {"color": "#17becf", "style": "-",  "label": "Fair: Prejudice Remover"},
    "postprocessing":            {"color": "#bcbd22", "style": "--", "label": "Fair: Post-Processing"}
}

SPECIAL_NAMES = {
    "svm_rbf": "SVM (RBF)",
    "svm_linear": "SVM (Linear)",
    "svm_poly": "SVM (Poly)",
    "svm_sigmoid": "SVM (Sigmoid)",
    "svm": "SVM",
    "mlp16": "MLP (16 Neurons)",
    "mlp32": "MLP (32 Neurons)",
    "mlp64": "MLP (64 Neurons)",
    "mlp128": "MLP (128 Neurons)",
    "mlp": "MLP",
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
    "gradient_boosting": "Gradient Boosting",
    "adaboost": "AdaBoost",
    "decision_tree": "Decision Tree",
    "naive_bayes": "Naive Bayes",
    "ridge": "Ridge",
    "sgd": "SGD"
}


def clean_display_name(name):
    if name.lower() in SPECIAL_NAMES:
        return SPECIAL_NAMES[name.lower()]
    return str(name).replace("_", " ").title()


def clean_experiment_name(exp_key):
    s = str(exp_key).replace(".set", "").replace("_training", "")
    parts = s.split("_")
    # e.g. lesion_uc2_manufacturer
    return " | ".join([p.capitalize() for p in parts])


def process_classifier_folder(exp_name, clf_name, clf_dir, global_rows):
    """
    Processes all variants inside a specific classifier folder:
    chkp/<experiment>/<classifier>/<variant>/
    Generates AUROC_comparison.png and metrics_comparison.csv.
    """
    variant_subdirs = [
        d for d in os.listdir(clf_dir)
        if os.path.isdir(os.path.join(clf_dir, d)) and not d.startswith(".")
    ]

    if not variant_subdirs:
        return

    clf_metrics = []
    variant_best_curves = {}

    variant_order = [
        "biased", "BGMMOCE", "reweighing", "disparate_impact_remover",
        "reductions", "inprocessing", "postprocessing"
    ]

    for variant in sorted(variant_subdirs):
        var_path = os.path.join(clf_dir, variant)
        json_files = glob.glob(os.path.join(var_path, "*_metrics.json"))
        if not json_files:
            continue

        for jf in sorted(json_files):
            model_key = os.path.basename(jf).replace("_metrics.json", "")
            try:
                with open(jf, "r") as f:
                    mdata = json.load(f)
            except Exception as e:
                print(f"    [!] Error reading {jf}: {e}")
                continue

            folds = mdata.get("folds", [])
            auc = mdata.get("mean_auc", 0.0)
            c_index = mdata.get("c_index", auc)
            brier = mdata.get("brier_score", np.nan)
            ipa = mdata.get("ipa", np.nan)
            ece = mdata.get("ece", np.nan)
            acc = mdata.get("accuracy", 0.0)
            f1 = mdata.get("f1", 0.0)
            prec = mdata.get("precision", 0.0)
            rec = mdata.get("recall", 0.0)
            fairness = mdata.get("fairness", {})
            spd = fairness.get("statistical_parity_difference", np.nan)
            di = fairness.get("disparate_impact", np.nan)
            eod = fairness.get("equal_opportunity_difference", np.nan)
            aod = fairness.get("average_odds_difference", np.nan)
            c_disp = mdata.get("survival", {}).get("c_index_disparity", np.nan)

            training_split = "BGMMOCE" if "bgmmoce" in variant.lower() else "biased"

            row = {
                "Experiment": exp_name,
                "Classifier_Family": clf_name,
                "Model_Name": model_key,
                "Variant": variant,
                "Training_Split": training_split,
                "AUC": round(auc, 4) if not np.isnan(auc) else None,
                "C_Index": round(c_index, 4) if not np.isnan(c_index) else None,
                "Brier_Score": round(brier, 4) if not np.isnan(brier) else None,
                "IPA": round(ipa, 4) if not np.isnan(ipa) else None,
                "ECE": round(ece, 4) if not np.isnan(ece) else None,
                "Accuracy": round(acc, 4) if not np.isnan(acc) else None,
                "F1": round(f1, 4) if not np.isnan(f1) else None,
                "Precision": round(prec, 4) if not np.isnan(prec) else None,
                "Recall": round(rec, 4) if not np.isnan(rec) else None,
                "Statistical_Parity_Diff": round(spd, 4) if not np.isnan(spd) else None,
                "Disparate_Impact": round(di, 4) if not np.isnan(di) else None,
                "Equal_Opportunity_Diff": round(eod, 4) if not np.isnan(eod) else None,
                "Average_Odds_Diff": round(aod, 4) if not np.isnan(aod) else None,
                "C_Index_Disparity": round(c_disp, 4) if not np.isnan(c_disp) else None
            }

            clf_metrics.append(row)
            global_rows.append(row)

            # Record candidate ROC curve
            if folds and "fpr" in folds[0] and "tpr" in folds[0]:
                curr_best = variant_best_curves.get(variant)
                if curr_best is None or auc > curr_best["auc"]:
                    variant_best_curves[variant] = {
                        "model_key": model_key,
                        "auc": auc,
                        "fpr": np.array(folds[0]["fpr"]),
                        "tpr": np.array(folds[0]["tpr"])
                    }

    # Generate Comparative AUROC Figure
    if variant_best_curves:
        plt.figure(figsize=(8, 7))
        sorted_variants = sorted(
            variant_best_curves.keys(),
            key=lambda v: variant_order.index(v) if v in variant_order else 99
        )

        for variant in sorted_variants:
            cdata = variant_best_curves[variant]
            style_info = VARIANT_STYLES.get(variant, {"color": None, "style": "-", "label": clean_display_name(variant)})
            base_label = style_info.get("label", variant)

            if cdata["model_key"] != variant and cdata["model_key"] not in ["random_forest", "mlp16", "mlp32", "mlp64", "mlp", "adaboost", "gradient_boosting"]:
                label = f"{base_label} [{cdata['model_key']}] (AUC={cdata['auc']:.2f})"
            else:
                label = f"{base_label} (AUC={cdata['auc']:.2f})"

            lw = 2.5 if variant in ["biased", "BGMMOCE"] else 2.0
            plt.plot(
                cdata["fpr"], cdata["tpr"], label=label,
                color=style_info.get("color"),
                linestyle=style_info.get("style", "-"),
                linewidth=lw
            )

        readable_clf = clean_display_name(clf_name)
        readable_exp = clean_experiment_name(exp_name)
        plt.plot([0, 1], [0, 1], 'k:', alpha=0.4, label='Random Chance')
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11)
        plt.title(f'AUROC Comparison: {readable_clf}\n{readable_exp}', fontsize=12)
        plt.legend(loc='lower right', fontsize=9)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()

        plot_path = os.path.join(clf_dir, "AUROC_comparison.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"  -> Generated AUROC: {plot_path}")

    if clf_metrics:
        metrics_df = pd.DataFrame(clf_metrics)
        metrics_csv = os.path.join(clf_dir, "metrics_comparison.csv")
        metrics_df.to_csv(metrics_csv, index=False)
        print(f"  -> Generated Metrics CSV: {metrics_csv}")


def aggregate_and_generate_reports(base_dir="chkp"):
    """
    Main aggregator: crawls <base_dir>/<experiment>/<classifier>/<variant>/
    Generates comparison plots and writes split benchmark_summary_<experiment>.csv files and global summary.
    """
    print("=" * 80)
    print("PROSTATENET RESULTS AGGREGATION & AUROC COMPARISONS")
    print(f"Target Directory: {base_dir}")
    print("=" * 80)

    global_rows = []

    # 1. Discover structured experiment folders in base_dir
    found_experiments = []
    for item in sorted(os.listdir(base_dir)):
        exp_path = os.path.join(base_dir, item)
        if not os.path.isdir(exp_path) or item in ["best_performing_models"] or item.startswith("."):
            continue

        # Check if contains classifier subdirectories
        clf_dirs = [
            d for d in os.listdir(exp_path)
            if os.path.isdir(os.path.join(exp_path, d)) and not d.startswith(".")
        ]
        if clf_dirs:
            found_experiments.append(item)

    if not found_experiments:
        print(f"[!] No experiment subdirectories with classifier folders found under: {base_dir}")
        return

    print(f"Found {len(found_experiments)} experiment folders: {found_experiments}\n")

    # 2. Iterate each experiment folder and its classifier folders
    for exp_name in found_experiments:
        exp_path = os.path.join(base_dir, exp_name)
        print(f">>> Processing Experiment: {exp_name.upper()}")

        clf_dirs = [
            d for d in os.listdir(exp_path)
            if os.path.isdir(os.path.join(exp_path, d)) and not d.startswith(".")
        ]

        exp_rows_before = len(global_rows)
        for clf_name in sorted(clf_dirs):
            clf_dir = os.path.join(exp_path, clf_name)
            process_classifier_folder(exp_name, clf_name, clf_dir, global_rows)

        # Write experiment-specific summary CSV
        exp_rows = global_rows[exp_rows_before:]
        if exp_rows:
            exp_df = pd.DataFrame(exp_rows)
            exp_summary_csv = os.path.join(base_dir, f"benchmark_summary_{exp_name}.csv")
            exp_df.to_csv(exp_summary_csv, index=False)
            print(f">>> Saved Experiment Summary: {exp_summary_csv}\n")

    # 3. Write comprehensive global summary CSV
    if global_rows:
        global_df = pd.DataFrame(global_rows)
        global_csv = os.path.join(base_dir, "benchmark_summary_global.csv")
        global_df.to_csv(global_csv, index=False)
        print("=" * 80)
        print(f"GLOBAL BENCHMARK AGGREGATION COMPLETE ({len(global_rows)} total model evaluations)")
        print(f"Saved Global Summary: {global_csv}")
        print("=" * 80)
        top10 = global_df.sort_values(by="AUC", ascending=False).head(10)
        print(top10[["Experiment", "Classifier_Family", "Variant", "Training_Split", "AUC", "Accuracy", "F1", "Statistical_Parity_Diff"]].to_string(index=False))


def find_target_dir(base_chkp="chkp", exp_name=None):
    if exp_name:
        path = os.path.join(base_chkp, exp_name)
        if os.path.exists(path):
            return path
        return exp_name

    exp_folders = sorted(glob.glob(os.path.join(base_chkp, "exp_*")))
    if exp_folders:
        return exp_folders[-1]
    return base_chkp


def main():
    parser = argparse.ArgumentParser(description="ProstateNet Global Results Aggregator & AUROC Plotter")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Specific experiment folder name under chkp/, e.g. exp_19.10_13h00m")
    parser.add_argument("--base-dir", type=str, default=None,
                        help="Path to experiment folder containing experiment subdirectories")
    args = parser.parse_args()

    target_dir = args.base_dir if args.base_dir else find_target_dir("chkp", args.exp_name)
    aggregate_and_generate_reports(target_dir)


if __name__ == "__main__":
    main()
