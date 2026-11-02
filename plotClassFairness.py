#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class Fairness & Disparity Visualizer for ProstateNet Single-Source Radiomics Benchmark

Visualizes per experiment:
1. Fairness Gap Reduction (Accuracy Disparity Gap between subgroups)
2. Selection Rate Disparity by subgroup
3. Error Rate by subgroup (Misclassification disparity)

Color Palette (Triplets):
- Biased (Baseline): Red (#d62728)
- Fair (Reweighing): Green (#2ca02c)
- BGMMOCE (Mitigated): Blue (#1f77b4)

Generates two sets of figures:
A. Standard figures across all classifier families:
   - fairness_gap_reduction_[experiment].png
   - fairness_selection_rate_disparity_[experiment].png
   - fairness_error_rate_by_subgroup_[experiment].png
   Saved in: chkp/exp_[date]_[time]/ (and inside each experiment folder)

B. Best-performing models figures (only triplets with at least one model > 0.70 AUC):
   - fairness_gap_reduction_[experiment].png
   - fairness_selection_rate_disparity_[experiment].png
   - fairness_error_rate_by_subgroup_[experiment].png
   Saved in: chkp/exp_[date]_[time]/best_performing_models/ (and inside best_performing_models/[experiment]/)

@author: Eleftherios Trivizakis & Google DeepMind Antigravity
"""

import os
import glob
import json
import argparse
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score

# Canonical Triplet Color Palette
COLOR_PALETTE = {
    "Biased": "#d62728",   # Red
    "Fair": "#2ca02c",     # Green (Reweighing)
    "BGMMOCE": "#1f77b4"   # Blue
}

CLF_DISPLAY_NAMES = {
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
    "svm_rbf": "SVM (RBF)",
    "svm_linear": "SVM (Linear)",
    "svm_poly": "SVM (Poly)",
    "svm_sigmoid": "SVM (Sigmoid)",
    "svm": "SVM",
    "gradient_boosting": "Gradient Boosting",
    "adaboost": "AdaBoost",
    "mlp16": "MLP (16)",
    "mlp32": "MLP (32)",
    "mlp64": "MLP (64)",
    "mlp128": "MLP (128)",
    "mlp": "MLP",
    "decision_tree": "Decision Tree",
    "naive_bayes": "Naive Bayes",
    "ridge": "Ridge",
    "sgd": "SGD"
}


def clean_display_name(clf_key):
    return CLF_DISPLAY_NAMES.get(clf_key, str(clf_key).replace("_", " ").title())


def get_model_category(clf_name):
    """Maps raw classifier folder names to visualization categories."""
    name = clf_name.lower()
    if "bgmmoce" in name:
        return "BGMMOCE"
    elif "reweigh" in name or "fair" in name:
        return "Fair"
    else:
        return "Biased"


def filter_best_performing_triplets(exp_data, auc_threshold=0.70):
    """
    Filters exp_data to only include classifier families (triplets) where
    at least one model variant (Biased, Fair, BGMMOCE) has AUC > auc_threshold.
    """
    filtered = {}
    for clf_fam, variants in exp_data.items():
        aucs = [v.get("auc") for v in variants.values() if v.get("auc") is not None]
        if aucs and max(aucs) > auc_threshold:
            filtered[clf_fam] = variants
    return filtered


def plot_fairness_gap_reduction_per_exp(exp_name, exp_data, output_dir, is_best_performing=False):
    """
    Generates fairness_gap_reduction_<experiment>.png
    Shows Accuracy Disparity Gap (|max(Acc_g) - min(Acc_g)|) across classifiers for:
    - Biased (Red)
    - Fair (Green)
    - BGMMOCE (Blue)
    Clean title: 'Fairness Gap Reduction'
    """
    plot_rows = []
    for clf_fam, variants in exp_data.items():
        clf_label = clean_display_name(clf_fam)
        for cat in ["Biased", "Fair", "BGMMOCE"]:
            if cat not in variants:
                continue
            data = variants[cat]
            y_pred = np.array(data["preds"])
            y_true = np.array(data["y_true"])
            biases = np.array(data["biases"])

            subgroup_accs = []
            for g in np.unique(biases):
                mask = (biases == g)
                if np.sum(mask) > 0:
                    subgroup_accs.append(accuracy_score(y_true[mask], y_pred[mask]))

            if len(subgroup_accs) >= 2:
                gap = abs(max(subgroup_accs) - min(subgroup_accs))
                plot_rows.append({
                    "Classifier": clf_label,
                    "Model Type": cat,
                    "Accuracy Gap": gap
                })

    if not plot_rows:
        print(f"  [!] No disparity gap data available for experiment: {exp_name}")
        return

    df = pd.DataFrame(plot_rows)
    n_clfs = len(df["Classifier"].unique())
    fig_width = max(7.0, min(14.0, n_clfs * 1.5 + 2.0))

    plt.figure(figsize=(fig_width, 5.5))
    sns.barplot(
        data=df,
        x="Classifier", y="Accuracy Gap", hue="Model Type",
        hue_order=["Biased", "Fair", "BGMMOCE"],
        palette=COLOR_PALETTE, edgecolor="black", alpha=0.9
    )
    plt.title("Fairness Gap Reduction", fontsize=14, fontweight='bold', pad=14)
    plt.ylabel("Accuracy Gap", fontsize=11)
    plt.xlabel("Classifier Family", fontsize=11)
    plt.xticks(rotation=20 if n_clfs > 4 else 0, ha='right' if n_clfs > 4 else 'center', fontsize=10)
    plt.ylim(0, max(0.4, df["Accuracy Gap"].max() * 1.15))
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title="Model Type", frameon=True)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"fairness_gap_reduction_{exp_name}.png")
    plt.savefig(save_path, dpi=300)

    # Also save inside experiment subfolder if it exists
    exp_subpath = os.path.join(output_dir, exp_name, f"fairness_gap_reduction_{exp_name}.png")
    if os.path.exists(os.path.join(output_dir, exp_name)):
        plt.savefig(exp_subpath, dpi=300)
    plt.close()
    print(f"  -> Saved: {save_path}")


def plot_selection_rate_disparity_per_exp(exp_name, exp_data, output_dir, is_best_performing=False):
    """
    Generates fairness_selection_rate_disparity_<experiment>.png
    Shows Selection Rate across subgroups for each classifier.
    Clean title: 'Fairness Selection Rate'
    """
    plot_rows = []
    for clf_fam, variants in exp_data.items():
        clf_label = clean_display_name(clf_fam)
        for cat in ["Biased", "Fair", "BGMMOCE"]:
            if cat not in variants:
                continue
            data = variants[cat]
            y_pred = np.array(data["preds"])
            biases = np.array(data["biases"])

            for g in np.unique(biases):
                mask = (biases == g)
                if np.sum(mask) > 0:
                    sel_rate = np.mean(y_pred[mask])
                    plot_rows.append({
                        "Classifier": clf_label,
                        "Model Type": cat,
                        "Subgroup": f"Group {int(g)}",
                        "Selection Rate": sel_rate
                    })

    if not plot_rows:
        print(f"  [!] No selection rate data available for experiment: {exp_name}")
        return

    df = pd.DataFrame(plot_rows)
    n_clfs = len(df["Classifier"].unique())
    col_wrap = min(4, max(1, n_clfs))

    g = sns.FacetGrid(df, col="Classifier", col_wrap=col_wrap, height=3.5, aspect=1.1, sharey=True)
    g.map_dataframe(
        sns.pointplot,
        x="Subgroup", y="Selection Rate", hue="Model Type",
        hue_order=["Biased", "Fair", "BGMMOCE"],
        palette=COLOR_PALETTE,
        markers=["o", "s", "D"], linestyles=["-", "--", "-."],
        dodge=0.2
    )
    g.add_legend(title="Model Type")
    g.set_titles(col_template="{col_name}", size=11)
    g.set_ylabels("Selection Rate", fontsize=11)
    g.set_xlabels("Subgroup", fontsize=11)
    plt.subplots_adjust(top=0.88)
    g.fig.suptitle("Fairness Selection Rate", fontsize=14, fontweight='bold')

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"fairness_selection_rate_disparity_{exp_name}.png")
    plt.savefig(save_path, dpi=300)

    exp_subpath = os.path.join(output_dir, exp_name, f"fairness_selection_rate_disparity_{exp_name}.png")
    if os.path.exists(os.path.join(output_dir, exp_name)):
        plt.savefig(exp_subpath, dpi=300)
    plt.close()
    print(f"  -> Saved: {save_path}")


def plot_error_rate_by_subgroup_per_exp(exp_name, exp_data, output_dir, is_best_performing=False):
    """
    Generates fairness_error_rate_by_subgroup_<experiment>.png
    Shows Misclassification Error Rate across subgroups for each classifier.
    Clean title: 'Fairness Error Rate'
    """
    plot_rows = []
    for clf_fam, variants in exp_data.items():
        clf_label = clean_display_name(clf_fam)
        for cat in ["Biased", "Fair", "BGMMOCE"]:
            if cat not in variants:
                continue
            data = variants[cat]
            y_pred = np.array(data["preds"])
            y_true = np.array(data["y_true"])
            biases = np.array(data["biases"])

            for g in np.unique(biases):
                mask = (biases == g)
                if np.sum(mask) > 0:
                    acc = accuracy_score(y_true[mask], y_pred[mask])
                    plot_rows.append({
                        "Classifier": clf_label,
                        "Model Type": cat,
                        "Subgroup": f"Group {int(g)}",
                        "Error Rate": 1.0 - acc
                    })

    if not plot_rows:
        print(f"  [!] No error rate data available for experiment: {exp_name}")
        return

    df = pd.DataFrame(plot_rows)
    n_clfs = len(df["Classifier"].unique())
    col_wrap = min(4, max(1, n_clfs))

    g = sns.catplot(
        data=df,
        x="Subgroup", y="Error Rate", hue="Model Type", col="Classifier",
        hue_order=["Biased", "Fair", "BGMMOCE"],
        kind="bar", col_wrap=col_wrap, height=3.5, aspect=1.1,
        palette=COLOR_PALETTE, edgecolor="black", alpha=0.85, sharey=True
    )
    g.set_titles(col_template="{col_name}", size=11)
    g.set_ylabels("Error Rate", fontsize=11)
    g.set_xlabels("Subgroup", fontsize=11)
    plt.subplots_adjust(top=0.88)
    g.fig.suptitle("Fairness Error Rate", fontsize=14, fontweight='bold')

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"fairness_error_rate_by_subgroup_{exp_name}.png")
    plt.savefig(save_path, dpi=300)

    exp_subpath = os.path.join(output_dir, exp_name, f"fairness_error_rate_by_subgroup_{exp_name}.png")
    if os.path.exists(os.path.join(output_dir, exp_name)):
        plt.savefig(exp_subpath, dpi=300)
    plt.close()
    print(f"  -> Saved: {save_path}")


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
    parser = argparse.ArgumentParser(description="ProstateNet Disparity & Fairness Visualization")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Specific experiment folder under chkp/, e.g. exp_19.10_13h00m")
    parser.add_argument("--base-dir", type=str, default=None,
                        help="Base directory containing experiment subfolders")
    parser.add_argument("--auc-threshold", type=float, default=0.70,
                        help="AUC threshold for best performing models (default: 0.70)")
    args = parser.parse_args()

    target_dir = args.base_dir if args.base_dir else find_target_dir("chkp", args.exp_name)
    print("=" * 80)
    print(f"PROSTATENET DISPARITY & FAIRNESS PLOT GENERATOR")
    print(f"Target Directory: {target_dir}")
    print(f"Best Performing Models AUC Threshold: > {args.auc_threshold:.2f}")
    print("=" * 80)

    target_clfs = [
        "random_forest", "logistic_regression",
        "svm_rbf", "svm_linear", "svm_poly", "svm_sigmoid",
        "gradient_boosting", "decision_tree", "adaboost", "naive_bayes",
        "mlp16", "mlp32", "mlp64", "mlp128",
        "ridge", "sgd"
    ]

    best_dir = os.path.join(target_dir, "best_performing_models")
    os.makedirs(best_dir, exist_ok=True)
    all_summary_records = []

    # Discover experiment subfolders inside target_dir
    candidate_subdirs = [
        d for d in os.listdir(target_dir)
        if os.path.isdir(os.path.join(target_dir, d)) and d != "best_performing_models" and not d.startswith(".")
    ]

    if not candidate_subdirs:
        print(f"[!] No experiment subdirectories found in: {target_dir}")
        return

    for exp_key in sorted(candidate_subdirs):
        exp_dir = os.path.join(target_dir, exp_key)

        # Check if this folder contains classifier family directories
        clf_subdirs = [
            d for d in os.listdir(exp_dir)
            if os.path.isdir(os.path.join(exp_dir, d)) and not d.startswith(".")
        ]
        if not clf_subdirs:
            continue

        print(f"\nProcessing Fairness & Disparity Visualizations for Experiment: {exp_key}")
        exp_data = {}  # {clf_fam: {"Biased": {...}, "Fair": {...}, "BGMMOCE": {...}}}

        sorted_clfs = [c for c in target_clfs if c in clf_subdirs] + [c for c in clf_subdirs if c not in target_clfs]

        for clf_fam in sorted_clfs:
            clf_fam_dir = os.path.join(exp_dir, clf_fam)
            if not os.path.isdir(clf_fam_dir):
                continue

            variant_map = {
                "Biased": os.path.join(clf_fam_dir, "biased"),
                "Fair": os.path.join(clf_fam_dir, "reweighing"),
                "BGMMOCE": os.path.join(clf_fam_dir, "BGMMOCE")
            }

            # If reweighing doesn't exist, check alternative fair mitigations
            if not os.path.exists(variant_map["Fair"]):
                for alt_fair in ["disparate_impact_remover", "postprocessing", "reductions", "inprocessing"]:
                    cand = os.path.join(clf_fam_dir, alt_fair)
                    if os.path.exists(cand):
                        variant_map["Fair"] = cand
                        break

            clf_records = {}

            for cat, v_dir in variant_map.items():
                if not os.path.isdir(v_dir):
                    continue

                preds_file = os.path.join(v_dir, "preds.npy")
                probs_file = os.path.join(v_dir, "probs.npy")
                y_file = os.path.join(v_dir, "y_test.npy")
                s_file = os.path.join(v_dir, "s_test.npy")

                if os.path.exists(preds_file) and os.path.exists(y_file) and os.path.exists(s_file):
                    preds = np.load(preds_file, allow_pickle=True)
                    y_true = np.load(y_file, allow_pickle=True)
                    biases_arr = np.load(s_file, allow_pickle=True)

                    auc_val = None
                    if os.path.exists(probs_file):
                        try:
                            probs = np.load(probs_file, allow_pickle=True)
                            if len(probs) == len(y_true):
                                if probs.ndim == 2 and probs.shape[1] == 2:
                                    auc_val = float(roc_auc_score(y_true, probs[:, 1]))
                                else:
                                    auc_val = float(roc_auc_score(y_true, probs))
                        except Exception:
                            auc_val = None

                    if auc_val is None:
                        m_files = glob.glob(os.path.join(v_dir, "*metrics.json"))
                        if m_files:
                            try:
                                with open(m_files[0], "r") as mf:
                                    m_data = json.load(mf)
                                    auc_val = float(m_data.get("mean_auc", 0.0))
                            except Exception:
                                auc_val = None

                    clf_records[cat] = {
                        "preds": preds,
                        "y_true": y_true,
                        "biases": biases_arr,
                        "auc": auc_val
                    }

            if clf_records:
                exp_data[clf_fam] = clf_records

        if not exp_data:
            print(f"  [!] No valid classifier predictions found for: {exp_key}")
            continue

        # 1. Generate Standard Figures (All Classifiers)
        print("  Generating Standard Fairness & Disparity Figures (All Classifiers)...")
        plot_fairness_gap_reduction_per_exp(exp_key, exp_data, target_dir, is_best_performing=False)
        plot_selection_rate_disparity_per_exp(exp_key, exp_data, target_dir, is_best_performing=False)
        plot_error_rate_by_subgroup_per_exp(exp_key, exp_data, target_dir, is_best_performing=False)

        # 2. Filter & Generate Best Performing Figures (AUC > threshold)
        best_data = filter_best_performing_triplets(exp_data, auc_threshold=args.auc_threshold)
        qual_count = len(best_data)
        total_count = len(exp_data)
        print(f"  Filtering Best-Performing Triplets (AUC > {args.auc_threshold:.2f}): {qual_count} of {total_count} qualified.")

        # Document qualification for summary CSV
        for clf_fam, variants in exp_data.items():
            b_auc = variants.get("Biased", {}).get("auc")
            f_auc = variants.get("Fair", {}).get("auc")
            bg_auc = variants.get("BGMMOCE", {}).get("auc")
            aucs = [x for x in [b_auc, f_auc, bg_auc] if x is not None]
            max_auc = max(aucs) if aucs else 0.0
            qualifies = max_auc > args.auc_threshold

            all_summary_records.append({
                "Experiment": exp_key,
                "Classifier_Family": clf_fam,
                "Classifier_Display": clean_display_name(clf_fam),
                "AUC_Biased": round(b_auc, 4) if b_auc is not None else None,
                "AUC_Fair": round(f_auc, 4) if f_auc is not None else None,
                "AUC_BGMMOCE": round(bg_auc, 4) if bg_auc is not None else None,
                "Max_Triplet_AUC": round(max_auc, 4),
                "Qualifies_Best_Performing": qualifies
            })

        if qual_count > 0:
            print(f"  Generating Best Performing Figures in: {best_dir}")
            plot_fairness_gap_reduction_per_exp(exp_key, best_data, best_dir, is_best_performing=True)
            plot_selection_rate_disparity_per_exp(exp_key, best_data, best_dir, is_best_performing=True)
            plot_error_rate_by_subgroup_per_exp(exp_key, best_data, best_dir, is_best_performing=True)
        else:
            print(f"  [Info] Skipping best-performing figures for {exp_key}: No classifier triplet achieved AUC > {args.auc_threshold:.2f}")

    # Save Best Performing Summary CSV
    if all_summary_records:
        summary_df = pd.DataFrame(all_summary_records)
        summary_csv_path = os.path.join(best_dir, "best_performing_models_summary.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print("\n" + "=" * 80)
        print(f"BEST PERFORMING MODELS AUDIT SUMMARY SAVED: {summary_csv_path}")
        print("=" * 80)
        top_qual = summary_df[summary_df["Qualifies_Best_Performing"]].sort_values(by="Max_Triplet_AUC", ascending=False)
        if not top_qual.empty:
            print(top_qual[["Experiment", "Classifier_Display", "AUC_Biased", "AUC_Fair", "AUC_BGMMOCE", "Max_Triplet_AUC"]].to_string(index=False))
        else:
            print("No classifier triplets met the qualification threshold across all experiments.")

    print("\nFairness & Disparity visualization generation complete.")


if __name__ == '__main__':
    main()
