#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patient-Level Misclassification Error Rate Visualizer for ProstateNet Single-Source Radiomics Benchmark

Analyzes and plots subgroup misclassification rates per patient subgroup for:
- Biased (Baseline): Red (#d62728)
- Fair (Reweighing): Green (#2ca02c)
- BGMMOCE (Mitigated): Blue (#1f77b4)

@author: Eleftherios Trivizakis & Google DeepMind Antigravity
"""

import os
import glob
import argparse
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

EXPERIMENTS = [
    "lesion_uc2_magnetic.field.strength",
    "lesion_uc2_manufacturer",
    "prostate_uc1_magnetic.field.strength",
    "prostate_uc1_manufacturer",
    "prostate_uc2_magnetic.field.strength",
    "prostate_uc2_manufacturer"
]

COLOR_PALETTE = {
    "Biased (Baseline)": "#d62728",
    "Fair (Reweighting)": "#2ca02c",
    "BGMMOCE": "#1f77b4"
}


def analyze_misclassifications(base_dir, experiments=None, target_classifier="random_forest"):
    if experiments is None:
        experiments = EXPERIMENTS

    print("=" * 80)
    print("PROSTATENET PATIENT-LEVEL FAIRNESS & MISCLASSIFICATION ANALYSIS")
    print(f"Base Directory: {base_dir}")
    print(f"Target Classifier Family: {target_classifier}")
    print("=" * 80)

    for exp_folder in experiments:
        exp_path = os.path.join(base_dir, exp_folder)
        if not os.path.exists(exp_path):
            # Check for alternative aliases e.g. mfs
            alias = exp_folder.replace("magnetic.field.strength", "mfs")
            alt_path = os.path.join(base_dir, alias)
            if os.path.exists(alt_path):
                exp_path = alt_path
            else:
                continue

        print(f"\nProcessing Experiment: {os.path.basename(exp_path)}")
        clf_dir = os.path.join(exp_path, target_classifier)
        if not os.path.exists(clf_dir):
            # Fallback to any available classifier directory
            subdirs = [d for d in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, d))]
            if not subdirs:
                print(f"  [!] No classifier directories found in {exp_path}")
                continue
            clf_dir = os.path.join(exp_path, subdirs[0])

        plot_data = []

        variant_checks = [
            ("Biased (Baseline)", "biased"),
            ("Fair (Reweighting)", "reweighing"),
            ("BGMMOCE", "BGMMOCE")
        ]

        for label, v_name in variant_checks:
            v_dir = os.path.join(clf_dir, v_name)
            if not os.path.exists(v_dir):
                continue

            preds_file = os.path.join(v_dir, "preds.npy")
            y_file = os.path.join(v_dir, "y_test.npy")
            s_file = os.path.join(v_dir, "s_test.npy")

            if os.path.exists(preds_file) and os.path.exists(y_file) and os.path.exists(s_file):
                preds = np.load(preds_file, allow_pickle=True)
                y_true = np.load(y_file, allow_pickle=True)
                s_arr = np.load(s_file, allow_pickle=True)

                errors = (preds != y_true).astype(int)

                temp_df = pd.DataFrame({
                    "Subgroup": [f"Group {int(g)}" for g in s_arr],
                    "Error": errors
                })

                mean_errs = temp_df.groupby("Subgroup")["Error"].mean().reset_index()
                for _, r in mean_errs.iterrows():
                    plot_data.append({
                        "Model": label,
                        "Subgroup": r["Subgroup"],
                        "Error_Rate": r["Error"]
                    })

        if plot_data:
            plot_df = pd.DataFrame(plot_data)
            plt.figure(figsize=(8, 5.5))
            sns.barplot(
                data=plot_df,
                x="Subgroup",
                y="Error_Rate",
                hue="Model",
                palette=COLOR_PALETTE,
                edgecolor="black",
                linewidth=1,
                alpha=0.88
            )

            exp_title = os.path.basename(exp_path).replace("_", " ").title()
            plt.title(f"Patient Misclassification Disparity by Subgroup\n{exp_title}", fontsize=13, fontweight='bold')
            plt.ylabel("Misclassification Rate (Lower is Better)", fontsize=11)
            plt.xlabel("Sensitive Subgroup", fontsize=11)
            plt.ylim(0, max(0.5, plot_df["Error_Rate"].max() * 1.25))
            plt.legend(title="Model Type", loc='upper right')
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()

            save_name = f"patient_misclassification_{os.path.basename(exp_path)}.png"
            save_path = os.path.join(exp_path, save_name)
            plt.savefig(save_path, dpi=300)
            plt.savefig(os.path.join(base_dir, save_name), dpi=300)
            plt.close()
            print(f"  -> Saved patient misclassification plot: {save_path}")
        else:
            print(f"  [!] No valid predictions found for {exp_folder}")


def main():
    parser = argparse.ArgumentParser(description="ProstateNet Patient-Level Misclassification Plotter")
    parser.add_argument("--base-dir", type=str, default="chkp",
                        help="Path to experiment root directory under chkp/")
    parser.add_argument("--classifier", type=str, default="random_forest",
                        help="Target classifier family for patient-level plots (default: random_forest)")
    args = parser.parse_args()

    analyze_misclassifications(args.base_dir, target_classifier=args.classifier)


if __name__ == "__main__":
    main()
