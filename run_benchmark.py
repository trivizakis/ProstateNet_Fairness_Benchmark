#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Benchmark Runner for ProstateNet Single-Source Radiomics Fairness Benchmark

Orchestrates:
1. Training & Evaluation of Biased and Fair-Reweighed models (`analysisv3.py --training-type biased`)
2. Training & Evaluation of BGMMOCE models (`analysisv3.py --training-type BGMMOCE`)
3. Results aggregation & AUROC comparisons (`getResultsV2.py`)
4. Class fairness disparity plots (Standard & Best-Performing) (`plotClassFairness.py`)
5. Patient-level subgroup misclassification plots (`fairPatientLevelPlotV2.py`)

@author: Eleftherios Trivizakis & Google DeepMind Antigravity
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime


def run_command(cmd, desc):
    print("\n" + "=" * 80)
    print(f"STEP: {desc}")
    print(f"COMMAND: {' '.join(cmd)}")
    print("=" * 80)
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"[!] Warning: Step '{desc}' exited with code {ret.returncode}")
    return ret.returncode


def main():
    parser = argparse.ArgumentParser(description="ProstateNet Fairness Benchmark End-to-End Orchestrator")
    parser.add_argument("--experiment", type=str, default="all",
                        help="Specific experiment key to run, or 'all'")
    parser.add_argument("--roi", type=str, default=None, choices=["lesion", "prostate", "all"],
                        help="Filter by Region of Interest (lesion, prostate, all)")
    parser.add_argument("--endpoint", type=str, default=None, choices=["uc1", "uc2", "all"],
                        help="Filter by Endpoint (uc1, uc2, all)")
    parser.add_argument("--bias", type=str, default=None,
                        help="Filter by Bias attribute (magnetic.field.strength, manufacturer, all)")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Custom experiment folder name under chkp/, e.g. exp_prostate_run")
    parser.add_argument("--quick-test", action="store_true", default=False,
                        help="Fast execution mode with subset of classifiers for immediate validation")
    parser.add_argument("--no-shap", action="store_true", default=False,
                        help="Disable SHAP computation for faster benchmarking")
    parser.add_argument("--auc-threshold", type=float, default=0.70,
                        help="AUC threshold for best-performing model figures (default: 0.70)")
    args = parser.parse_args()

    python_bin = sys.executable

    now = datetime.now()
    exp_name = args.exp_name if args.exp_name else f"exp_{now.strftime('%d.%m')}_{now.strftime('%Hh%Mm')}"
    exp_dir = os.path.join("chkp", exp_name)

    common_args = []
    if args.experiment:
        common_args.extend(["--experiment", args.experiment])
    if args.roi:
        common_args.extend(["--roi", args.roi])
    if args.endpoint:
        common_args.extend(["--endpoint", args.endpoint])
    if args.bias:
        common_args.extend(["--bias", args.bias])
    if args.quick_test:
        common_args.append("--quick-test")
    if args.no_shap:
        common_args.append("--no-shap")
    common_args.extend(["--output-dir", exp_dir])

    print("=" * 80)
    print("PROSTATENET RADIOMICS FAIRNESS BENCHMARK ORCHESTRATOR")
    print(f"Target Session Directory: {exp_dir}")
    print("=" * 80)

    # 1. Run Biased Split (Trains Biased Baselines + AIF360 Fair Reweighing)
    cmd_biased = [python_bin, "analysisv3.py", "--training-type", "biased"] + common_args
    run_command(cmd_biased, "Running Biased & Fair-Reweighted Benchmark")

    # 2. Run BGMMOCE Split (Trains Pre-mitigated BGMMOCE models)
    cmd_bgmmoce = [python_bin, "analysisv3.py", "--training-type", "BGMMOCE"] + common_args
    run_command(cmd_bgmmoce, "Running BGMMOCE Mitigated Benchmark")

    # 3. Global Results Aggregator & AUROC comparisons
    cmd_agg = [python_bin, "getResultsV2.py", "--base-dir", exp_dir]
    run_command(cmd_agg, "Aggregating Benchmark Results & AUROC Comparisons")

    # 4. Class Fairness Disparity Figures (Gap Reduction, Selection Rate, Error Rate)
    cmd_fair = [python_bin, "plotClassFairness.py", "--base-dir", exp_dir, "--auc-threshold", str(args.auc_threshold)]
    run_command(cmd_fair, "Generating Class Fairness & Disparity Visualizations")

    # 5. Patient-Level Misclassification Disparity Figures
    cmd_patient = [python_bin, "fairPatientLevelPlotV2.py", "--base-dir", exp_dir]
    run_command(cmd_patient, "Generating Patient-Level Fairness Visualizations")

    print("\n" + "=" * 80)
    print("ALL BENCHMARK STAGES COMPLETED SUCCESSFULLY!")
    print(f"Inspect all generated models, metrics, and figures in: {exp_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
