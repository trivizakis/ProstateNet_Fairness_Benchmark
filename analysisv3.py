#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ProstateNet Fairness Benchmark: Prostate Cancer Detection & Aggressiveness Single-Source Radiomics Classification

Radiomics-only analysis for:
- Endpoints: UC1 (Prostate Cancer Detection) & UC2 (Prostate Cancer Aggressiveness)
- Regions of Interest: Lesion & Prostate
- Biases / Sensitive Attributes: Magnetic Field Strength (magnetic.field.strength) & Manufacturer (manufacturer)
- Datasets: Biased (observational), Fair Reweighing (AIF360 sample weights), and BGMMOCE (pre-mitigated)
- Evaluated on a common external testing set (dataset/testing_set/)

Integrates:
- 16 comprehensive Scikit-Learn classifier families supporting sample reweighting
- Single-source radiomics feature scaling (RobustScaler) and hierarchical selection (Variance -> ANOVA -> L1 Sparsity)
- Exact generalization of AIF360 Sample Reweighing for binary and multi-class sensitive attributes
- Disparity impact remover, in-processing, and post-processing mitigation options
- SHAP explainability for test inference with radiomics feature attribution
- Complete performance & fairness metric suite (AUC, C-Index, Brier, IPA, ECE, Accuracy, F1, Stat Parity, Disparate Impact, Eq Odds)

@author: Eleftherios Trivizakis & Google DeepMind Antigravity
"""

import os
import sys
import time
import glob
import json
import warnings
import argparse
from datetime import datetime
import pickle as pkl
import numpy as np
import pandas as pd
import scipy.stats as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Selection & Scalers
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel, VarianceThreshold

# Models (16 Comprehensive Scikit-Learn Classifiers supporting sample weights)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Metrics & Resampling
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    brier_score_loss,
    confusion_matrix
)
from imblearn.over_sampling import SMOTE
from lifelines.utils import concordance_index

# AIF360 Fairness & Mitigation
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
from aif360.algorithms.inprocessing import (
    PrejudiceRemover,
    ExponentiatedGradientReduction,
    GridSearchReduction
)
from aif360.algorithms.postprocessing import (
    RejectOptionClassification,
    EqOddsPostprocessing,
    CalibratedEqOddsPostprocessing
)

# SHAP Explainability
import shap

warnings.filterwarnings('ignore')


def shorten_feature_name(name):
    """
    Shortens radiomics feature names for SHAP plots and tables:
    Removes 'original_1_original_', '1_original_', 'rad_', etc.
    """
    s = str(name)
    if s.startswith("rad_"):
        s = s[4:]
    s = s.replace("original_1_original_", "")
    s = s.replace("1_original_", "")
    s = s.replace("original_", "")
    return s


def get_classifier_family_and_variant(name, training_type="biased"):
    """
    Maps a model name to its classifier family and comparable variant subfolder:
    Returns (classifier_family, variant_name)
    Hierarchy: <experiment> / <classifier_family> / <variant>
    """
    is_bgmmoce = (str(training_type).upper() == "BGMMOCE")

    base_family_map = {
        "random_forest": "random_forest",
        "logistic_l2": "logistic_regression",
        "logistic_l1": "logistic_regression",
        "logistic_elastic": "logistic_regression",
        "svm_rbf": "svm_rbf",
        "svm_linear": "svm_linear",
        "svm_poly": "svm_poly",
        "svm_sigmoid": "svm_sigmoid",
        "gradient_boosting": "gradient_boosting",
        "adaboost": "adaboost",
        "mlp16": "mlp16",
        "mlp32": "mlp32",
        "mlp64": "mlp64",
        "mlp128": "mlp128",
        "decision_tree": "decision_tree",
        "naive_bayes": "naive_bayes",
        "ridge": "ridge",
        "sgd": "sgd"
    }

    if not name.startswith("fair_"):
        family = base_family_map.get(name, name)
        variant = "BGMMOCE" if is_bgmmoce else "biased"
        return family, variant

    prefix = "BGMMOCE_" if is_bgmmoce else ""

    if "reweighted" in name:
        fam_key = name.replace("fair_reweighted_", "")
        fam_map = {
            "rf": "random_forest",
            "lr": "logistic_regression",
            "svm_rbf": "svm_rbf",
            "svm_linear": "svm_linear",
            "svm_poly": "svm_poly",
            "svm_sigmoid": "svm_sigmoid",
            "svm": "svm_rbf",
            "boost": "gradient_boosting",
            "ada": "adaboost",
            "dt": "decision_tree",
            "nb": "naive_bayes",
            "ridge": "ridge",
            "sgd": "sgd",
            "mlp16": "mlp16",
            "mlp32": "mlp32",
            "mlp64": "mlp64",
            "mlp128": "mlp128"
        }
        return fam_map.get(fam_key, fam_key), f"{prefix}reweighing"

    if "dir" in name:
        fam_key = name.replace("fair_dir_", "")
        fam_map = {
            "rf": "random_forest",
            "lr": "logistic_regression",
            "svm_rbf": "svm_rbf",
            "svm_linear": "svm_linear",
            "svm_poly": "svm_poly",
            "svm_sigmoid": "svm_sigmoid",
            "svm": "svm_rbf",
            "boost": "gradient_boosting",
            "ada": "adaboost",
            "dt": "decision_tree",
            "nb": "naive_bayes",
            "ridge": "ridge",
            "sgd": "sgd",
            "mlp16": "mlp16",
            "mlp32": "mlp32",
            "mlp64": "mlp64",
            "mlp128": "mlp128"
        }
        return fam_map.get(fam_key, fam_key), f"{prefix}disparate_impact_remover"

    if "prejudice" in name:
        return "logistic_regression", f"{prefix}inprocessing"

    if "exp_grad" in name or "grid_search" in name:
        if "rf" in name:
            return "random_forest", f"{prefix}reductions"
        elif "dt" in name:
            return "decision_tree", f"{prefix}reductions"
        else:
            return "logistic_regression", f"{prefix}reductions"

    if "roc" in name or "eq_odds" in name or "calib" in name:
        if "rf" in name:
            return "random_forest", f"{prefix}postprocessing"
        elif "mlp16" in name:
            return "mlp16", f"{prefix}postprocessing"
        elif "mlp32" in name:
            return "mlp32", f"{prefix}postprocessing"
        elif "mlp64" in name:
            return "mlp64", f"{prefix}postprocessing"
        elif "mlp128" in name:
            return "mlp128", f"{prefix}postprocessing"
        elif "svm_linear" in name:
            return "svm_linear", f"{prefix}postprocessing"
        elif "svm_poly" in name:
            return "svm_poly", f"{prefix}postprocessing"
        elif "svm_sigmoid" in name:
            return "svm_sigmoid", f"{prefix}postprocessing"
        elif "svm_rbf" in name or "svm" in name:
            return "svm_rbf", f"{prefix}postprocessing"
        elif "boost" in name:
            return "gradient_boosting", f"{prefix}postprocessing"
        elif "ada" in name:
            return "adaboost", f"{prefix}postprocessing"
        elif "dt" in name:
            return "decision_tree", f"{prefix}postprocessing"
        elif "nb" in name:
            return "naive_bayes", f"{prefix}postprocessing"
        elif "sgd" in name:
            return "sgd", f"{prefix}postprocessing"
        else:
            return "logistic_regression", f"{prefix}postprocessing"

    return "other", f"{prefix}mitigated"


# ==============================================================================
# 1. Single-Source Radiomics Data Loader
# ==============================================================================
class ProstateRadiomicsDataLoader:
    """
    Loads and standardizes single-source Radiomics data for ProstateNet fairness experiments.
    
    File naming convention:
      dataset/[BGMMOCE | biased]/[roi]_[endpoint]_[bias]_training.set.csv
      dataset/testing_set/[roi]_[endpoint]_[bias]_testing.set.csv

    Supported Experiments:
      - lesion_uc2_magnetic.field.strength
      - lesion_uc2_manufacturer
      - prostate_uc1_magnetic.field.strength
      - prostate_uc1_manufacturer
      - prostate_uc2_magnetic.field.strength
      - prostate_uc2_manufacturer
    """
    KNOWN_EXPERIMENTS = [
        "lesion_uc2_magnetic.field.strength",
        "lesion_uc2_manufacturer",
        "prostate_uc1_magnetic.field.strength",
        "prostate_uc1_manufacturer",
        "prostate_uc2_magnetic.field.strength",
        "prostate_uc2_manufacturer"
    ]

    def __init__(self, base_dir="dataset"):
        self.base_dir = base_dir

    @staticmethod
    def normalize_bias_name(bias_str):
        b = str(bias_str).lower().strip()
        if b in ["mfs", "magnetic_field_strength", "magnetic.field.strength", "magnetic"]:
            return "magnetic.field.strength"
        if b in ["manufacturer", "scanner_manufacturer", "vendor"]:
            return "manufacturer"
        return bias_str

    @staticmethod
    def normalize_experiment_id(exp_id):
        """Converts user input like 'lesion_uc2_mfs' to canonical 'lesion_uc2_magnetic.field.strength'."""
        s = str(exp_id).strip()
        s = s.replace("_mfs", "_magnetic.field.strength")
        s = s.replace("_magnetic_field_strength", "_magnetic.field.strength")
        return s

    def get_available_experiments(self):
        """Discovers all valid experiment keys available in dataset/biased/."""
        biased_dir = os.path.join(self.base_dir, "biased")
        if not os.path.exists(biased_dir):
            return self.KNOWN_EXPERIMENTS
        files = glob.glob(os.path.join(biased_dir, "*_training.set.csv"))
        exps = []
        for f in sorted(files):
            basename = os.path.basename(f)
            exp_key = basename.replace("_training.set.csv", "")
            exps.append(exp_key)
        return exps if exps else self.KNOWN_EXPERIMENTS

    def load_single_csv(self, file_path, exp_key):
        """
        Parses a single ProstateNet CSV:
        - Drops non-feature identifiers: 'Unnamed: 0', 'pid'
        - Identifies target endpoint ('uc1' or 'uc2' or last column)
        - Identifies sensitive attribute ('magnetic_field_strength' or 'manufacturer')
        - Retains all 642 radiomics features as float32/float64 DataFrame
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        df = pd.read_csv(file_path)

        # 1. Determine target column: last column is always the ground truth
        # Handles edge cases where BGMMOCE file for prostate_uc2 is labeled 'uc1' in the last column
        target_col = df.columns[-1]
        y = df[target_col].astype(float).fillna(0.0).astype(int).values

        # 2. Determine sensitive attribute column
        if "magnetic.field.strength" in exp_key or "mfs" in exp_key:
            sensitive_col = "magnetic_field_strength"
        elif "manufacturer" in exp_key:
            sensitive_col = "manufacturer"
        else:
            # Fallback detection
            if "magnetic_field_strength" in df.columns:
                sensitive_col = "magnetic_field_strength"
            elif "manufacturer" in df.columns:
                sensitive_col = "manufacturer"
            else:
                sensitive_col = df.columns[-2]

        if sensitive_col not in df.columns:
            raise KeyError(f"Sensitive attribute '{sensitive_col}' not found in {file_path}. Columns: {list(df.columns[-5:])}")

        s = df[sensitive_col].astype(float).fillna(0.0).astype(int).values

        # 3. Patient IDs
        if "pid" in df.columns:
            patient_ids = df["pid"].astype(str).tolist()
        else:
            patient_ids = [f"sample_{i}" for i in range(len(df))]

        # 4. Extract feature columns (drop ID cols, sensitive cols, target col)
        drop_cols = [c for c in ['Unnamed: 0', 'pid', 'magnetic_field_strength', 'manufacturer', 'uc1', 'uc2', target_col] if c in df.columns]
        # Also drop other non-feature columns if present
        X = df.drop(columns=drop_cols).apply(pd.to_numeric, errors='coerce').fillna(0.0)

        # Ensure column names are clean strings
        X.columns = [str(c) for c in X.columns]

        return {
            "X": X,
            "y": y,
            "s": s,
            "patient_ids": patient_ids,
            "target_col": target_col,
            "bias_col": sensitive_col,
            "feature_names": list(X.columns),
            "file_path": file_path
        }

    def load_dataset(self, exp_key, training_set_type="biased"):
        """
        Loads training split and corresponding external testing split for a given experiment.
        exp_key e.g. 'lesion_uc2_magnetic.field.strength'
        """
        norm_exp = self.normalize_experiment_id(exp_key)

        train_file = os.path.join(self.base_dir, training_set_type, f"{norm_exp}_training.set.csv")
        test_file  = os.path.join(self.base_dir, "testing_set", f"{norm_exp}_testing.set.csv")

        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Training CSV not found: {train_file}")
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Testing CSV not found: {test_file}")

        print(f"\n[Loader] Loading Training Set ({training_set_type}): {os.path.basename(train_file)}")
        train_data = self.load_single_csv(train_file, norm_exp)
        print(f"[Loader] Loading Testing Set:               {os.path.basename(test_file)}")
        test_data  = self.load_single_csv(test_file, norm_exp)

        train_data["exp_key"] = norm_exp
        test_data["exp_key"]  = norm_exp

        print(f"  -> Train shape: {train_data['X'].shape} | Target positive rate: {np.mean(train_data['y']):.3f} | Subgroups: {np.unique(train_data['s'])}")
        print(f"  -> Test shape:  {test_data['X'].shape} | Target positive rate: {np.mean(test_data['y']):.3f} | Subgroups: {np.unique(test_data['s'])}")

        return train_data, test_data


# ==============================================================================
# 2. Single-Source Radiomics Preprocessing & Feature Selection
# ==============================================================================
class RadiomicsFeatureSelector:
    """
    Hierarchical preprocessing for high-dimensional radiomics features:
    1. Robust Scaling (5th to 95th percentile quantile range).
    2. Variance Threshold filter to discard constant/quasi-constant textures.
    3. ANOVA F-test filter (custom p-value threshold or SelectKBest).
    4. L1 Logistic Regression sparsity selection with safe fallback.
    """
    def __init__(self, hypes):
        self.hypes = hypes
        self.scaler = None
        self.selected_features = []

    def fit_transform(self, train_data, test_data):
        p_one = self.hypes.get("p_one", 5.0)
        p_two = self.hypes.get("p_two", 95.0)

        # 1. Robust Scaling
        self.scaler = RobustScaler(quantile_range=(p_one, p_two))
        X_tr_raw = train_data["X"]
        X_ts_raw = test_data["X"]
        y_tr = train_data["y"]

        X_tr_sc = pd.DataFrame(self.scaler.fit_transform(X_tr_raw), columns=X_tr_raw.columns)
        X_ts_sc = pd.DataFrame(self.scaler.transform(X_ts_raw), columns=X_ts_raw.columns)

        # 2. Variance Threshold Filter
        var_thresh = self.hypes.get("variance_threshold", 0.0)
        remover = VarianceThreshold(threshold=var_thresh)
        remover.fit(X_tr_sc)
        support_var = remover.get_support(indices=True)
        if len(support_var) == 0:
            cols_after_var = X_tr_sc.columns
            X_var = X_tr_sc.values
        else:
            cols_after_var = X_tr_sc.columns[support_var]
            X_var = X_tr_sc[cols_after_var].values

        # 3. ANOVA F-Test Filter
        use_custom_anova = self.hypes.get("use_custom_anova", True)
        if use_custom_anova:
            f_scores, p_values = f_classif(X_var, y_tr)
            p_thresh = self.hypes.get("p_value", 0.05)
            n_classes = len(np.unique(y_tr))
            dfn = max(1, n_classes - 1)
            dfd = max(1, len(y_tr) - n_classes)
            crit_val = st.f.ppf(q=1.0 - p_thresh, dfn=dfn, dfd=dfd)
            selected_idx = [i for i, p in enumerate(p_values) if p < p_thresh and f_scores[i] >= crit_val]

            min_needed = self.hypes.get("min_features", 10)
            if len(selected_idx) < min_needed:
                k_fallback = min(self.hypes.get("k_best", 40), X_var.shape[1])
                selector = SelectKBest(f_classif, k=k_fallback)
                selector.fit(X_var, y_tr)
                selected_idx = selector.get_support(indices=True)
        else:
            k = min(self.hypes.get("k_best", 40), X_var.shape[1])
            selector = SelectKBest(f_classif, k=k)
            selector.fit(X_var, y_tr)
            selected_idx = selector.get_support(indices=True)

        X_anova = X_var[:, selected_idx]
        cols_after_anova = cols_after_var[selected_idx]

        # 4. L1 Sparsity Control (Logistic Regression)
        l1_c = self.hypes.get("l1_c_value", 0.5)
        l1_sel = SelectFromModel(
            LogisticRegression(
                C=l1_c,
                penalty='l1',
                solver='liblinear',
                max_iter=5000,
                random_state=42
            )
        )
        l1_sel.fit(X_anova, y_tr)
        l1_support = l1_sel.get_support()
        final_cols = cols_after_anova[l1_support].tolist()

        # Fallback if L1 eliminates all features
        if len(final_cols) == 0:
            print(f"  [Selector Warning] L1 selected 0 features, retaining top ANOVA features.")
            final_cols = cols_after_anova[:min(20, len(cols_after_anova))].tolist()

        self.selected_features = final_cols
        print(f"  [Selector] Features selected: {len(self.selected_features)} of {X_tr_raw.shape[1]}")

        X_tr_final = X_tr_sc[self.selected_features]
        X_ts_final = X_ts_sc[self.selected_features]

        return X_tr_final, X_ts_final

    def save_selected_features_txt(self, output_dir, exp_key, training_type):
        os.makedirs(output_dir, exist_ok=True)
        txt_path = os.path.join(output_dir, f"selected_features_{training_type}.txt")
        with open(txt_path, "w") as f:
            f.write(f"Experiment: {exp_key}\n")
            f.write(f"Training Set: {training_type}\n")
            f.write(f"Total Selected Features: {len(self.selected_features)}\n")
            f.write("=" * 60 + "\n")
            for i, feat in enumerate(self.selected_features, 1):
                f.write(f"{i:3d}. {feat}\n")


# ==============================================================================
# 3. Comprehensive Benchmarker & Fairness Auditor
# ==============================================================================
class ComprehensiveBenchmarker:
    """
    Executes benchmark evaluations for:
    - Biased Baseline (observational training set)
    - Fair Reweighing (AIF360 sample weights applied during model training)
    - BGMMOCE (fairness-mitigated training set split)
    - Optional In-processing & Post-processing mitigations on biased split
    """
    def __init__(self, hypes):
        self.hypes = hypes
        self.results_data = {}

    def pandas_to_aif360(self, X_df, y, sensitive_attr_values, sensitive_attr_name):
        df = X_df.copy()
        df['target'] = np.array(y).flatten().astype(float)
        df[sensitive_attr_name] = np.array(sensitive_attr_values).flatten().astype(float)
        df = df.astype(float)
        return BinaryLabelDataset(
            df=df,
            label_names=['target'],
            protected_attribute_names=[sensitive_attr_name],
            favorable_label=1.0,
            unfavorable_label=0.0
        )

    def compute_generalized_sample_weights(self, y, s):
        """
        Calculates exact Statistical Parity sample weights for Reweighing:
          W(s, y) = P(s) * P(y) / P(s, y) = (N_s * N_y) / (N * N_sy)
        Guaranteed to generalize across binary AND multi-class sensitive attributes (e.g. manufacturer {0, 1, 2}),
        while being mathematically identical to AIF360 Reweighing on binary attributes.
        """
        N = len(y)
        p_s = {val: np.mean(s == val) for val in np.unique(s)}
        p_y = {val: np.mean(y == val) for val in np.unique(y)}
        
        p_sy = {}
        for s_val in np.unique(s):
            for y_val in np.unique(y):
                mask = (s == s_val) & (y == y_val)
                prob = np.mean(mask)
                p_sy[(s_val, y_val)] = prob if prob > 0 else 1e-8

        weights = np.array([
            (p_s[s_i] * p_y[y_i]) / p_sy[(s_i, y_i)]
            for s_i, y_i in zip(s, y)
        ])

        if np.isnan(weights).any() or np.isinf(weights).any():
            weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
        return weights

    def compute_fairness_metrics(self, y_true, y_pred, s_values, s_attr_name):
        """Computes fairness metrics via AIF360 ClassificationMetric."""
        df_true = pd.DataFrame({'target': y_true.astype(float), s_attr_name: s_values.astype(float)})
        df_pred = pd.DataFrame({'target': y_pred.astype(float), s_attr_name: s_values.astype(float)})
        bld_true = BinaryLabelDataset(df=df_true, label_names=['target'], protected_attribute_names=[s_attr_name])
        bld_pred = BinaryLabelDataset(df=df_pred, label_names=['target'], protected_attribute_names=[s_attr_name])

        unique_groups = np.unique(s_values)
        if len(unique_groups) == 2:
            unprivileged = [{s_attr_name: float(unique_groups[0])}]
            privileged   = [{s_attr_name: float(unique_groups[1])}]
        else:
            # Multi-class sensitive attribute: largest group as privileged, others as unprivileged
            group_counts = pd.Series(s_values).value_counts()
            priv_val = float(group_counts.index[0])
            privileged = [{s_attr_name: priv_val}]
            unprivileged = [{s_attr_name: float(g)} for g in unique_groups if float(g) != priv_val]

        try:
            metric = ClassificationMetric(bld_true, bld_pred, unprivileged_groups=unprivileged, privileged_groups=privileged)
            spd = float(metric.statistical_parity_difference())
            di  = float(metric.disparate_impact())
            eod = float(metric.equal_opportunity_difference())
            aod = float(metric.average_odds_difference())
        except Exception as e:
            spd, di, eod, aod = 0.0, 1.0, 0.0, 0.0

        return {
            "statistical_parity_difference": spd,
            "disparate_impact": di,
            "equal_opportunity_difference": eod,
            "average_odds_difference": aod
        }

    def compute_discrimination_and_calibration_metrics(self, y_true, probs, s_values):
        """
        Computes key discrimination, calibration, and subgroup metrics:
        1. Harrell's Concordance Index (C-Index)
        2. Subgroup Concordance Indices & Disparity
        3. Brier Score & Index of Prediction Accuracy (IPA)
        4. Expected Calibration Error (ECE)
        """
        try:
            c_index = float(concordance_index(event_times=y_true, predicted_scores=probs))
        except Exception:
            c_index = 0.5

        sub_c = {}
        for g in np.unique(s_values):
            mask = (s_values == g)
            if len(np.unique(y_true[mask])) >= 2:
                try:
                    sub_c[f"c_index_group_{int(g)}"] = float(concordance_index(y_true[mask], probs[mask]))
                except Exception:
                    sub_c[f"c_index_group_{int(g)}"] = 0.5
            else:
                sub_c[f"c_index_group_{int(g)}"] = 0.5

        group_keys = [k for k in sub_c if k.startswith("c_index_group_")]
        if len(group_keys) >= 2:
            vals = [sub_c[k] for k in group_keys]
            c_index_disp = float(abs(max(vals) - min(vals)))
        else:
            c_index_disp = 0.0

        try:
            brier = float(brier_score_loss(y_true, probs))
            null_prob = float(np.mean(y_true))
            brier_null = float(brier_score_loss(y_true, np.full_like(probs, null_prob)))
            ipa = float(1.0 - (brier / brier_null)) if brier_null > 0 else 0.0
        except Exception:
            brier = 0.25
            ipa = 0.0

        try:
            n_bins = 5
            bins = np.linspace(0.0, 1.0, n_bins + 1)
            bin_indices = np.digitize(probs, bins) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            ece = 0.0
            for i in range(n_bins):
                mask = (bin_indices == i)
                if np.sum(mask) > 0:
                    bin_acc = np.mean(y_true[mask])
                    bin_conf = np.mean(probs[mask])
                    ece += (np.sum(mask) / len(y_true)) * abs(bin_acc - bin_conf)
            ece = float(ece)
        except Exception:
            ece = 0.0

        return {
            "c_index": c_index,
            "brier_score": brier,
            "ipa": ipa,
            "ece": ece,
            "c_index_disparity": c_index_disp,
            **sub_c
        }

    def get_all_classifiers(self, s_attr_name):
        """
        Defines the 16 Scikit-Learn classifier families supporting sample reweighting:
        1. random_forest
        2. logistic_regression (l2, l1, elastic)
        3. svm_rbf
        4. svm_linear
        5. svm_poly
        6. svm_sigmoid
        7. gradient_boosting
        8. adaboost
        9. mlp16
        10. mlp32
        11. mlp64
        12. mlp128
        13. decision_tree
        14. naive_bayes
        15. ridge
        16. sgd
        """
        svm_c = self.hypes.get("svm_c", 1.0)

        base_lr = LogisticRegression(penalty='l2', solver='lbfgs', C=0.1, max_iter=5000, random_state=42)
        base_rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
        base_svm_rbf = SVC(kernel='rbf', probability=True, C=svm_c, gamma='scale', random_state=42)
        base_svm_linear = SVC(kernel='linear', probability=True, C=svm_c, random_state=42)
        base_svm_poly = SVC(kernel='poly', degree=3, probability=True, C=svm_c, random_state=42)
        base_svm_sigmoid = SVC(kernel='sigmoid', probability=True, C=svm_c, random_state=42)
        base_boost = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)

        models = {
            # === Standard Scikit-Learn Classifiers (Biased / BGMMOCE) ===
            "logistic_l1": LogisticRegression(penalty='l1', solver='liblinear', C=0.5, max_iter=5000, random_state=42),
            "logistic_l2": base_lr,
            "logistic_elastic": LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=0.1, max_iter=5000, random_state=42),
            "ridge": RidgeClassifier(alpha=1.0, random_state=42),
            "sgd": SGDClassifier(loss='log_loss', penalty='l2', alpha=0.01, max_iter=2000, random_state=42),
            "svm_linear": base_svm_linear,
            "svm_rbf": base_svm_rbf,
            "svm_poly": base_svm_poly,
            "svm_sigmoid": base_svm_sigmoid,
            "decision_tree": DecisionTreeClassifier(max_depth=5, min_samples_split=4, random_state=42),
            "random_forest": base_rf,
            "gradient_boosting": base_boost,
            "adaboost": AdaBoostClassifier(n_estimators=50, random_state=42),
            "naive_bayes": GaussianNB(),
            "mlp16": MLPClassifier(hidden_layer_sizes=(16,), max_iter=1000, random_state=42),
            "mlp32": MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=42),
            "mlp64": MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, random_state=42),
            "mlp128": MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, random_state=42),

            # === AIF360 Pre-Processing Mitigation: Reweighing ===
            "fair_reweighted_lr": LogisticRegression(penalty='l2', solver='lbfgs', C=0.1, max_iter=5000, random_state=42),
            "fair_reweighted_rf": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1),
            "fair_reweighted_svm_rbf": SVC(kernel='rbf', probability=True, C=svm_c, gamma='scale', random_state=42),
            "fair_reweighted_svm_linear": SVC(kernel='linear', probability=True, C=svm_c, random_state=42),
            "fair_reweighted_svm_poly": SVC(kernel='poly', degree=3, probability=True, C=svm_c, random_state=42),
            "fair_reweighted_svm_sigmoid": SVC(kernel='sigmoid', probability=True, C=svm_c, random_state=42),
            "fair_reweighted_boost": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42),
            "fair_reweighted_ada": AdaBoostClassifier(n_estimators=50, random_state=42),
            "fair_reweighted_dt": DecisionTreeClassifier(max_depth=5, min_samples_split=4, random_state=42),
            "fair_reweighted_nb": GaussianNB(),
            "fair_reweighted_ridge": RidgeClassifier(alpha=1.0, random_state=42),
            "fair_reweighted_sgd": SGDClassifier(loss='log_loss', penalty='l2', alpha=0.01, max_iter=2000, random_state=42),
            "fair_reweighted_mlp16": MLPClassifier(hidden_layer_sizes=(16,), max_iter=1000, random_state=42),
            "fair_reweighted_mlp32": MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=42),
            "fair_reweighted_mlp64": MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, random_state=42),
            "fair_reweighted_mlp128": MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, random_state=42),

            # === AIF360 Pre-Processing Mitigation: Disparate Impact Remover ===
            "fair_dir_lr": LogisticRegression(penalty='l2', solver='lbfgs', C=0.1, max_iter=5000, random_state=42),
            "fair_dir_rf": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1),
            "fair_dir_svm_rbf": SVC(kernel='rbf', probability=True, C=svm_c, gamma='scale', random_state=42),
            "fair_dir_svm_linear": SVC(kernel='linear', probability=True, C=svm_c, random_state=42),
            "fair_dir_svm_poly": SVC(kernel='poly', degree=3, probability=True, C=svm_c, random_state=42),
            "fair_dir_svm_sigmoid": SVC(kernel='sigmoid', probability=True, C=svm_c, random_state=42),
            "fair_dir_boost": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42),
            "fair_dir_ada": AdaBoostClassifier(n_estimators=50, random_state=42),
            "fair_dir_dt": DecisionTreeClassifier(max_depth=5, min_samples_split=4, random_state=42),
            "fair_dir_nb": GaussianNB(),
            "fair_dir_ridge": RidgeClassifier(alpha=1.0, random_state=42),
            "fair_dir_sgd": SGDClassifier(loss='log_loss', penalty='l2', alpha=0.01, max_iter=2000, random_state=42),
            "fair_dir_mlp16": MLPClassifier(hidden_layer_sizes=(16,), max_iter=1000, random_state=42),
            "fair_dir_mlp32": MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=42),
            "fair_dir_mlp64": MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, random_state=42),
            "fair_dir_mlp128": MLPClassifier(hidden_layer_sizes=(128,), max_iter=1000, random_state=42),

            # === AIF360 In-Processing Mitigation: Prejudice Remover ===
            "fair_prejudice_remover": PrejudiceRemover(eta=1.0, sensitive_attr=s_attr_name),

            # === AIF360 In-Processing Mitigation: Reductions ===
            "fair_exp_grad_eo_lr": ExponentiatedGradientReduction(
                estimator=LogisticRegression(solver='liblinear'),
                constraints="EqualizedOdds",
                drop_prot_attr=False
            ),
            "fair_exp_grad_eo_rf": ExponentiatedGradientReduction(
                estimator=RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42),
                constraints="EqualizedOdds",
                drop_prot_attr=False
            ),
            "fair_grid_search_eo": GridSearchReduction(
                estimator=LogisticRegression(solver='liblinear'),
                constraints="EqualizedOdds",
                grid_size=10,
                drop_prot_attr=False
            ),

            # === AIF360 Post-Processing Mitigation ===
            "fair_roc_rf": ("post_roc", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)),
            "fair_roc_lr": ("post_roc", LogisticRegression(penalty='l2', solver='lbfgs', C=0.1, max_iter=5000)),
            "fair_roc_svm_rbf": ("post_roc", SVC(kernel='rbf', probability=True, C=svm_c, gamma='scale', random_state=42)),
            "fair_roc_svm_linear": ("post_roc", SVC(kernel='linear', probability=True, C=svm_c, random_state=42)),
            "fair_roc_svm_poly": ("post_roc", SVC(kernel='poly', degree=3, probability=True, C=svm_c, random_state=42)),
            "fair_roc_svm_sigmoid": ("post_roc", SVC(kernel='sigmoid', probability=True, C=svm_c, random_state=42)),
            "fair_roc_boost": ("post_roc", GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)),
            "fair_roc_dt": ("post_roc", DecisionTreeClassifier(max_depth=5, min_samples_split=4, random_state=42)),
            "fair_eq_odds_rf": ("post_eq_odds", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)),
            "fair_eq_odds_lr": ("post_eq_odds", LogisticRegression(penalty='l2', solver='lbfgs', C=0.1, max_iter=5000)),
            "fair_calib_eq_odds_rf": ("post_calib_eq_odds", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)),
            "fair_calib_eq_odds_lr": ("post_calib_eq_odds", LogisticRegression(penalty='l2', solver='lbfgs', C=0.1, max_iter=5000))
        }

        return models

    def save_confusion_matrix(self, y_true, y_pred, model_name, path, endpoint_label="Prostate Target"):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix: {model_name}')
        plt.ylabel(f'Actual ({endpoint_label})')
        plt.xlabel(f'Predicted ({endpoint_label})')
        plt.tight_layout()
        plt.savefig(os.path.join(path, f"{model_name}_cm.png"), dpi=300)
        plt.close()

    def explain_with_shap(self, clf, name, X_tr, X_ts, output_dir):
        """
        Computes SHAP explainability on test inference, saves summary plots,
        bar importance plots, and ranked feature importance table.
        """
        try:
            is_tree = isinstance(clf, (RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier))
            is_linear = isinstance(clf, (LogisticRegression, RidgeClassifier, SGDClassifier))

            if is_tree:
                explainer = shap.TreeExplainer(clf)
                shap_obj = explainer(X_ts)
                shap_vals = shap_obj.values
                if len(shap_vals.shape) == 3:
                    shap_vals = shap_vals[:, :, 1]
            elif is_linear:
                explainer = shap.LinearExplainer(clf, X_tr)
                shap_obj = explainer(X_ts)
                shap_vals = shap_obj.values
            else:
                background = shap.sample(X_tr, min(20, len(X_tr)))
                if hasattr(clf, "predict_proba"):
                    predict_fn = lambda x: clf.predict_proba(x)[:, 1]
                elif hasattr(clf, "decision_function"):
                    predict_fn = clf.decision_function
                else:
                    predict_fn = clf.predict
                explainer = shap.KernelExplainer(predict_fn, background, silent=True)
                shap_vals = np.array(explainer.shap_values(X_ts, silent=True))
                if len(shap_vals.shape) == 3:
                    shap_vals = shap_vals[:, :, 1]

            np.save(os.path.join(output_dir, f"{name}_shap_values.npy"), shap_vals)

            X_ts_display = X_ts.copy()
            X_ts_display.columns = [shorten_feature_name(c) for c in X_ts.columns]

            # Beeswarm Summary Plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_vals, X_ts_display, show=False, max_display=15)
            plt.title(f"SHAP Summary (Test Inference): {name}", fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{name}_shap_summary.png"), dpi=300)
            plt.close()

            # Bar Plot (Mean |SHAP|)
            plt.figure(figsize=(8, 6))
            shap.summary_plot(shap_vals, X_ts_display, plot_type="bar", show=False, max_display=15)
            plt.title(f"SHAP Feature Importance: {name}", fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{name}_shap_bar.png"), dpi=300)
            plt.close()

            # Feature Importance Table
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            feat_df = pd.DataFrame({
                "feature": [shorten_feature_name(c) for c in X_ts.columns],
                "raw_feature": X_ts.columns,
                "mean_abs_shap": mean_abs_shap
            }).sort_values(by="mean_abs_shap", ascending=False)
            feat_df.to_csv(os.path.join(output_dir, f"{name}_shap_importance.csv"), index=False)

        except Exception as e:
            print(f"    [SHAP Warning] Skipping SHAP for {name} due to: {e}")

    def run_pipeline(self, train_data, test_data, X_tr, X_ts, output_dir, training_type="biased"):
        """
        Runs complete model training, evaluation, fairness calculation, and artifact logging.
        """
        y_train = train_data["y"]
        s_train = train_data["s"]
        y_test  = test_data["y"]
        s_test  = test_data["s"]
        s_attr_name = train_data["bias_col"]
        target_name = train_data.get("target_col", "target")

        os.makedirs(output_dir, exist_ok=True)
        print(f"\n[Benchmarker] Output Directory: {output_dir}")

        if str(training_type).upper() == "BGMMOCE":
            # Per prompt: No reweighing experiments for BGMMOCE, only for the biased radiomics
            all_clfs = self.get_all_classifiers(s_attr_name)
            classifiers = {k: v for k, v in all_clfs.items() if not k.startswith("fair_")}
            reweigh_weights = None
            X_tr_repaired, X_ts_repaired = None, None
            bld_tr, bld_ts = None, None
        else:
            classifiers = self.get_all_classifiers(s_attr_name)
            unique_s = np.unique(s_train)
            if len(unique_s) == 2:
                unprivileged_groups = [{s_attr_name: float(unique_s[0])}]
                privileged_groups   = [{s_attr_name: float(unique_s[1])}]
            else:
                s_counts = pd.Series(s_train).value_counts()
                priv_val = float(s_counts.index[0])
                privileged_groups = [{s_attr_name: priv_val}]
                unprivileged_groups = [{s_attr_name: float(g)} for g in unique_s if float(g) != priv_val]

            # Precompute sample weights for Reweighing
            reweigh_weights = self.compute_generalized_sample_weights(y_train, s_train)

            # Precompute AIF360 datasets & Disparate Impact Remover
            try:
                bld_tr = self.pandas_to_aif360(X_tr, y_train, s_train, s_attr_name)
                bld_ts = self.pandas_to_aif360(X_ts, y_test, s_test, s_attr_name)
                dir_remover = DisparateImpactRemover(repair_level=1.0, sensitive_attribute=s_attr_name)
                bld_tr_repaired = dir_remover.fit_transform(bld_tr)
                bld_ts_repaired = dir_remover.fit_transform(bld_ts)
                sens_idx = bld_tr.feature_names.index(s_attr_name)
                X_tr_repaired = np.delete(bld_tr_repaired.features, sens_idx, axis=1)
                X_ts_repaired = np.delete(bld_ts_repaired.features, sens_idx, axis=1)
            except Exception as e:
                print(f"  [AIF360 Setup Warning] {e}")
                bld_tr, bld_ts = None, None
                X_tr_repaired, X_ts_repaired = X_tr.values, X_ts.values

        # If quick test mode enabled in hypes, filter classifiers to key candidates
        if self.hypes.get("quick_test", False):
            quick_keys = [
                "random_forest", "logistic_l2", "svm_rbf",
                "fair_reweighted_rf", "fair_reweighted_lr", "fair_reweighted_svm_rbf"
            ]
            classifiers = {k: v for k, v in classifiers.items() if k in quick_keys}

        summary_rows = []

        for name, clf_item in classifiers.items():
            print(f"Evaluating: {name}...")

            try:
                # CASE A: AIF360 In-Processing
                if ("fair_prejudice" in name or "fair_exp_grad" in name or "fair_grid_search" in name) and bld_tr is not None:
                    clf = clf_item
                    clf.fit(bld_tr)
                    pred_bld = clf.predict(bld_ts)
                    preds = pred_bld.labels.flatten().astype(int)
                    if hasattr(pred_bld, "scores") and pred_bld.scores is not None and len(pred_bld.scores) > 0:
                        probs = pred_bld.scores.flatten()
                    else:
                        probs = preds.astype(float)
                    fit_model_for_save = clf

                # CASE B: AIF360 Pre-Processing: Disparate Impact Remover
                elif "fair_dir_" in name and X_tr_repaired is not None:
                    clf = clf_item
                    clf.fit(X_tr_repaired, y_train)
                    preds = clf.predict(X_ts_repaired)
                    if hasattr(clf, "predict_proba"):
                        probs = clf.predict_proba(X_ts_repaired)[:, 1]
                    else:
                        probs = preds.astype(float)
                    fit_model_for_save = clf

                # CASE C: AIF360 Pre-Processing: Fair Reweighing
                elif "fair_reweighted" in name:
                    clf = clf_item
                    clf.fit(X_tr, y_train, sample_weight=reweigh_weights)
                    preds = clf.predict(X_ts)
                    if hasattr(clf, "predict_proba"):
                        probs = clf.predict_proba(X_ts)[:, 1]
                    elif hasattr(clf, "decision_function"):
                        df_scores = clf.decision_function(X_ts)
                        probs = (df_scores - df_scores.min()) / (df_scores.max() - df_scores.min() + 1e-8)
                    else:
                        probs = preds.astype(float)
                    fit_model_for_save = clf

                # CASE D: AIF360 Post-Processing
                elif isinstance(clf_item, tuple) and clf_item[0].startswith("post_") and bld_tr is not None:
                    post_type, base_clf = clf_item
                    base_clf.fit(X_tr, y_train)
                    tr_probs = base_clf.predict_proba(X_tr)[:, 1]
                    tr_preds = (tr_probs >= 0.5).astype(float)
                    bld_tr_pred = bld_tr.copy()
                    bld_tr_pred.scores = tr_probs.reshape(-1, 1)
                    bld_tr_pred.labels = tr_preds.reshape(-1, 1)

                    if post_type == "post_roc":
                        post_proc = RejectOptionClassification(
                            unprivileged_groups=unprivileged_groups,
                            privileged_groups=privileged_groups,
                            metric_name="Statistical parity difference"
                        )
                    elif post_type == "post_eq_odds":
                        post_proc = EqOddsPostprocessing(
                            unprivileged_groups=unprivileged_groups,
                            privileged_groups=privileged_groups
                        )
                    elif post_type == "post_calib_eq_odds":
                        post_proc = CalibratedEqOddsPostprocessing(
                            unprivileged_groups=unprivileged_groups,
                            privileged_groups=privileged_groups,
                            cost_constraint="weighted"
                        )
                    post_proc.fit(bld_tr, bld_tr_pred)

                    ts_probs = base_clf.predict_proba(X_ts)[:, 1]
                    ts_preds = (ts_probs >= 0.5).astype(float)
                    bld_ts_pred = bld_ts.copy()
                    bld_ts_pred.scores = ts_probs.reshape(-1, 1)
                    bld_ts_pred.labels = ts_preds.reshape(-1, 1)

                    bld_ts_transformed = post_proc.predict(bld_ts_pred)
                    preds = bld_ts_transformed.labels.flatten().astype(int)
                    probs = bld_ts_transformed.scores.flatten() if bld_ts_transformed.scores is not None else preds.astype(float)
                    fit_model_for_save = (base_clf, post_proc)

                # CASE E: Standard Scikit-Learn Classifiers
                else:
                    clf = clf_item
                    if self.hypes.get("use_smote", False):
                        try:
                            sm = SMOTE(
                                sampling_strategy=self.hypes.get("smote_strategy", 1.0),
                                k_neighbors=min(5, np.sum(y_train == 0) - 1),
                                random_state=42
                            )
                            X_tr_fit, y_tr_fit = sm.fit_resample(X_tr, y_train)
                        except Exception as e:
                            print(f"  [SMOTE Warning] {e}. Falling back to unresampled data.")
                            X_tr_fit, y_tr_fit = X_tr, y_train
                    else:
                        X_tr_fit, y_tr_fit = X_tr, y_train

                    clf.fit(X_tr_fit, y_tr_fit)
                    preds = clf.predict(X_ts)
                    if hasattr(clf, "predict_proba"):
                        probs = clf.predict_proba(X_ts)[:, 1]
                    elif hasattr(clf, "decision_function"):
                        df_scores = clf.decision_function(X_ts)
                        probs = (df_scores - df_scores.min()) / (df_scores.max() - df_scores.min() + 1e-8)
                    else:
                        probs = preds.astype(float)
                    fit_model_for_save = clf

                # Metrics Calculation
                try:
                    auc = float(roc_auc_score(y_test, probs))
                except Exception:
                    auc = 0.5

                acc = float(accuracy_score(y_test, preds))
                f1  = float(f1_score(y_test, preds, zero_division=0))
                prec = float(precision_score(y_test, preds, zero_division=0))
                rec = float(recall_score(y_test, preds, zero_division=0))

                disc_metrics = self.compute_discrimination_and_calibration_metrics(y_test, probs, s_test)
                fairness_metrics = self.compute_fairness_metrics(y_test, preds, s_test, s_attr_name)
                fpr, tpr, _ = roc_curve(y_test, probs)

                self.results_data[name] = {
                    "mean_auc": auc,
                    "std_auc": 0.0,
                    "c_index": disc_metrics["c_index"],
                    "brier_score": disc_metrics["brier_score"],
                    "ipa": disc_metrics["ipa"],
                    "ece": disc_metrics["ece"],
                    "accuracy": acc,
                    "f1": f1,
                    "precision": prec,
                    "recall": rec,
                    "survival": disc_metrics,
                    "fairness": fairness_metrics,
                    "folds": [{
                        "auc": auc,
                        "c_index": disc_metrics["c_index"],
                        "brier_score": disc_metrics["brier_score"],
                        "f1": f1,
                        "accuracy": acc,
                        "fpr": fpr.tolist(),
                        "tpr": tpr.tolist()
                    }]
                }

                summary_rows.append({
                    "Model": name,
                    "AUC": round(auc, 4),
                    "C_Index": round(disc_metrics["c_index"], 4),
                    "Brier_Score": round(disc_metrics["brier_score"], 4),
                    "IPA": round(disc_metrics["ipa"], 4),
                    "ECE": round(disc_metrics["ece"], 4),
                    "Accuracy": round(acc, 4),
                    "F1": round(f1, 4),
                    "Precision": round(prec, 4),
                    "Recall": round(rec, 4),
                    "Statistical_Parity_Diff": round(fairness_metrics["statistical_parity_difference"], 4),
                    "Disparate_Impact": round(fairness_metrics["disparate_impact"], 4),
                    "Equal_Opportunity_Diff": round(fairness_metrics["equal_opportunity_difference"], 4),
                    "Average_Odds_Diff": round(fairness_metrics["average_odds_difference"], 4),
                    "C_Index_Disparity": round(disc_metrics["c_index_disparity"], 4)
                })

                print(f"  -> AUC: {auc:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | StatParityDiff: {fairness_metrics['statistical_parity_difference']:.4f}")

                # Save Model, Selected Features, and Predictions in variant folder
                clf_family, variant_name = get_classifier_family_and_variant(name, training_type)
                variant_dir = os.path.join(output_dir, clf_family, variant_name)
                os.makedirs(variant_dir, exist_ok=True)

                with open(os.path.join(variant_dir, f"{name}_clf.pkl"), "wb") as f:
                    pkl.dump(fit_model_for_save, f, protocol=pkl.HIGHEST_PROTOCOL)
                with open(os.path.join(variant_dir, "model.pkl"), "wb") as f:
                    pkl.dump(fit_model_for_save, f, protocol=pkl.HIGHEST_PROTOCOL)

                np.save(os.path.join(variant_dir, f"{name}_selected_feats.npy"), np.array(X_tr.columns))
                np.save(os.path.join(variant_dir, "selected_feats.npy"), np.array(X_tr.columns))

                np.save(os.path.join(variant_dir, "preds.npy"), np.array(preds))
                np.save(os.path.join(variant_dir, "probs.npy"), np.array(probs))
                np.save(os.path.join(variant_dir, "y_test.npy"), np.array(y_test))
                np.save(os.path.join(variant_dir, "s_test.npy"), np.array(s_test))

                with open(os.path.join(variant_dir, f"{name}_metrics.json"), "w") as f:
                    json.dump(self.results_data[name], f, indent=4)

                self.save_confusion_matrix(y_test, preds, name, variant_dir, endpoint_label=target_name)

                # SHAP Explainability
                run_shap = self.hypes.get("run_shap", True)
                shap_targets = self.hypes.get("shap_models", [
                    "random_forest", "gradient_boosting", "logistic_l2",
                    "svm_rbf", "svm_linear", "svm_poly", "svm_sigmoid",
                    "decision_tree",
                    "fair_reweighted_rf", "fair_reweighted_lr", "fair_reweighted_boost", "fair_reweighted_dt",
                    "fair_reweighted_svm_rbf", "fair_reweighted_svm_linear", "fair_reweighted_svm_poly", "fair_reweighted_svm_sigmoid",
                    "adaboost", "naive_bayes",
                    "mlp16", "mlp32", "mlp64", "mlp128",
                    "fair_reweighted_mlp16", "fair_reweighted_mlp32", "fair_reweighted_mlp64", "fair_reweighted_mlp128"
                ])
                if run_shap and (shap_targets == "all" or name in shap_targets):
                    if "fair_dir_" in name and X_tr_repaired is not None:
                        rep_cols = [f"rep_feat_{i}" for i in range(X_tr_repaired.shape[1])]
                        df_tr_rep = pd.DataFrame(X_tr_repaired, columns=rep_cols)
                        df_ts_rep = pd.DataFrame(X_ts_repaired, columns=rep_cols)
                        self.explain_with_shap(clf, name, df_tr_rep, df_ts_rep, variant_dir)
                    elif not ("fair_prejudice" in name or "fair_exp_grad" in name or "fair_grid_search" in name or isinstance(clf_item, tuple)):
                        self.explain_with_shap(clf, name, X_tr, X_ts, variant_dir)

            except Exception as e:
                print(f"  [!] Failed evaluation for {name}: {e}")

        # Update JSON results at experiment level
        json_path = os.path.join(output_dir, "benchmark_results.json")
        combined_results = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    combined_results = json.load(f)
            except Exception:
                combined_results = {}
        for k, v in self.results_data.items():
            key_tag = f"{k}_BGMMOCE" if str(training_type).upper() == "BGMMOCE" else k
            combined_results[key_tag] = v
        with open(json_path, "w") as f:
            json.dump(combined_results, f, indent=4)

        # Update summary CSV at experiment level
        csv_path = os.path.join(output_dir, "benchmark_summary.csv")
        new_summary_df = pd.DataFrame(summary_rows)
        new_summary_df["Training_Split"] = training_type
        if os.path.exists(csv_path):
            try:
                old_df = pd.read_csv(csv_path)
                combined_df = pd.concat([old_df[old_df.get("Training_Split", "") != training_type], new_summary_df], ignore_index=True)
            except Exception:
                combined_df = new_summary_df
        else:
            combined_df = new_summary_df
        combined_df.to_csv(csv_path, index=False)

        print("\n" + "=" * 80)
        print("EXPERIMENT BENCHMARK SUMMARY (Top 10 by AUC):")
        print("=" * 80)
        if not combined_df.empty:
            print(combined_df.sort_values(by="AUC", ascending=False).head(10).to_string(index=False))
        print(f"\nAll outputs saved to: {output_dir}\n")


def get_experiment_dir(base_dir="chkp", exp_name=None):
    os.makedirs(base_dir, exist_ok=True)
    if exp_name:
        target = os.path.join(base_dir, exp_name)
        os.makedirs(target, exist_ok=True)
        return target

    now = datetime.now()
    today_prefix = f"exp_{now.strftime('%d.%m')}_"
    existing_today = sorted(glob.glob(os.path.join(base_dir, f"{today_prefix}*")))
    if existing_today:
        latest = existing_today[-1]
        if (time.time() - os.path.getmtime(latest)) < 2700:
            return latest

    default_name = f"exp_{now.strftime('%d.%m')}_{now.strftime('%Hh%Mm')}"
    target = os.path.join(base_dir, default_name)
    os.makedirs(target, exist_ok=True)
    return target


# ==============================================================================
# 4. Main Execution Block
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="ProstateNet Single-Source Radiomics Fairness Benchmark")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Specific experiment key to run (e.g. lesion_uc2_magnetic.field.strength, or 'all')")
    parser.add_argument("--roi", type=str, default=None, choices=["lesion", "prostate", "all"],
                        help="Filter by Region of Interest (lesion, prostate, all)")
    parser.add_argument("--endpoint", type=str, default=None, choices=["uc1", "uc2", "all"],
                        help="Filter by Endpoint (uc1 = detection, uc2 = aggressiveness, all)")
    parser.add_argument("--bias", type=str, default=None,
                        help="Filter by Bias attribute (magnetic.field.strength, manufacturer, all)")
    parser.add_argument("--training-type", type=str, default="biased", choices=["biased", "BGMMOCE"],
                        help="Training dataset split: 'biased' (runs base + reweighing) or 'BGMMOCE' (no reweighing)")
    parser.add_argument("--use-smote", action="store_true", default=False,
                        help="Apply SMOTE resampling to training data")
    parser.add_argument("--quick-test", action="store_true", default=False,
                        help="Run quick verification with representative subset of classifiers and reduced SHAP")
    parser.add_argument("--no-shap", action="store_true", default=False,
                        help="Disable SHAP computation for faster execution")
    parser.add_argument("--exp-name", type=str, default=None,
                        help="Experiment folder name under chkp/, e.g. exp_19.10_13h00m")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Custom output directory path")
    args = parser.parse_args()

    hypes = {
        "p_one": 5.0,
        "p_two": 95.0,
        "variance_threshold": 0.0,
        "use_custom_anova": True,
        "p_value": 0.005,
        "k_best": 40,
        "min_features": 10,
        "l1_c_value": 0.5,
        "use_smote": args.use_smote,
        "smote_strategy": 1.0,
        "svm_c": 1.0,
        "quick_test": args.quick_test,
        "run_shap": not args.no_shap,
        "shap_models": [
            "random_forest", "gradient_boosting", "logistic_l2",
            "svm_rbf", "svm_linear", "svm_poly", "svm_sigmoid",
            "decision_tree",
            "fair_reweighted_rf", "fair_reweighted_lr", "fair_reweighted_boost", "fair_reweighted_dt",
            "fair_reweighted_svm_rbf", "fair_reweighted_svm_linear", "fair_reweighted_svm_poly", "fair_reweighted_svm_sigmoid",
            "adaboost", "naive_bayes",
            "mlp16", "mlp32", "mlp64", "mlp128",
            "fair_reweighted_mlp16", "fair_reweighted_mlp32", "fair_reweighted_mlp64", "fair_reweighted_mlp128"
        ] if not args.quick_test else ["random_forest", "logistic_l2"]
    }

    loader = ProstateRadiomicsDataLoader(base_dir="dataset")
    available_exps = loader.get_available_experiments()

    # Determine which experiments to run
    if args.experiment and args.experiment != "all":
        exps_to_run = [loader.normalize_experiment_id(args.experiment)]
    else:
        exps_to_run = available_exps.copy()
        if args.roi and args.roi != "all":
            exps_to_run = [e for e in exps_to_run if e.startswith(args.roi)]
        if args.endpoint and args.endpoint != "all":
            exps_to_run = [e for e in exps_to_run if f"_{args.endpoint}_" in e]
        if args.bias and args.bias != "all":
            norm_b = loader.normalize_bias_name(args.bias)
            exps_to_run = [e for e in exps_to_run if norm_b in e]

    if not exps_to_run:
        print(f"[!] No matching experiments found for arguments: roi={args.roi}, endpoint={args.endpoint}, bias={args.bias}, experiment={args.experiment}")
        print(f"    Available experiments: {available_exps}")
        return

    exp_dir = args.output_dir if args.output_dir else get_experiment_dir("chkp", args.exp_name)

    print("\n" + "=" * 80)
    print("PROSTATENET RADIOMICS FAIRNESS BENCHMARK")
    print(f"EXPERIMENT ROOT DIRECTORY: {exp_dir}")
    print(f"TRAINING SPLIT:           {args.training_type}")
    print(f"EXPERIMENTS TO RUN:       {exps_to_run}")
    print("=" * 80)

    for exp_key in exps_to_run:
        print("\n" + "#" * 80)
        print(f"RUNNING BENCHMARK: {exp_key} | Split={args.training_type}")
        print("#" * 80)

        # 1. Load Data
        train_data, test_data = loader.load_dataset(exp_key=exp_key, training_set_type=args.training_type)

        # 2. Radiomics Feature Selection
        selector = RadiomicsFeatureSelector(hypes)
        X_tr, X_ts = selector.fit_transform(train_data, test_data)

        # 3. Setup output folder: <exp_dir>/<experiment_name>/
        out_dir = os.path.join(exp_dir, exp_key)
        os.makedirs(out_dir, exist_ok=True)
        selector.save_selected_features_txt(out_dir, exp_key, args.training_type)

        # 4. Run Benchmark
        benchmarker = ComprehensiveBenchmarker(hypes)
        benchmarker.run_pipeline(
            train_data=train_data,
            test_data=test_data,
            X_tr=X_tr,
            X_ts=X_ts,
            output_dir=out_dir,
            training_type=args.training_type
        )

    # 5. Check if both 'biased' and 'BGMMOCE' variants exist in exp_dir to automatically generate reports and disparity plots
    has_biased = any(glob.glob(os.path.join(exp_dir, "*", "*", "biased")))
    has_bgmmoce = any(glob.glob(os.path.join(exp_dir, "*", "*", "BGMMOCE")))
    if has_biased and has_bgmmoce:
        print("\n" + "=" * 80)
        print("DETECTED BOTH BIASED AND BGMMOCE RESULTS: Running Automated Post-Processing")
        print("=" * 80)
        py_bin = sys.executable
        try:
            import subprocess
            print(">>> Running getResultsV2.py...")
            subprocess.run([py_bin, "getResultsV2.py", "--base-dir", exp_dir], check=False)
            print(">>> Running plotClassFairness.py (standard + best_performing_models)...")
            subprocess.run([py_bin, "plotClassFairness.py", "--base-dir", exp_dir], check=False)
            print(">>> Automated Post-Processing complete.")
        except Exception as e:
            print(f"  [!] Note: Automated post-processing encountered an issue: {e}")


if __name__ == '__main__':
    main()
