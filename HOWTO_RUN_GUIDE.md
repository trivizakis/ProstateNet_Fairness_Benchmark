# How-To Run Guide: ProstateNet Single-Source Radiomics Fairness Benchmark

This repository implements a single-source radiomics machine learning benchmark for **Prostate Cancer Detection (UC1)** and **Prostate Cancer Aggressiveness (UC2)** across **Lesion** and **Prostate** regions of interest, with a focus on institutional and technical fairness auditing across **Magnetic Field Strength** and **Scanner Manufacturer**.

---

## 1. Quick Start (End-to-End Execution)

To run the complete benchmark pipeline across all experiments, both training splits (`biased` + Fair Reweighing vs `BGMMOCE`), and automatically generate all performance metrics and fairness disparity plots:

```bash
# 1. Activate environment
conda activate nsclc_fairness

# 2. Run end-to-end benchmark (or single experiment)
python run_benchmark.py --experiment all
```

Or run step-by-step with custom session naming:

```bash
# Step 1: Train Biased baselines and AIF360 Fair-Reweighted models
python analysisv3.py --experiment all --training-type biased --exp-name exp_prostate_run

# Step 2: Train BGMMOCE models (no reweighing on BGMMOCE)
python analysisv3.py --experiment all --training-type BGMMOCE --exp-name exp_prostate_run

# Step 3: Aggregate performance metrics & AUROC comparison curves
python getResultsV2.py --base-dir chkp/exp_prostate_run

# Step 4: Generate Class Fairness Disparity Figures (Standard & Best-Performing)
python plotClassFairness.py --base-dir chkp/exp_prostate_run --auc-threshold 0.70

# Step 5: Generate Patient-Level Subgroup Misclassification Figures
python fairPatientLevelPlotV2.py --base-dir chkp/exp_prostate_run
```

*(Note: When both `biased` and `BGMMOCE` runs finish, `analysisv3.py` will also automatically invoke `getResultsV2.py` and `plotClassFairness.py` if not called explicitly).*

---

## 2. Environment & Prerequisites

### Python Interpreter
Ensure you are using the dedicated conda environment:
```bash
conda activate nsclc_fairness
# Or use direct interpreter:
/home/trivizakis/anaconda3/envs/nsclc_fairness/bin/python
```

### Dependencies
- **Machine Learning**: `scikit-learn`, `numpy`, `scipy`, `pandas`
- **Fairness & Debiasing**: `aif360`
- **Explainability**: `shap`
- **Survival & Discrimination Analysis**: `lifelines`
- **Visualization**: `matplotlib`, `seaborn`

---

## 3. Dataset Architecture

The dataset is located in `dataset/` with three parallel splits:
1. `dataset/biased/`: Observational radiomics data (unmitigated, biased distribution).
2. `dataset/BGMMOCE/`: Pre-mitigated balanced distribution via Balanced Gaussian Mixture Model Over-sampling with Clustering & Ensembling.
3. `dataset/testing_set/`: Fixed independent testing split used as the evaluation ground truth across all experiments.

### The 6 Experiment Combinations:
The CSV files follow the naming convention: `[region of interest]_[endpoint]_[bias]_[training or testing set].csv`

| Experiment Key | Region of Interest | Endpoint | Sensitive Attribute (Bias) |
| :--- | :---: | :---: | :---: |
| `lesion_uc2_magnetic.field.strength` | Lesion | UC2 (Aggressiveness) | `magnetic_field_strength` |
| `lesion_uc2_manufacturer` | Lesion | UC2 (Aggressiveness) | `manufacturer` |
| `prostate_uc1_magnetic.field.strength` | Prostate | UC1 (Detection) | `magnetic_field_strength` |
| `prostate_uc1_manufacturer` | Prostate | UC1 (Detection) | `manufacturer` |
| `prostate_uc2_magnetic.field.strength` | Prostate | UC2 (Aggressiveness) | `magnetic_field_strength` |
| `prostate_uc2_manufacturer` | Prostate | UC2 (Aggressiveness) | `manufacturer` |

*(Note: For Lesion, only UC2 is present in the dataset as cancer aggressiveness is evaluated on detected lesions).*

---

## 4. The 16 Classifier Families

The benchmark evaluates 16 scikit-learn classifier families supporting sample reweighting:
1. `random_forest` — Random Forest
2. `logistic_regression` — Logistic Regression (L2, L1, ElasticNet)
3. `svm_rbf` — Support Vector Machine (RBF kernel)
4. `svm_linear` — Support Vector Machine (Linear kernel)
5. `svm_poly` — Support Vector Machine (Polynomial kernel)
6. `svm_sigmoid` — Support Vector Machine (Sigmoid kernel)
7. `gradient_boosting` — Gradient Boosting
8. `adaboost` — AdaBoost
9. `mlp16` — Multi-Layer Perceptron (16 Neurons)
10. `mlp32` — Multi-Layer Perceptron (32 Neurons)
11. `mlp64` — Multi-Layer Perceptron (64 Neurons)
12. `mlp128` — Multi-Layer Perceptron (128 Neurons)
13. `decision_tree` — Decision Tree
14. `naive_bayes` — Gaussian Naive Bayes
15. `ridge` — Ridge Classifier
16. `sgd` — Stochastic Gradient Descent (Log Loss)

---

## 5. Model Variants Evaluated per Family (The Triplet)

For each classifier family, three primary configurations are evaluated:
- **Biased (Red, `#d62728`)**: Trained on observational radiomics features.
- **Fair / Reweighting (Green, `#2ca02c`)**: Trained on biased data using generalized sample weights $W(s, y) = \frac{P(s)P(y)}{P(s, y)}$.
- **BGMMOCE (Blue, `#1f77b4`)**: Trained on the pre-mitigated BGMMOCE distribution (no reweighing applied to BGMMOCE).

---

## 6. Output Directory Structure

Each experiment run generates the following hierarchical layout under `chkp/`:

```text
chkp/exp_[date]_[time]/
├── benchmark_summary_global.csv                          # Comprehensive summary across all evaluations
├── benchmark_summary_[experiment].csv                    # Summary per experiment
│
├── fairness_gap_reduction_[experiment].png               # Accuracy gap between subgroups
├── fairness_selection_rate_disparity_[experiment].png    # Selection rate across subgroups
├── fairness_error_rate_by_subgroup_[experiment].png      # Misclassification rate by subgroup
├── patient_misclassification_[experiment].png            # Patient-level error rates
│
├── best_performing_models/                               # Filtered models with AUC > 0.70
│   ├── best_performing_models_summary.csv                # Triplet AUCs and qualification flags
│   ├── fairness_gap_reduction_[experiment].png
│   ├── fairness_selection_rate_disparity_[experiment].png
│   └── fairness_error_rate_by_subgroup_[experiment].png
│
├── [experiment_key]/                                     # e.g., lesion_uc2_magnetic.field.strength/
│   ├── selected_features_biased.txt
│   ├── selected_features_BGMMOCE.txt
│   ├── benchmark_results.json
│   ├── benchmark_summary.csv
│   │
│   ├── random_forest/
│   │   ├── AUROC_comparison.png                          # AUROC curve comparison for this classifier
│   │   ├── metrics_comparison.csv
│   │   ├── biased/
│   │   │   ├── model.pkl
│   │   │   ├── preds.npy
│   │   │   ├── probs.npy
│   │   │   ├── y_test.npy
│   │   │   ├── s_test.npy
│   │   │   ├── random_forest_metrics.json
│   │   │   ├── random_forest_cm.png                      # Confusion Matrix
│   │   │   ├── random_forest_shap_summary.png            # SHAP Beeswarm Plot
│   │   │   └── random_forest_shap_bar.png                # SHAP Bar Importance Plot
│   │   ├── reweighing/                                   # Fair Reweighted variant
│   │   └── BGMMOCE/                                      # BGMMOCE variant
│   └── ... (all 16 families)
```

---

## 7. Command-Line Reference

### `analysisv3.py`
| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--experiment` | `str` | `all` | Specific experiment key to run, or `all` |
| `--roi` | `str` | `all` | Filter by Region of Interest (`lesion`, `prostate`, `all`) |
| `--endpoint` | `str` | `all` | Filter by Endpoint (`uc1`, `uc2`, `all`) |
| `--bias` | `str` | `all` | Filter by Bias (`magnetic.field.strength`, `manufacturer`, `all`) |
| `--training-type`| `str` | `biased` | Training set split: `biased` (base + reweighing) or `BGMMOCE` |
| `--quick-test` | `flag` | `False` | Run fast verification with key representative classifiers |
| `--no-shap` | `flag` | `False` | Disable SHAP explainability for accelerated benchmarking |
| `--exp-name` | `str` | `None` | Custom experiment folder name under `chkp/` |

### `plotClassFairness.py`
| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--base-dir` | `str` | `None` | Path to experiment folder (e.g. `chkp/exp_19.10_13h00m/`) |
| `--exp-name` | `str` | `None` | Experiment folder name under `chkp/` |
| `--auc-threshold`| `float`| `0.70` | Minimum triplet AUC threshold for `best_performing_models` |

### `getResultsV2.py`
| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--base-dir` | `str` | `None` | Path to experiment folder under `chkp/` |
| `--exp-name` | `str` | `None` | Experiment folder name under `chkp/` |

### `fairPatientLevelPlotV2.py`
| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--base-dir` | `str` | `chkp` | Path to experiment folder under `chkp/` |
| `--classifier` | `str` | `random_forest` | Target classifier family for patient-level plots |
