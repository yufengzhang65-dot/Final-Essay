# MCIS Dissertation Experiment Repository

This repository contains the source code, synthetic data, model outputs, result tables, figures, and supporting evidence for the MCIS dissertation project:

**Explainable and Lightweight Anomaly Detection for SME Cloud File-Access Log Monitoring: A Comparative Study of Rule-Based, Machine Learning, and Simple Deep Learning Approaches**

This repository is provided as an experiment evidence package for the dissertation. It keeps the original working folder structure used during development rather than a fully refactored software-package structure. The purpose of this repository is to show that the synthetic audit-log dataset, anomaly-detection experiments, result tables, and dissertation figures were produced through an implemented coding workflow.

## Project purpose

The project investigates how suspicious cloud file-access behaviour can be detected in small and medium-sized enterprise (SME) settings using lightweight and explainable anomaly-detection methods.

The study compares:

* Rules-only baseline
* Isolation Forest
* One-Class SVM
* Local Outlier Factor
* Simple autoencoder

The models are evaluated using the same synthetic SME-style audit-log dataset, the same engineered feature representation, and the same train-validation-test protocol.

## Research question

Which lightweight anomaly-detection approach can best maintain detection coverage while reducing alert burden when detecting suspicious behaviour from SME cloud file-access audit logs?

## Data statement

All data in this repository are synthetic and were generated for research evaluation purposes. No real enterprise tenant data, real user audit logs, confidential files, IP addresses, or organisational access patterns are included.

The synthetic data were created to simulate SME-style cloud file-access audit events, including normal activity and injected suspicious behaviour.

## Attack scenarios

The experiments include four behaviour-oriented suspicious activity categories:

1. **Exfiltration**: burst downloads, high byte volume, or external/public sharing.
2. **Reconnaissance**: broad low-volume browsing across many files.
3. **Privilege misuse**: unusual or repeated permission changes.
4. **Tamper-like activity**: bursts of rename, edit, or delete actions.

## Repository structure

The repository keeps the original project structure used during the dissertation development.

```text
.
├── paper_outputs/
│   ├── figures/
│   └── tables/
├── evidence/
│   ├── 01_project_folder_overview.png
│   ├── 02_paper_outputs_folder.png
│   ├── ...
│   └── evidence_note.txt
├── config.py
├── subject_data.py
├── subject_features.py
├── run_subject_pipeline.py
├── ai_common.py
├── isolation_forest.py
├── ocsvm.py
├── lof.py
├── autoencoder.py
├── run_if_experiments.py
├── run_ocsvm_experiments.py
├── run_lof_experiments.py
├── run_autoencoder_experiments.py
├── make_paper_outputs_clean_names.py
├── normal_logs.csv
├── all_logs_with_attacks.csv
├── train_normal_features.csv
├── val_features.csv
├── test_features.csv
├── *_experiment_summary.csv
├── *_per_intent_summary.csv
├── *_vs_rules_comparison.csv
├── README.md
└── requirements.txt
```

Some filenames may reflect the original development process. The final dissertation tables and figures are located in:

```text
paper_outputs/tables/
paper_outputs/figures/
```

## Main Python files

### `config.py`

This file contains shared configuration values used by the experiment scripts. It defines common paths, filenames, model parameters, feature column names, and output locations. The purpose of this file is to reduce repeated hard-coded values across different scripts.

Typical contents include:

* Input CSV filenames.
* Output CSV filenames.
* Feature column lists.
* Random seed values.
* Model configuration values.
* Directory paths for tables and figures.

### `subject_data.py`

This script creates the synthetic SME-style audit-log dataset. It is responsible for generating the subject-side data environment used in the dissertation.

It defines or supports:

* SME-style user personas, such as office staff, manager, contractor, and IT administrator.
* Synthetic file catalogue metadata, such as department, file owner, file type, and sensitivity.
* Normal cloud file-access events, including view, edit, download, share, delete, rename, and permission-change actions.
* Contextual event attributes, such as timestamp, user role, department, device, city, file sensitivity, share scope, and byte volume.

Main outputs include:

```text
normal_logs.csv
all_logs_with_attacks.csv
val_logs_with_attacks.csv
test_logs_with_attacks.csv
```

These files provide the synthetic event-level logs used for feature engineering and model evaluation.

### `subject_features.py`

This script converts event-level synthetic audit logs into model-ready numerical feature tables.

The engineered features include:

* 10-minute activity features, such as event count, views, downloads, edits, deletes, renames, and files touched.
* 30-minute sharing and permission features, such as shares, external shares, and permission changes.
* Context flags, such as new device, new city, off-hours activity, and weekend activity.
* User-relative byte-volume features, such as `bytes_10m` and `bytes_zscore`.

Main outputs include:

```text
train_normal_features.csv
val_features.csv
test_features.csv
```

These feature files are used by the rules baseline and all AI-based anomaly-detection models.

### `run_subject_pipeline.py`

This script runs the full synthetic data-generation and feature-engineering pipeline.

It combines the workflow from `subject_data.py` and `subject_features.py` to generate:

1. Normal synthetic audit logs.
2. Validation and test logs with injected attack scenarios.
3. Engineered feature tables for training, validation, and testing.

This script provides the main evidence that the dataset was generated through a repeatable coding workflow rather than manually constructed.

### `ai_common.py`

This file contains shared helper functions used by the AI model scripts.

It may include functions for:

* Loading feature CSV files.
* Selecting feature columns.
* Scaling numerical features.
* Computing precision, recall, F1-score, PR-AUC, and alert burden.
* Selecting anomaly thresholds using the validation set.
* Calculating per-attack recall.
* Saving result summaries.

The purpose of this file is to keep the model scripts consistent so that each model is evaluated under the same experimental protocol.

### `isolation_forest.py`

This script implements the Isolation Forest model.

Isolation Forest is used as a lightweight tree-based anomaly detector. It is included to test whether strong and burst-like abnormal behaviours, such as exfiltration or tamper-like activity, can be detected effectively.

The script typically:

1. Loads the engineered feature tables.
2. Fits the Isolation Forest model using normal-only training data.
3. Produces anomaly scores on the validation and test sets.
4. Selects an alert threshold using the validation set.
5. Applies the selected threshold to the test set.
6. Saves model outputs and summary metrics.

Outputs may include:

```text
test_if_results.csv
val_if_results.csv
if_experiment_summary.csv
if_per_intent_summary.csv
if_vs_rules_comparison.csv
```

### `run_if_experiments.py`

This script runs the Isolation Forest experiment workflow. It may call functions from `isolation_forest.py` and repeat the experiment using selected random seeds.

Its purpose is to produce the final Isolation Forest result files used in the dissertation.

### `ocsvm.py`

This script implements the One-Class SVM model.

One-Class SVM is used as a boundary-based anomaly detector. It learns a boundary around normal behaviour and identifies samples outside that boundary as suspicious. In the dissertation results, OCSVM provided the best overall balance between detection coverage and alert burden.

The script typically:

1. Loads training, validation, and test feature tables.
2. Applies feature scaling.
3. Trains the OCSVM model using normal-only training data.
4. Scores validation and test events.
5. Selects a validation-based threshold.
6. Saves the final test-set predictions and summary metrics.

Outputs may include:

```text
test_ocsvm_results.csv
val_ocsvm_results.csv
ocsvm_experiment_summary.csv
ocsvm_per_intent_summary.csv
ocsvm_vs_rules_comparison.csv
```

### `run_ocsvm_experiments.py`

This script runs the One-Class SVM experiment. It was used to compare OCSVM parameter settings and produce the final selected OCSVM result files.

The final dissertation configuration used:

```text
nu = 0.05
gamma = 0.05
```

### `lof.py`

This script implements the Local Outlier Factor model.

LOF is used as a density-based anomaly detector. It is included to test whether locally unusual behaviour can be detected more sensitively than by global or boundary-based approaches.

The script typically:

1. Loads the engineered feature tables.
2. Trains LOF in novelty-detection mode using normal-only training data.
3. Produces anomaly scores.
4. Selects a validation-based threshold.
5. Evaluates the model on the test set.
6. Saves summary outputs.

Outputs may include:

```text
test_lof_results.csv
val_lof_results.csv
lof_experiment_summary.csv
lof_per_intent_summary.csv
lof_vs_rules_comparison.csv
```

### `run_lof_experiments.py`

This script runs the LOF experiment. It was used to compare LOF settings and generate the final LOF output files.

The final dissertation configuration used:

```text
n_neighbors = 15
```

### `autoencoder.py`

This script implements the simple autoencoder model used as the deep-learning comparator.

The autoencoder is a reconstruction-based anomaly detector. It learns to reconstruct normal feature vectors and treats higher reconstruction error as more suspicious.

The architecture used in the dissertation was:

```text
16 → 8 → 4 → 8 → 16
```

The script typically:

1. Loads normal-only training features.
2. Scales features.
3. Trains the autoencoder on normal data.
4. Computes reconstruction error on validation and test data.
5. Selects a threshold using validation-set recall.
6. Evaluates final test-set performance.
7. Saves output files and summary metrics.

Outputs may include:

```text
test_ae_results.csv
val_ae_results.csv
ae_experiment_summary.csv
ae_per_intent_summary.csv
ae_vs_rules_comparison.csv
```

### `run_autoencoder_experiments.py`

This script runs the autoencoder experiment. The filename reflects the original development process. It was used to train and evaluate the simple autoencoder and generate the final autoencoder result files.

### `make_paper_outputs_clean_names.py`

This script creates the final dissertation-ready tables and figures from the experiment output CSV files.

It reads the model summary files and produces cleaned tables and figures for the dissertation. The generated outputs are stored in:

```text
paper_outputs/tables/
paper_outputs/figures/
```

Typical outputs include:

```text
paper_outputs/tables/table_07_overall_test_results.csv
paper_outputs/tables/table_08_per_intent_recall.csv
paper_outputs/figures/figure_03_model_performance_comparison.png
paper_outputs/figures/figure_04_alert_burden_comparison.png
```

### Figure 5 and Figure 6 fixing script

During the final dissertation editing stage, the generated chart outputs for Figure 5 and Figure 6 required correction. The issue was related to the final presentation of chart numbering, labels, and dissertation-ready formatting. The underlying experiment results were not changed; the fix was applied to ensure the final figures matched the dissertation text and table interpretation.

The final corrected versions of Figure 5 and Figure 6 are stored in paper_outputs/tables/ as fig_05_recall_alert_tradeoff_fixed.png and fig_06_per_intent_recall_fixed.png. Earlier generated versions remain in paper_outputs/figures/ as intermediate outputs.

The fixing script, stored under `paper_outputs/tables/` or the project folder, was used to regenerate the corrected versions of the relevant figures.

Final corrected outputs include:

```text
paper_outputs/tables/fig_05_recall_alert_tradeoff_fixed.png
paper_outputs/tables/fig_06_per_intent_recall_fixed.png
```

or the final renamed versions in:

```text
paper_outputs/figures/
```

The corrected figures were used in the final dissertation. The purpose of this correction was to improve presentation consistency, not to modify the underlying model results.

## CSV data and result files

### Synthetic log files

The main synthetic log files include:

```text
normal_logs.csv
all_logs_with_attacks.csv
val_logs_with_attacks.csv
test_logs_with_attacks.csv
```

These files contain event-level synthetic cloud file-access audit logs.

### Feature files

The main feature files include:

```text
train_normal_features.csv
val_features.csv
test_features.csv
```

These files contain model-ready engineered features generated from the event-level logs.

### Model result files

The model-specific result files include outputs such as:

```text
if_experiment_summary.csv
ocsvm_experiment_summary.csv
lof_experiment_summary.csv
ae_experiment_summary.csv
```

These files summarise the overall performance of each model.

Per-attack recall outputs include:

```text
if_per_intent_summary.csv
ocsvm_per_intent_summary.csv
lof_per_intent_summary.csv
ae_per_intent_summary.csv
```

Comparison files include:

```text
if_vs_rules_comparison.csv
ocsvm_vs_rules_comparison.csv
lof_vs_rules_comparison.csv
ae_vs_rules_comparison.csv
```

These files compare each AI model against the rules-only baseline.

## Final dissertation outputs

The final tables and figures used in the dissertation are stored in:

```text
paper_outputs/tables/
paper_outputs/figures/
```

### Final tables

The final dissertation tables include:

```text
table_01_research_question_and_objectives.csv
table_02_user_personas.csv
table_03_attack_scenario_design.csv
table_04_engineered_feature_groups.csv
table_05_detection_methods_compared.csv
table_06_model_configuration.csv
table_07_overall_test_results.csv
table_08_per_intent_recall.csv
table_08_per_intent_recall_long_format.csv
```

### Final figures

The dissertation figures include:

* Figure 1: Overall research pipeline
* Figure 2: Synthetic audit-log generation pipeline
* Figure 3: Overall test-set performance by model
* Figure 4: Alert burden by detection method
* Figure 5: Per-attack recall by detection method
* Figure 6: Detection coverage versus alert burden trade-off

Figures 1 and 2 were manually redrawn in Figma named new_fig_01 and new_fig_02 to make the research workflow and synthetic audit-log generation pipeline clearer for the dissertation. They are based on the implemented coding workflow and data-generation structure.

Figures 3–6 were generated or updated using the experimental outputs and final result tables.

## Known issue and correction for generated figures

Figures 1 and 2 were manually redrawn in Figma as `new_fig_01.png` and `new_fig_02.png` to present the research workflow and synthetic audit-log generation pipeline more clearly. They are based on the implemented workflow and data-generation structure.

Figures 3 and 4 were generated from the experimental result tables.

The initial generated versions of Figure 5 and Figure 6 were later corrected for dissertation presentation. The correction was related to chart presentation, numbering, and clarity, not to changes in the underlying experiment results. The corrected final versions are stored in `paper_outputs/tables/` as:

- `fig_05_recall_alert_tradeoff_fixed.png`
- `fig_06_per_intent_recall_fixed.png`

The correction script is:

- `paper_outputs/tables/fix_figure5_figure6.py`

## How to run the project

Install the required Python packages:

```bash
pip install -r requirements.txt
```

A typical workflow is:

```bash
python run_subject_pipeline.py
python run_if_experiments.py
python run_ocsvm_experiments.py
python run_lof_experiments.py
python run_autoencoder_experiments.py
python make_paper_outputs_clean_names.py
```

The corrected Figure 5 and Figure 6 versions can be regenerated using:

```bash
python paper_outputs/tables/fix_figure5_figure6.py
```

The final dissertation outputs are stored in paper_outputs/. The final corrected versions of Figure 5 and Figure 6 are stored in paper_outputs/tables/.

## Evidence folder

The `evidence/` folder contains screenshots and notes used to support experiment transparency. These files are not required to run the code. They are included to show the project folder structure, generated synthetic CSV files, engineered feature tables, final output tables, final figures, and the local Python environment used during the dissertation experiments.

The folder includes:

- project folder overview screenshot
- paper output folder screenshot
- final tables folder screenshot
- final figures folder screenshot
- raw synthetic log preview
- engineered feature table preview
- overall test-result table preview
- per-attack recall table preview
- Python environment screenshot
- `evidence_note.txt`

## Requirements

The main Python libraries used in this project include:

```text
pandas
numpy
scikit-learn
matplotlib
openpyxl
tensorflow
```

If the autoencoder implementation uses PyTorch rather than TensorFlow, replace `tensorflow` with `torch` in `requirements.txt`.

## Experiment environment

The experiments were conducted locally on a Windows machine.

Example environment information:

```text
Operating system: Windows 11
Python version: 3.13.3
IDE/editor: Visual Studio Code
Main libraries: pandas, numpy, scikit-learn, matplotlib, openpyxl, tensorflow/torch
Data type: synthetic SME-style cloud file-access audit logs
Real enterprise data used: No
Real user audit logs used: No
```

Screenshots of the project folder, output files, and experiment evidence are stored in:

```text
evidence/
```

## Reproducibility note

This repository is intended to support transparency and reproducibility at the dissertation level. It shows the implemented scripts, synthetic data files, intermediate outputs, final result tables, and final figures used in the dissertation.

Because this repository keeps the original experiment folder structure, some scripts may depend on local relative paths and filenames. The main purpose is to provide evidence of the conducted experimental workflow and final outputs rather than to provide a fully packaged software tool.

## Academic note

The results should be interpreted as comparative evidence within a controlled synthetic testbed. They should not be interpreted as direct deployment performance on real enterprise cloud audit logs.

The dissertation findings show that One-Class SVM achieved the strongest overall balance between detection coverage and operational alert burden in the controlled experimental setting, while LOF achieved high recall but produced excessive alerts.
