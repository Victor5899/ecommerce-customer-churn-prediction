#  E-Commerce Customer Churn Prediction & Analytics Platform

An end-to-end machine learning project that predicts customer churn from a real-world e-commerce dataset. It includes duplicate-aware data preparation, a leakage-resistant train/validation/test split, hyperparameter tuning of four candidate models, validation-based decision-threshold selection, and a Streamlit dashboard for **batch** churn scoring of uploaded customer files.

---

##  Project Overview

Customer churn is one of the biggest challenges faced by e-commerce businesses. This project identifies customers who are likely to stop purchasing so that retention efforts can be targeted at them.

What is implemented:

- Data loading and target validation
- Duplicate-record audit and deduplication before splitting
- Missing-value imputation, categorical encoding and feature scaling inside a single scikit-learn pipeline
- Hyperparameter tuning and comparison of four models
- Decision-threshold selection on a validation set
- Final evaluation on an untouched test set
- Feature importance plot
- Streamlit dashboard for batch predictions from CSV/XLSX files
- Automated tests (pytest)

No feature engineering is performed: the model uses the 18 original features as provided.

---

# Dataset

**File:** `data/E Commerce Dataset.xlsx`. It has two sheets: `Data Dict` (column descriptions) and `E Comm` (the data). Only `E Comm` is used.

- **Rows:** 5,630 customers (5,074 after removing exact duplicates, see below)
- **Target:** `Churn` (1 = churned, 0 = retained), churn rate 16.8%
- **Identifier:** `CustomerID` (excluded from the features)
- **Features (18):**
  - Numeric (13): `Tenure`, `CityTier`, `WarehouseToHome`, `HourSpendOnApp`, `NumberOfDeviceRegistered`, `SatisfactionScore`, `NumberOfAddress`, `Complain`, `OrderAmountHikeFromlastYear`, `CouponUsed`, `OrderCount`, `DaySinceLastOrder`, `CashbackAmount`
  - Categorical (5): `PreferredLoginDevice`, `PreferredPaymentMode`, `Gender`, `PreferedOrderCat`, `MaritalStatus`

To train on your own data, provide an `.xlsx` file with an `E Comm` sheet containing `Churn` (0/1), the 18 feature columns above (exact names), and optionally `CustomerID`.

### Known data-quality issues

- **Duplicate records.** 556 records repeat another record's 18 feature values *and* churn label exactly; only `CustomerID` differs. All 556 duplicate groups are pairs, none have conflicting labels, and 546 of the pairs have CustomerIDs exactly 1,470 apart. This points to a copied block of records rather than distinct customers who happen to look identical, so each pair is collapsed to the record with the lower CustomerID before splitting. Removed records are listed in `outputs/removed_duplicates.csv`, and the full audit is in `outputs/data_quality_report.json`.
- **Inconsistent category labels** are kept as-is (not merged): `Phone` vs `Mobile Phone`, `CC` vs `Credit Card`, `COD` vs `Cash on Delivery`, and `Mobile` vs `Mobile Phone` in the order category.
- **Missing values** in 7 numeric columns (about 250–310 each) are imputed by the pipeline.

---

#  Tech Stack

| Area | Tools |
|---|---|
| Language | Python (tested with 3.9) |
| Data | Pandas, NumPy, openpyxl |
| Machine learning | Scikit-Learn, XGBoost |
| Visualization | Matplotlib |
| Dashboard | Streamlit |
| Model storage | Joblib |
| Testing | pytest |

---

# Machine Learning Pipeline

```
E Comm sheet (5,630 rows)
      │
      ▼
Target validation + duplicate audit
      │
      ▼
Deduplication (556 identical records removed → 5,074 rows)
      │
      ▼
Group-aware stratified split: train 64% / validation 16% / test 20%
(identical feature records can never span two partitions)
      │
      ▼
Per candidate model, on the training partition only:
  Pipeline = [median / most-frequent imputation → scaling / one-hot encoding → classifier]
  RandomizedSearchCV (20 iterations, 3-fold StratifiedGroupKFold, scoring = PR-AUC)
      │
      ▼
Model selection: highest mean cross-validated PR-AUC
      │
      ▼
Threshold selection on the validation partition
      │
      ▼
Single final evaluation on the test partition
      │
      ▼
Saved pipeline + metadata → Streamlit dashboard
```

Preprocessing is fitted inside the pipeline, so imputation and scaling statistics are learned from training folds only and are saved together with the classifier in `best_model.joblib`.

---

#  Models Compared

Logistic Regression, Random Forest, Gradient Boosting and XGBoost are each tuned with `RandomizedSearchCV`. **The model is selected solely by the mean cross-validated PR-AUC (average precision) on the training partition**; the validation and test sets play no part in model selection.

Results from the latest run (`outputs/model_comparison.csv`):

| Model | CV PR-AUC (mean ± std) | Validation PR-AUC | Validation ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.690 ± 0.035 | 0.706 | 0.887 |
| **Random Forest (selected)** | **0.873 ± 0.018** | 0.932 | 0.981 |
| Gradient Boosting | 0.815 ± 0.029 | 0.833 | 0.952 |
| XGBoost | 0.865 ± 0.022 | 0.940 | 0.975 |

Random Forest and XGBoost are close. Random Forest wins under the pre-defined selection criterion. The comparison file also contains each model's validation precision, recall and F1 at its own validation-tuned threshold. Those numbers are optimistic, because the threshold was tuned on the same data, and are intended only for comparison. Test metrics are computed only for the selected model.

Selected hyperparameters: `n_estimators=500`, `max_depth=16`, `min_samples_split=2`, `min_samples_leaf=1`, `class_weight="balanced_subsample"`, seed 42.

---

# Decision Threshold

Churn is predicted when `probability >= threshold`. The threshold is chosen on the **validation** partition by scanning 0.05–0.90 in steps of 0.01 and maximising the F2 score (which weights recall more than precision) subject to precision ≥ 0.55. Ties go to the lower threshold. The latest run selected **0.25**.

The threshold is stored in `outputs/metrics.json` and used for both the reported test metrics and the dashboard predictions. The dashboard also checks that `metrics.json` belongs to the loaded model file (SHA-256). If the metadata is missing, invalid or stale, the dashboard shows an error, still displays probabilities, but does not classify customers and does not show the saved evaluation results.

---

#  Model Performance

**Random Forest** on the held-out test set: 1,015 customers, 168 churners, threshold 0.25.

| Metric | Score |
|---------|--------|
| Accuracy | **94.2%** |
| ROC-AUC | **0.987** |
| PR-AUC (average precision) | **0.945** |
| Precision (churn) | **76.3%** |
| Recall (churn) | **94.0%** |
| F1 Score (churn) | **84.3%** |

Confusion matrix: 158 churners caught, 10 missed, 49 false alarms, 798 correctly predicted as staying.

For reference, the same model at the default threshold of 0.5 has 93.9% precision and 73.2% recall. Saved under `test_metrics_reference_threshold_0_5`; not used anywhere else.

> Earlier versions of this README reported ROC-AUC 0.989 / PR-AUC 0.962 / recall 92.6%. Those numbers came from data with duplicate records that crossed the train/test split (158 test rows had an identical copy in training), and the dashboard did not apply the reported threshold. The figures above come from the corrected pipeline.

---

# Streamlit Dashboard

A multi-page Streamlit analytics app (sidebar navigation) built on the saved artifacts in `outputs/`. It never retrains the model. Charts are interactive [Altair](https://altair-viz.github.io/) charts (Altair ships with Streamlit, so no extra dependency) with tooltips on every bar. The sidebar shows the loaded model, its decision threshold, and whether `metrics.json` matches the model file (SHA-256).

Historical churn is always calculated from the actual `Churn` labels. Model predictions are summarised separately and are never treated as outcomes.

| Page | Purpose |
|---|---|
| **Executive Overview** | Historical record count and actual churn rate (raw and de-duplicated), next to the number and share of customers predicted to churn in the current scoring batch. Shows the churn distribution, actual churn rate by tenure, satisfaction score, complaint status and order count, and a short list of observations computed from the data. |
| **Customer Analytics** | Explore any model feature for the historical dataset or the current scored batch. Filters cover tenure, city tier, order category and complaints, with missing values optionally included. Each feature gets a distribution chart (split by actual outcome when labels exist), an actual churn rate by segment, and median feature values for retained vs churned customers. Churn patterns are shown only when the source has a complete 0/1 `Churn` column. |
| **Churn Prediction** | Batch scoring of CSV/XLSX uploads (see below). Shows a prediction summary for the current upload, configurable Low/Medium/High risk bands, a probability histogram with the decision threshold marked, and a risk list you can filter and sort. Downloads cover the full results or the filtered view. |
| **Model Performance** | Saved held-out test metrics at the tuned threshold (precision, recall, F1, ROC-AUC, PR-AUC, accuracy) with an interactive confusion matrix and a reference comparison at 0.5. Cross-validation and validation results are on separate tabs, along with the candidate comparison and the saved plots. Further tabs show permutation feature importance and probability calibration (see below). Missing or out-of-date artifacts produce a warning, and if the metadata fails its checks the results are hidden. |
| **Data Quality** | Training-time findings: dataset size, missing values per column, the duplicate-removal report with the removed records (downloadable), the train/validation/test split, alternative category spellings found in the data, and known limitations. A separate tab holds checks on the current upload: missing values, duplicate rows and IDs, unseen categories and label availability. |
| **Retention Intelligence** | Retention planning for the current scored batch (see [Retention Intelligence](#retention-intelligence)): a ranked prioritisation table, rule-based suggested actions with reasons, a budget planner, a what-if scenario comparison and a CSV action plan. It reuses the batch scores and never re-runs the model. |

### Batch prediction

The dashboard scores a **batch** of customers from an uploaded file. It has no single-customer form and no real-time API.

- Upload customer data as CSV or XLSX. For workbooks, the `E Comm` sheet is used when present (so the original dataset file works), a single-sheet workbook is read directly, and for other multi-sheet workbooks you choose the sheet.
- Required columns are validated before prediction. Missing columns, non-numeric values in numeric columns, empty files and unreadable files produce an error message instead of a traceback.
- `CustomerID` and `Churn` are never used as features. If the upload has a complete `Churn` column, its actual churn rate is reported separately from the predictions. Extra columns are listed, ignored for prediction, and kept in the output.
- Missing values are imputed by the trained pipeline. Category values not seen during training are reported and encoded as "none of the known categories".
- Results keep the original rows, order and identifiers, and add `Churn Prediction` (0/1 at the configured threshold), `Churn Probability` (%) and `Risk Segment`.
- Risk bands are for presentation only. By default the Medium band starts at the decision threshold, and moving the boundaries never changes `Churn Prediction`.
- The last scored batch stays available on the other pages for the rest of the browser session.

### Dataset compatibility checker

Before scoring, every upload is checked against the model's 18 required input features (`src/compatibility.py`) and given one of three statuses:

| Status | Meaning | What happens |
|---|---|---|
| **Ready to score** | Every required column is present with usable values | Scoring works exactly as before |
| **Needs mapping/preprocessing** | Safe changes would make the file scorable | The changes are listed with a before/after preview. Nothing is applied until you tick **Apply these changes and continue**. Only the session copy changes, never your file |
| **Incompatible** | Required columns are missing, ambiguous, duplicated or have invalid values | The reasons are shown and scoring is blocked |

**Checks performed:**
- **Columns:** required columns that are missing, extra columns, and duplicate columns. Duplicates are detected from the raw header, because pandas would otherwise rename a repeated `Tenure` to `Tenure.1`. A duplicated model, `CustomerID` or `Churn` column blocks scoring; a duplicated extra column only produces a warning.
- **Values:** non-numeric, infinite or true/false values in numeric features, and categories not seen in training.
- **Data quality:** missing values, columns with no values at all, rows with no feature values, negative values, exact duplicate rows, and missing or duplicated `CustomerID`s.

**Changes that may be proposed (all of them need your confirmation):**
- *Renames* where the name differs from a required column only in capitalisation, spacing or punctuation (for example `tenure`, `Days Since Last Order`, `customer_id`), or is a listed spelling variant of the same feature (for example `PreferredOrderCat` for the dataset's `PreferedOrderCat`, or `NumberOfAddresses`).
- *Yes/No* or *True/False* text converted to 1/0 in the binary `Complain` and `Churn` columns. Missing values stay missing.
- *Category capitalisation*, for example `mobile phone` matched to the known `Mobile Phone`.

`Churn` is optional. If it is a complete 0/1 column, it is used only to report actual churn in the file. If it is incomplete or has other values, you get a warning and it is ignored. It is never a prediction input.

**Not proposed, by design:**
- Columns with different names are never matched just because the names look similar. `Tenures` or `MonthsAsCustomer` are not mapped to `Tenure`, and `Complaint` is not mapped to `Complain`, because the name alone does not show that the data is the same. Similar names are shown as hints only.
- Number formats are not guessed. `1,234.5` could use either a thousands or a decimal separator.
- Missing required columns are never filled in. Partially missing values are imputed by the fitted pipeline, as before.

The model, metadata and historical dataset are cached (`st.cache_resource` / `st.cache_data`) and reloaded only when their files change. Uploaded files are parsed once per content hash.

### Screenshot checklist

The images in `screenshots/` predate this dashboard and the threshold/deduplication fixes. Retake them from a local run on the real dataset and saved artifacts:

- [ ] Executive Overview: KPI cards, churn distribution, segment charts
- [ ] Customer Analytics: filters expanded, `Tenure` and `CashbackAmount` charts
- [ ] Churn Prediction: original workbook scored, risk histogram and filtered risk list
- [ ] Model Performance: test-set tab with confusion matrix; cross-validation tab
- [ ] Data Quality: training-data tab and current-upload tab
- [ ] Retention Intelligence: prioritisation table, budget planner and scenario comparison

---

#  Feature Importance

![Feature Importance](outputs/feature_importance.png)

The Random Forest's impurity-based importances rank `Tenure` highest by a wide margin, followed by `CashbackAmount`, `WarehouseToHome`, `Complain`, `DaySinceLastOrder`, `NumberOfAddress`, `OrderAmountHikeFromlastYear` and `SatisfactionScore`. These show which features the model relies on, not the direction of their effect or a causal relationship.

---

# Project Structure

```
ecommerce-customer-churn-prediction
│
├── app.py                     Streamlit entry point: cached loading, model checks, navigation
├── .streamlit/config.toml     Dashboard theme
├── README.md
├── requirements.txt           Runtime dependencies
├── requirements-dev.txt       + pytest
├── pytest.ini
├── LICENSE
│
├── data
│   ├── E Commerce Dataset.xlsx
│   └── generate_customers.py  Legacy synthetic generator (different schema, not used)
│
├── src
│   ├── utils.py               Loading, duplicate audit/deduplication, grouped split, JSON helpers
│   ├── inference.py           File ingestion, input validation, threshold policy, scoring
│   ├── dashboard.py           Dashboard calculations (actual vs predicted churn, segments, quality)
│   ├── charts.py              Altair chart builders
│   ├── dashboard_pages.py     Streamlit page renderers
│   ├── explainability.py      Permutation importance, impurity mapping, per-customer Shapley values
│   ├── calibration.py         Calibrators, Brier/log loss/reliability, calibration artifact checks
│   ├── explain_calibrate.py   Builds the explainability and calibration artifacts (no retraining)
│   ├── retention.py           Retention planning rules: bands, suggestions, ranking, budgets, scenarios
│   ├── compatibility.py       Upload compatibility checker and confirmed, previewed fixes
│   └── train_models.py        Training, model selection, threshold tuning, evaluation, artifacts
│
├── tests                      pytest suite (synthetic fixtures + dashboard smoke tests)
│
├── outputs                    Generated by training
│   ├── best_model.joblib
│   ├── metrics.json
│   ├── classification_report.txt
│   ├── model_comparison.csv
│   ├── data_quality_report.json
│   ├── removed_duplicates.csv
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── pr_curve.png
│   ├── roc_curve.png
│   ├── explainability.json
│   ├── calibration.json
│   └── calibrator.joblib
│
└── screenshots
```

---

# ⚡ Installation

```bash
git clone https://github.com/Victor5899/ecommerce-customer-churn-prediction.git
cd ecommerce-customer-churn-prediction

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements-dev.txt   # or requirements.txt without pytest
```

The saved model is a joblib/pickle file. Load it only with the pinned scikit-learn version, and only from a trusted source.

---

# ▶ Run Model Training

```bash
python src/train_models.py --input "data/E Commerce Dataset.xlsx"
```

This takes under a minute on a laptop and overwrites the files in `outputs/`. Useful options:

| Option | Default | Meaning |
|---|---|---|
| `--outdir` | `outputs` | Where artifacts are written |
| `--seed` | `42` | Random seed for splits, CV and models |
| `--test-size` / `--val-size` | `0.2` / `0.2` | Test fraction; validation fraction of the remainder |
| `--n-iter` / `--cv-folds` | `20` / `3` | Randomized search budget and CV folds |
| `--n-jobs` | `-1` | Parallel jobs for the search |
| `--models` | `logreg,rf,gb,xgb` | Candidate subset |
| `--conflict-policy` | `keep_grouped` | Identical features with conflicting labels: keep both and keep them in one partition (`keep_grouped`), remove them (`drop`), or stop (`error`) |
| `--beta` / `--min-precision` | `2.0` / `0.55` | Threshold objective |

Quick smoke run:

```bash
python src/train_models.py --input "data/E Commerce Dataset.xlsx" --outdir /tmp/churn_smoke --models logreg,rf --n-iter 2
```

---

# ▶ Run Streamlit Dashboard

```bash
python -m streamlit run app.py
```

Open http://localhost:8501 and use the sidebar to switch pages. The app needs `outputs/best_model.joblib` and `outputs/metrics.json` (run training first if they are missing). The historical pages also read `data/E Commerce Dataset.xlsx`.

---

# ▶ Run Tests

```bash
python -m pytest
```

Unit tests use small synthetic datasets and temporary directories; no network access is needed. `tests/test_app.py` additionally runs the dashboard against the trained artifacts in `outputs/` and is skipped if they are missing.

---

# Explainability and Calibration

These artifacts are built from the **saved** model without retraining it. `best_model.joblib` and `metrics.json` are left unchanged:

```bash
python src/explain_calibrate.py --input "data/E Commerce Dataset.xlsx"   # about 10 seconds
```

The script rebuilds the train/validation/test partitions from the settings in `metrics.json`. It stops unless the partition sizes and the recorded validation and test PR-AUC are reproduced exactly.

- **Permutation importance** (Model Performance → Feature importance) shuffles each raw input feature on the **validation partition** 10 times and records the drop in PR-AUC, reporting the mean and standard deviation. Shuffling happens before preprocessing, so one-hot-encoded categories count as a single feature. The results are compared with the Random Forest's impurity importance, with one-hot columns summed back to their source feature. Importance shows how much the model relies on a feature, not what causes churn.
- **Customer explanations** (Churn Prediction → Explain individual customers) estimate interventional Shapley values for up to 5 scored customers at a time. Each estimate uses antithetic permutation sampling over a 100-customer sample from the training partition; this is a small numpy implementation, so the `shap` package is not needed. Contributions are in percentage points of the model's uncalibrated churn score. They add up exactly to the customer's score minus the reference value (the mean score of the background sample), and the dashboard checks this against the model's own prediction. Sampling standard errors are shown. Results are cached per customer. The contributions describe the model's behaviour, not causes.
- **Calibration** (Model Performance → Calibration) compares no calibration, sigmoid calibration (logistic regression on the score's log-odds) and isotonic calibration. The method with the lowest out-of-fold Brier score in 5-fold cross-validation on the **validation partition** is selected and refitted on that partition. The test partition is used once, for the final comparison: Brier score, log loss, expected calibration error, ROC-AUC, PR-AUC and reliability diagrams. With a calibrator available, predictions add a `Calibrated Churn Probability` column. `Churn Prediction`, `Churn Probability`, the risk bands and the decision threshold still use the uncalibrated score, so labels and customer ranking are unchanged.

| Test partition (1,015 customers) | Uncalibrated | Sigmoid-calibrated |
|---|---|---|
| Brier score | 0.0445 | 0.0317 |
| Log loss | 0.1697 | 0.1071 |
| Expected calibration error (10 bins) | 0.0830 | 0.0125 |
| ROC-AUC | 0.9874 | 0.9874 |
| PR-AUC | 0.9446 | 0.9446 |

The Random Forest's raw scores are under-confident; the calibrator's log-odds slope is about 2.5. A raw score at the 0.25 threshold corresponds to a calibrated probability of about 20%. Both the threshold and the calibrator were fitted on the validation partition.

The dashboard checks both artifacts against the loaded model's SHA-256 and the calibrator file's SHA-256. If an artifact is missing, belongs to a different model, or was built from a different dataset file, the dashboard shows a warning and runs without it.

---

# Retention Intelligence

The **Retention Intelligence** page turns the current scored batch into an outreach plan. The planning logic lives in `src/retention.py` as pure pandas functions, kept apart from the ML pipeline and the Streamlit renderers. It reads the columns already produced by scoring, so it never refits or re-scores the model, never changes the decision threshold, and writes no files. Score a file on **Churn Prediction** first.

**Planning risk bands.** These are Low / Medium / High bands on the uncalibrated model score, with configurable boundaries (default: Medium starts at the decision threshold, High at 60%). They are independent of `Churn Prediction`: moving them never changes the official 0/1 label.

**Suggested actions.** The rules are configurable and apply in this order. The first rule that applies sets the action, and the other applicable actions are listed next to it:

| Order | Action | Applies when (default) |
|---|---|---|
| 1 | Complaint resolution follow-up | `Complain = 1` |
| 2 | Onboarding assistance | `Tenure ≤ 2` months |
| 3 | Service-quality follow-up | `SatisfactionScore ≤ 2` |
| 4 | Re-engagement message | `DaySinceLastOrder ≥ 10` |
| 5 | Manual review | no rule applies and a rule feature is missing |
| 6 | General check-in | no rule applies and every rule feature is observed |

Each suggestion records the action, a reason quoting the observed values, the condition that applied, and the label "Heuristic rule; not a validated treatment". Missing values never trigger a rule. No rule suggests discounts.

**Ranking policies.** Rank 1 means "contact first", and ties always fall back to file order:
- *Priority ranking*: planning band (High first), then the number of rules that applied, then model score.
- *Highest score first*: model score only.

Calibrated probabilities are shown when the validated calibrator is loaded. They are a monotonic remap of the score, so they are not used for ranking and would not change it.

**Budget planner.** You set the total budget, a cost per contact (one cost for everyone, or a separate cost per suggested action), an optional contact limit, a minimum score, the lowest eligible band, optionally "official predicted churners only", and the strategy. Eligible customers are taken strictly in rank order while the running cost stays within budget and under the contact limit. Selection stops at the first customer who does not fit, so a lower-ranked customer is never chosen instead of a higher-ranked one. The planner reports eligible and selected counts, estimated cost, unspent budget, selected customers per band, and the observed characteristics of selected vs not-selected customers. The default budget (1,000) and cost (10) are placeholders and use no particular currency.

**What-if scenarios.** This is an editable table of scenarios. Each row sets a budget, a cost per customer, a contact limit, a minimum score, the lowest band and a strategy. The presets (*Conservative* = High band only, *Moderate* = Medium and above, *Broad* = all bands) share the planner's budget and cost. The comparison table and charts show selection counts, cost against budget, the band mix and the characteristics of the selected customers. Invalid rows are skipped with a warning.

**Export.** The action plan CSV has a fixed column layout. The full plan is in original file order; the selected-customers file is in selection order. Every row has `Row ID` (its position in the upload) and an `Identifier Type` column. When `CustomerID` is missing or duplicated, `Row ID` is the only identifier, and it is not stable across files.

**What it does not do.** The page does not estimate customers retained, revenue saved, customer lifetime value or intervention effectiveness. The dataset contains no outcomes of past outreach, so none of these can be validated. A scenario that contacts more customers is not more effective just because it reaches more people.

---

# Output Files

| File | Contents |
|---|---|
| `best_model.joblib` | Fitted preprocessing + classifier pipeline |
| `metrics.json` | Seed, environment, data/deduplication summary, split sizes, feature schema, selection criterion and parameters, CV score, threshold and objective, validation and test metrics, all candidates, model SHA-256 |
| `classification_report.txt` | Human-readable summary and per-class test report at the threshold |
| `model_comparison.csv` | CV and validation metrics for every candidate |
| `data_quality_report.json` | Duplicate audit and deduplication result |
| `removed_duplicates.csv` | Removed records and the CustomerID kept in their place |
| `*.png` | Test-set ROC, PR (with the threshold marked) and confusion matrix; feature importance |
| `explainability.json` | Permutation importance on the validation partition, impurity-importance comparison, 100-row training background sample; model and dataset SHA-256 |
| `calibration.json` | Calibration method selection (validation CV), test-set Brier/log loss/ECE/ROC-AUC/PR-AUC and reliability bins, threshold mapping; model and calibrator SHA-256 |
| `calibrator.joblib` | The selected calibrator (written only if a method other than `none` is selected and passes its checks) |

---

# Limitations

- The final model is fitted on the training partition only (64% of the deduplicated data); the validation partition is reserved for threshold tuning.
- Probabilities are not calibrated; class weighting shifts them upward. Use them for ranking and with the configured threshold rather than as exact likelihoods.
- Feature importance is impurity-based (training data), not SHAP or permutation importance.
- Gradient Boosting and XGBoost are trained without class weighting, while Logistic Regression and Random Forest use it.
- Inconsistent category labels are not harmonised.
- The random split assumes rows are independent snapshots; no time-based validation is possible because the dataset has no dates.
- No REST API, database, authentication, deployment, monitoring or automated retraining.
- Retention suggestions are fixed heuristic rules on four observed features. They have not been tested as treatments, and the planner does not estimate retained customers, revenue, customer lifetime value or uplift. Costs and budgets are user assumptions.
- The compatibility checker recognises only formatting differences and a short list of spelling variants of the training column names. Other schemas must be renamed by hand, and unit or scale differences (for example tenure in days instead of months) cannot be detected.
- Retention plans exist only in the browser session. Without a unique `CustomerID`, rows can be identified only by their position in the uploaded file.

---

# Future Improvements

- SHAP explainability
- Probability calibration and cost-based threshold selection
- Customer Lifetime Value Prediction
- Real-time REST API
- Docker Deployment
- Cloud Deployment (AWS/Azure)
- Automated Model Retraining and drift monitoring

---

# Author

**Victor**
---
