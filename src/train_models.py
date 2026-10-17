import argparse
import os
import platform
import tempfile
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, fbeta_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from inference import (
    METADATA_SCHEMA_VERSION, apply_threshold, file_sha256, positive_class_proba, schema_from_model,
)
from utils import (
    CONFLICT_POLICIES, DATA_SHEET, ID_COL, TARGET_COL, count_shared_feature_records, deduplicate,
    default_file_mode, feature_row_hash, load_raw_dataset, split_xy, to_jsonable, write_json_atomic,
)

SELECTION_CRITERION = "mean_cv_average_precision"
THRESHOLD_GRID = np.round(np.arange(0.05, 0.9 + 1e-9, 0.01), 2)
DEFAULT_BETA = 2.0
DEFAULT_MIN_PRECISION = 0.55
DEFAULT_THRESHOLD_REFERENCE = 0.5
MODEL_NAMES = {
    "logreg": "Logistic Regression",
    "rf": "Random Forest",
    "gb": "Gradient Boosting",
    "xgb": "XGBoost",
}


def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def build_preprocessor(X: pd.DataFrame):

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    return pre, num_cols, cat_cols


def model_spaces(seed: int = 42):
    return {
        "logreg": (
            LogisticRegression(max_iter=3000, class_weight="balanced"),
            {
                "clf__C": np.logspace(-2, 2, 12),
                "clf__penalty": ["l2"],
                "clf__solver": ["lbfgs", "saga"]
            }
        ),
        "rf": (
            RandomForestClassifier(random_state=seed, class_weight="balanced_subsample"),
            {
                "clf__n_estimators": [200, 300, 500],
                "clf__max_depth": [None, 8, 12, 16],
                "clf__min_samples_split": [2, 5, 10],
                "clf__min_samples_leaf": [1, 2, 4]
            }
        ),
        "gb": (
            GradientBoostingClassifier(random_state=seed),
            {
                "clf__n_estimators": [150, 200, 300],
                "clf__learning_rate": [0.03, 0.05, 0.1],
                "clf__max_depth": [2, 3],
                "clf__subsample": [0.7, 0.85, 1.0]
            }
        ),
        "xgb": (
            XGBClassifier(
                random_state=seed,
                eval_metric="logloss"
            ),
            {
                "clf__n_estimators": [100, 200, 300],
                "clf__max_depth": [3, 5, 7],
                "clf__learning_rate": [0.01, 0.05, 0.1],
                "clf__subsample": [0.8, 1.0],
                "clf__colsample_bytree": [0.8, 1.0]
            }
        ),
    }


def randomized_cv(X, y, groups, pre, model, params, seed=42, n_iter=20, cv_folds=3, n_jobs=-1):
    """Tune one candidate with group-aware stratified CV; identical records never span folds."""
    pipe = Pipeline([("pre", pre), ("clf", model)])
    search = RandomizedSearchCV(
        pipe,
        param_distributions=params,
        n_iter=min(n_iter, len(ParameterGrid(params))),
        cv=StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed),
        n_jobs=n_jobs,
        random_state=seed,
        scoring="average_precision",
        refit=True,
        verbose=0,
    )
    search.fit(X, y, groups=groups)
    return search


def pick_threshold(y_true, proba, beta=DEFAULT_BETA, min_precision=DEFAULT_MIN_PRECISION,
                   thresholds=THRESHOLD_GRID):
    """Choose the threshold maximising F-beta subject to a minimum precision.

    Ties go to the lowest threshold (higher recall). If no threshold reaches
    ``min_precision``, the unconstrained F-beta maximum is returned with
    ``constraint_satisfied=False`` so the fallback is visible in the metadata.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    b2 = beta * beta
    candidates = []
    for thr in thresholds:
        pred = apply_threshold(p, thr)
        tp = int(((y == 1) & (pred == 1)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        denom = b2 * precision + recall
        fbeta = (1 + b2) * precision * recall / denom if denom else 0.0
        candidates.append({"value": float(thr), "fbeta": fbeta, "precision": precision, "recall": recall})

    eligible = [c for c in candidates if c["precision"] >= min_precision]
    best = max(eligible or candidates, key=lambda c: c["fbeta"])
    return {
        **best,
        "beta": float(beta),
        "min_precision": float(min_precision),
        "constraint_satisfied": bool(eligible),
        "grid": {"start": float(thresholds[0]), "stop": float(thresholds[-1]),
                 "n_values": int(len(thresholds))},
    }


def classification_metrics(y_true, proba, threshold, beta=DEFAULT_BETA) -> dict:
    y = np.asarray(y_true).astype(int)
    pred = apply_threshold(proba, threshold)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y)) == 2
    return {
        "threshold": float(threshold),
        "n_samples": int(len(y)),
        "n_positive": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, proba)) if both_classes else None,
        "pr_auc": float(average_precision_score(y, proba)) if both_classes else None,
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "beta": float(beta),
        "fbeta": float(fbeta_score(y, pred, beta=beta, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def plot_roc(y_true, proba, name, outpath):
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc = roc_auc_score(y_true, proba)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0,1], [0,1], "--")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (test set)")
    ax.legend()
    fig.tight_layout(); fig.savefig(outpath, dpi=160); plt.close(fig)


def plot_pr(y_true, proba, name, outpath, threshold=None):
    prec, rec, _ = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
    if threshold is not None:
        pred = apply_threshold(proba, threshold)
        ax.scatter([recall_score(y_true, pred, zero_division=0)],
                   [precision_score(y_true, pred, zero_division=0)],
                   color="red", zorder=3, label=f"Decision threshold {threshold:.2f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (test set, positive: churn)")
    ax.legend()
    fig.tight_layout(); fig.savefig(outpath, dpi=160); plt.close(fig)


def plot_confusion(y_true, y_pred, outpath, threshold=None):
    fig, ax = plt.subplots(figsize=(7,7))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred, labels=[0, 1])).plot(ax=ax, colorbar=False)
    suffix = f" (test set, threshold {threshold:.2f})" if threshold is not None else ""
    ax.set_title(f"Confusion Matrix{suffix}")
    fig.tight_layout(); fig.savefig(outpath, dpi=160); plt.close(fig)


def plot_feature_importance(pipe, num_cols, cat_cols, outpath):
    ohe = pipe.named_steps["pre"].named_transformers_["cat"]
    num_names = num_cols
    cat_names = list(ohe.get_feature_names_out(cat_cols)) if hasattr(ohe, "get_feature_names_out") else []
    feature_names = num_names + cat_names

    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_).ravel()
    else:
        importances = np.zeros(len(feature_names))

    idx = np.argsort(importances)[-20:]
    imp = importances[idx]
    names = np.array(feature_names)[idx]

    fig, ax = plt.subplots(figsize=(10,6))
    ax.barh(range(len(imp)), imp)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(names)
    ax.set_title("Top Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout(); fig.savefig(outpath, dpi=160); plt.close(fig)


def _partition_summary(y) -> dict:
    return {"n_rows": int(len(y)), "n_positive": int(y.sum()), "churn_rate": float(y.mean())}


def _dump_atomic(obj, path):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp")
    os.close(fd)
    try:
        dump(obj, tmp)
        os.chmod(tmp, default_file_mode())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _format_report(metrics, sk_report) -> str:
    sel, thr = metrics["model_selection"], metrics["threshold"]
    dedup = metrics["data"]["deduplication"]
    val, test = metrics["validation_metrics"], metrics["test_metrics"]
    lines = [
        f"Best model: {sel['selected_model']} ({sel['selected_model_name']})",
        f"Best params: {sel['selected_params']}",
        f"Selection criterion: {sel['criterion']} = {sel['cv_pr_auc_mean']:.4f} "
        f"(std {sel['cv_pr_auc_std']:.4f}, {sel['cv_folds']}-fold grouped stratified CV on training data)",
        f"Random seed: {metrics['random_seed']}",
        "",
        f"Data: {metrics['data']['n_raw_rows']} raw rows, "
        f"{dedup['n_removed_identical_records']} identical duplicate records removed, "
        f"{dedup['n_conflicting_label_groups']} conflicting-label groups "
        f"(policy: {dedup['conflict_policy']}), "
        f"{metrics['data']['n_rows_after_deduplication']} rows used.",
        "",
        f"Decision threshold: {thr['value']:.2f} ({thr['objective']}; "
        f"constraint satisfied: {thr['constraint_satisfied']})",
        f"Validation at threshold: precision={val['precision']:.3f} recall={val['recall']:.3f} "
        f"F{val['beta']:g}={val['fbeta']:.3f} PR-AUC={val['pr_auc']:.3f}",
        "",
        f"Test set ({test['n_samples']} rows, {test['n_positive']} churners), threshold {test['threshold']:.2f}:",
        f"ROC-AUC={test['roc_auc']:.3f} PR-AUC={test['pr_auc']:.3f}",
        "",
        sk_report,
    ]
    return "\n".join(lines)


def run_training(raw_df: pd.DataFrame, outdir: str, *, seed: int = 42, test_size: float = 0.2,
                 val_size: float = 0.2, n_iter: int = 20, cv_folds: int = 3, n_jobs: int = -1,
                 spaces=None, conflict_policy: str = "keep_grouped", beta: float = DEFAULT_BETA,
                 min_precision: float = DEFAULT_MIN_PRECISION, source=None) -> dict:
    """Deduplicate, split, tune all candidates, select, tune the threshold, evaluate, and save."""
    ensure_outdir(outdir)

    clean, dedup_report, removed = deduplicate(raw_df, conflict_policy=conflict_policy)
    df = clean.drop(columns=[ID_COL], errors="ignore")
    X_train, X_val, X_test, y_train, y_val, y_test = split_xy(df, test_size, val_size, seed)

    shared = {
        "train_validation": count_shared_feature_records(X_train, X_val),
        "train_test": count_shared_feature_records(X_train, X_test),
        "validation_test": count_shared_feature_records(X_val, X_test),
    }
    if any(shared.values()):
        raise RuntimeError(f"Identical feature records cross partitions: {shared}")

    groups_train = feature_row_hash(X_train)
    pre, num_cols, cat_cols = build_preprocessor(X_train)
    spaces = spaces if spaces is not None else model_spaces(seed)

    candidates, fitted = [], {}
    for name, (model, params) in spaces.items():
        search = randomized_cv(X_train, y_train, groups_train, pre, model, params,
                               seed=seed, n_iter=n_iter, cv_folds=cv_folds, n_jobs=n_jobs)
        best = search.best_estimator_
        proba_val = positive_class_proba(best, X_val)
        thr_info = pick_threshold(y_val.values, proba_val, beta=beta, min_precision=min_precision)
        val_metrics = classification_metrics(y_val, proba_val, thr_info["value"], beta=beta)
        candidates.append({
            "model": name,
            "model_name": MODEL_NAMES.get(name, name),
            "best_params": to_jsonable(search.best_params_),
            "n_iter": int(search.n_iter),
            "cv_pr_auc_mean": float(search.best_score_),
            "cv_pr_auc_std": float(search.cv_results_["std_test_score"][search.best_index_]),
            "validation_threshold": thr_info["value"],
            "validation_roc_auc": val_metrics["roc_auc"],
            "validation_pr_auc": val_metrics["pr_auc"],
            "validation_precision": val_metrics["precision"],
            "validation_recall": val_metrics["recall"],
            "validation_f1": val_metrics["f1"],
            "validation_fbeta": val_metrics["fbeta"],
        })
        fitted[name] = (best, thr_info, val_metrics)
        print(f"[{name}] CV PR-AUC={search.best_score_:.4f}  "
              f"validation PR-AUC={val_metrics['pr_auc']:.4f}  threshold={thr_info['value']:.2f}")

    best_candidate = max(candidates, key=lambda c: c["cv_pr_auc_mean"])
    best_name = best_candidate["model"]
    for c in candidates:
        c["selected"] = c["model"] == best_name
    best_pipe, thr_info, val_metrics = fitted[best_name]
    threshold = thr_info["value"]

    proba_test = positive_class_proba(best_pipe, X_test)
    test_metrics = classification_metrics(y_test, proba_test, threshold, beta=beta)
    reference_metrics = classification_metrics(y_test, proba_test, DEFAULT_THRESHOLD_REFERENCE, beta=beta)
    y_pred_test = apply_threshold(proba_test, threshold)
    sk_report = classification_report(y_test, y_pred_test, digits=3, zero_division=0)

    display_name = MODEL_NAMES.get(best_name, best_name)
    plot_roc(y_test, proba_test, display_name, os.path.join(outdir, "roc_curve.png"))
    plot_pr(y_test, proba_test, display_name, os.path.join(outdir, "pr_curve.png"), threshold)
    plot_confusion(y_test, y_pred_test, os.path.join(outdir, "confusion_matrix.png"), threshold)
    plot_feature_importance(best_pipe, num_cols, cat_cols, os.path.join(outdir, "feature_importance.png"))

    model_path = os.path.join(outdir, "best_model.joblib")
    _dump_atomic(best_pipe, model_path)

    comparison = pd.DataFrame(candidates)
    comparison["best_params"] = comparison["best_params"].astype(str)
    comparison.to_csv(os.path.join(outdir, "model_comparison.csv"), index=False)
    removed.to_csv(os.path.join(outdir, "removed_duplicates.csv"), index=False)
    write_json_atomic(dedup_report, os.path.join(outdir, "data_quality_report.json"))

    metrics = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_seed": seed,
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "data": {
            "source": source,
            "sheet": DATA_SHEET,
            "target": TARGET_COL,
            "excluded_columns": [ID_COL],
            "n_raw_rows": int(len(raw_df)),
            "n_rows_after_deduplication": int(len(clean)),
            "deduplication": {k: v for k, v in dedup_report.items() if k != "feature_identity_columns"},
            "split": {
                "method": "stratified split of identical-feature-record groups "
                          "(no identical feature record appears in two partitions)",
                "test_size": test_size,
                "validation_size_of_remaining": val_size,
                "partitions": {
                    "train": _partition_summary(y_train),
                    "validation": _partition_summary(y_val),
                    "test": _partition_summary(y_test),
                },
                "shared_feature_records_between_partitions": shared,
            },
        },
        "features": schema_from_model(best_pipe).to_dict(),
        "model_selection": {
            "criterion": SELECTION_CRITERION,
            "description": "Highest mean cross-validated average precision (PR-AUC) on the "
                           "training partition. Validation and test data are not used.",
            "cv": f"StratifiedGroupKFold(n_splits={cv_folds}, shuffle=True, random_state={seed})",
            "cv_folds": cv_folds,
            "n_iter": n_iter,
            "selected_model": best_name,
            "selected_model_name": display_name,
            "selected_params": best_candidate["best_params"],
            "cv_pr_auc_mean": best_candidate["cv_pr_auc_mean"],
            "cv_pr_auc_std": best_candidate["cv_pr_auc_std"],
            "final_model_fit_on": "training partition only (validation is reserved for the threshold)",
        },
        "threshold": {
            **thr_info,
            "objective": f"maximise F{beta:g} on the validation partition subject to "
                         f"precision >= {min_precision}",
            "tuned_on": "validation",
            "rule": "predict churn when probability >= threshold",
        },
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_metrics_reference_threshold_0_5": reference_metrics,
        "candidates": candidates,
        "artifacts": {
            "model_file": "best_model.joblib",
            "model_sha256": file_sha256(model_path),
            "model_comparison": "model_comparison.csv",
            "data_quality_report": "data_quality_report.json",
            "removed_duplicates": "removed_duplicates.csv",
            "plots": ["roc_curve.png", "pr_curve.png", "confusion_matrix.png", "feature_importance.png"],
        },
    }

    with open(os.path.join(outdir, "classification_report.txt"), "w") as f:
        f.write(_format_report(metrics, sk_report))
    write_json_atomic(metrics, os.path.join(outdir, "metrics.json"))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--cv-folds", type=int, default=3)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--models", default=",".join(MODEL_NAMES),
                    help="Comma-separated subset of: " + ", ".join(MODEL_NAMES))
    ap.add_argument("--conflict-policy", choices=CONFLICT_POLICIES, default="keep_grouped")
    ap.add_argument("--beta", type=float, default=DEFAULT_BETA)
    ap.add_argument("--min-precision", type=float, default=DEFAULT_MIN_PRECISION)
    args = ap.parse_args()

    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = sorted(set(requested) - set(MODEL_NAMES))
    if unknown:
        ap.error(f"Unknown model(s): {', '.join(unknown)}")
    spaces = {k: v for k, v in model_spaces(args.seed).items() if k in requested}

    raw = load_raw_dataset(args.input)
    metrics = run_training(
        raw, args.outdir, seed=args.seed, test_size=args.test_size, val_size=args.val_size,
        n_iter=args.n_iter, cv_folds=args.cv_folds, n_jobs=args.n_jobs, spaces=spaces,
        conflict_policy=args.conflict_policy, beta=args.beta, min_precision=args.min_precision,
        source=args.input,
    )

    sel, test = metrics["model_selection"], metrics["test_metrics"]
    dedup = metrics["data"]["deduplication"]
    print("[OK] Training complete.")
    print(f"Removed {dedup['n_removed_identical_records']} identical duplicate records; "
          f"{dedup['n_conflicting_label_groups']} conflicting-label groups.")
    print(f"Best model: {sel['selected_model']} (CV PR-AUC={sel['cv_pr_auc_mean']:.3f})")
    print(f"Threshold: {metrics['threshold']['value']:.2f}")
    print(f"Test ROC-AUC={test['roc_auc']:.3f}, PR-AUC(AP)={test['pr_auc']:.3f}, "
          f"precision={test['precision']:.3f}, recall={test['recall']:.3f}, F1={test['f1']:.3f}")
    print(f"Outputs saved to: {args.outdir}")

if __name__ == "__main__":
    main()
