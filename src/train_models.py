import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import RandomizedSearchCV
from utils import load_dataset, split_xy

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

def model_spaces():
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
            RandomForestClassifier(random_state=42, class_weight="balanced_subsample"),
            {
                "clf__n_estimators": [200, 300, 500],
                "clf__max_depth": [None, 8, 12, 16],
                "clf__min_samples_split": [2, 5, 10],
                "clf__min_samples_leaf": [1, 2, 4]
            }
        ),
        "gb": (
            GradientBoostingClassifier(random_state=42),
            {
                "clf__n_estimators": [150, 200, 300],
                "clf__learning_rate": [0.03, 0.05, 0.1],
                "clf__max_depth": [2, 3],
                "clf__subsample": [0.7, 0.85, 1.0]
            }
        ),
        "xgb": (
            XGBClassifier(
                random_state=42,
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

def randomized_cv(X, y, pre, model, params, seed=42):
    pipe = Pipeline([("pre", pre), ("clf", model)])
    search = RandomizedSearchCV(
        pipe, param_distributions=params, n_iter=20, cv=3, n_jobs=-1, random_state=seed,
        scoring="average_precision", refit=True, verbose=0
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, search.best_score_

def pick_threshold(y_true, proba, beta=2.0, min_precision=0.55):
    best = {"thr": 0.5, "fbeta": -1, "precision": None, "recall": None}
    for thr in np.linspace(0.05, 0.9, 86):
        pred = (proba >= thr).astype(int)
        tp = ((y_true == 1) & (pred == 1)).sum()
        fp = ((y_true == 0) & (pred == 1)).sum()
        fn = ((y_true == 1) & (pred == 0)).sum()
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        if precision < min_precision:
            continue
        b2 = beta * beta
        fbeta = (1 + b2) * (precision * recall) / (b2 * precision + recall + 1e-9)
        if fbeta > best["fbeta"]:
            best = {"thr": float(thr), "fbeta": float(fbeta), "precision": float(precision), "recall": float(recall)}
    if best["fbeta"] < 0:
        thr = 0.5
        pred = (proba >= thr).astype(int)
        tp = ((y_true == 1) & (pred == 1)).sum()
        fp = ((y_true == 0) & (pred == 1)).sum()
        fn = ((y_true == 1) & (pred == 0)).sum()
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        b2 = beta * beta
        fbeta = (1 + b2) * (precision * recall) / (b2 * precision + recall + 1e-9)
        best = {"thr": 0.5, "fbeta": float(fbeta), "precision": float(precision), "recall": float(recall)}
    return best

def plot_roc(y_true, proba, name, outpath):
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc = roc_auc_score(y_true, proba)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0,1], [0,1], "--")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout(); fig.savefig(outpath, dpi=160); plt.close(fig)

def plot_pr(y_true, proba, name, outpath):
    prec, rec, _ = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (Positive: churn)")
    ax.legend()
    fig.tight_layout(); fig.savefig(outpath, dpi=160); plt.close(fig)

def plot_confusion(y_true, y_pred, outpath):
    fig, ax = plt.subplots(figsize=(7,7))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred)).plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
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
        import numpy as np
        importances = np.abs(clf.coef_).ravel()
    else:
        import numpy as np
        importances = np.zeros(len(feature_names))

    import numpy as np
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ensure_outdir(args.outdir)

    df = load_dataset(args.input)
    X_train, X_val, X_test, y_train, y_val, y_test = split_xy(df, args.test_size, args.val_size, args.seed)

    pre, num_cols, cat_cols = build_preprocessor(X_train)

    best_pipe, best_name, best_params, best_val_ap = None, None, None, -1.0
    for name, (model, params) in model_spaces().items():
        est, params_, score_ = randomized_cv(X_train, y_train, pre, model, params, seed=args.seed)
        if score_ > best_val_ap:
            best_val_ap = score_
            best_pipe = est
            best_name = name
            best_params = params_

    # Tune threshold on validation set
    if hasattr(best_pipe.named_steps["clf"], "predict_proba"):
        proba_val = best_pipe.predict_proba(X_val)[:, 1]
    else:
        dec = best_pipe.decision_function(X_val)
        proba_val = (dec - dec.min()) / (dec.max() - dec.min() + 1e-9)

    thr_info = pick_threshold(y_val.values, proba_val, beta=2.0, min_precision=0.55)

    # Evaluate on test
    if hasattr(best_pipe.named_steps["clf"], "predict_proba"):
        proba_test = best_pipe.predict_proba(X_test)[:, 1]
    else:
        dec = best_pipe.decision_function(X_test)
        proba_test = (dec - dec.min()) / (dec.max() - dec.min() + 1e-9)

    y_pred_test = (proba_test >= thr_info["thr"]).astype(int)

    auc = roc_auc_score(y_test, proba_test)
    ap_score = average_precision_score(y_test, proba_test)
    report = classification_report(y_test, y_pred_test, digits=3, zero_division=0)

    with open(os.path.join(args.outdir, "classification_report.txt"), "w") as f:
        f.write(f"Best model: {best_name}\n")
        f.write(f"Best params: {best_params}\n")
        f.write(f"Tuned threshold: {thr_info}\n\n")
        f.write(report)

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump({
            "selection": {"best_model": best_name, "best_params": best_params, "val_pr_auc": best_val_ap, "threshold": thr_info},
            "test_metrics": {"roc_auc": float(auc), "average_precision": float(ap_score)}
        }, f, indent=2)

    plot_roc(y_test, proba_test, best_name, os.path.join(args.outdir, "roc_curve.png"))
    plot_pr(y_test, proba_test, best_name, os.path.join(args.outdir, "pr_curve.png"))
    plot_confusion(y_test, y_pred_test, os.path.join(args.outdir, "confusion_matrix.png"))
    plot_feature_importance(best_pipe, num_cols, cat_cols, os.path.join(args.outdir, "feature_importance.png"))

    dump(best_pipe, os.path.join(args.outdir, "best_model.joblib"))

    print("[OK] Training complete.")
    print(f"Best model: {best_name}")
    print(f"Threshold: {thr_info}")
    print(f"Test ROC-AUC={auc:.3f}, PR-AUC(AP)={ap_score:.3f}")
    print(f"Outputs saved to: {args.outdir}")

if __name__ == "__main__":
    main()
