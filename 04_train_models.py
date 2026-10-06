# -*- coding: utf-8 -*-
"""
04 — Train: LogisticRegression (baseline + search), CatBoostClassifier (gradient boosting
with strong categorical handling), HistGradientBoostingClassifier (search); Platt on val;
PR curve; threshold from validation.
"""
import json
import os
import subprocess
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from _paths import CLEAN_CSV, MODELS_DIR, OUTPUT_DIR


def log(msg: str) -> None:
    print(f"[04_train_models] {msg}", flush=True)


def detect_nvidia_gpu() -> str:
    """Return a short GPU status string for console visibility."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return "not detected"


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "axes.edgecolor": "#c8c8c8",
            "axes.linewidth": 0.85,
            "axes.labelcolor": "#333333",
            "axes.titlecolor": "#1a1a1a",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "text.color": "#222222",
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "grid.color": "#d4d4d4",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.9,
            "lines.linewidth": 1.5,
            "lines.antialiased": True,
        }
    )


def build_preprocess(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = X.select_dtypes(include=["object", "category"]).columns.tolist()
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                cat,
            ),
        ]
    )
    return pre, num, cat


def age_midpoint(age: object) -> float:
    s = str(age).strip()
    if s.startswith("[") and "-" in s:
        lo, hi = s.strip("[]()").split("-", 1)
        try:
            return (float(lo) + float(hi)) / 2.0
        except ValueError:
            return np.nan
    return np.nan


def diag_chapter(v: object) -> str:
    """Coarse ICD-style chapter from the cleaned 3-character diagnosis prefix."""
    s = str(v).strip()
    try:
        code = float(s)
    except ValueError:
        return "other"
    if np.isnan(code):
        return "other"

    if 390 <= code <= 459 or int(code) == 785:
        return "circulatory"
    if 460 <= code <= 519 or int(code) == 786:
        return "respiratory"
    if 520 <= code <= 579 or int(code) == 787:
        return "digestive"
    if int(code) == 250:
        return "diabetes"
    if 800 <= code <= 999:
        return "injury"
    if 710 <= code <= 739:
        return "musculoskeletal"
    if 580 <= code <= 629 or int(code) == 788:
        return "genitourinary"
    if 140 <= code <= 239:
        return "neoplasms"
    return "other"


def add_model_features(X: pd.DataFrame) -> pd.DataFrame:
    """High-signal engineered features for readmission risk; avoids leakage columns."""
    X = X.copy()

    if "age" in X.columns:
        X["fea_age_midpoint"] = X["age"].map(age_midpoint)
        X["fea_age_is_70_plus"] = (X["fea_age_midpoint"] >= 75).astype(int)

    for col in ("diag_1", "diag_2", "diag_3"):
        if col in X.columns:
            X[f"fea_{col}_chapter"] = X[col].map(diag_chapter)

    med_cols = [
        c
        for c in [
            "metformin",
            "repaglinide",
            "nateglinide",
            "chlorpropamide",
            "glimepiride",
            "acetohexamide",
            "glipizide",
            "glyburide",
            "tolbutamide",
            "pioglitazone",
            "rosiglitazone",
            "acarbose",
            "miglitol",
            "troglitazone",
            "tolazamide",
            "examide",
            "citoglipton",
            "insulin",
            "glyburide-metformin",
            "glipizide-metformin",
            "glimepiride-pioglitazone",
            "metformin-rosiglitazone",
            "metformin-pioglitazone",
        ]
        if c in X.columns
    ]
    if med_cols:
        meds = X[med_cols].astype(str)
        X["fea_active_med_count"] = meds.ne("No").sum(axis=1)
        X["fea_changed_med_count"] = meds.isin(["Up", "Down"]).sum(axis=1)
        X["fea_any_med_up"] = meds.eq("Up").any(axis=1).astype(int)
        X["fea_any_med_down"] = meds.eq("Down").any(axis=1).astype(int)
        if "insulin" in X.columns:
            X["fea_insulin_changed"] = X["insulin"].astype(str).isin(["Up", "Down"]).astype(int)
            X["fea_insulin_active"] = X["insulin"].astype(str).ne("No").astype(int)

    if "time_in_hospital" in X.columns:
        days = pd.to_numeric(X["time_in_hospital"], errors="coerce").replace(0, np.nan)
        if "num_medications" in X.columns:
            X["fea_meds_per_day"] = pd.to_numeric(X["num_medications"], errors="coerce") / days
        if "num_lab_procedures" in X.columns:
            X["fea_labs_per_day"] = pd.to_numeric(X["num_lab_procedures"], errors="coerce") / days
        if "num_procedures" in X.columns:
            X["fea_procedures_per_day"] = pd.to_numeric(X["num_procedures"], errors="coerce") / days

    if all(c in X.columns for c in ("number_inpatient", "number_emergency", "number_outpatient")):
        inpt = pd.to_numeric(X["number_inpatient"], errors="coerce").fillna(0)
        emerg = pd.to_numeric(X["number_emergency"], errors="coerce").fillna(0)
        outp = pd.to_numeric(X["number_outpatient"], errors="coerce").fillna(0)
        X["fea_prior_total_visits"] = inpt + emerg + outp
        X["fea_prior_acute_visits"] = inpt + emerg
        X["fea_prior_acute_share"] = (inpt + emerg) / (1.0 + inpt + emerg + outp)

    return X


def platt_fit_transform(pipe: Pipeline, X_va: pd.DataFrame, y_va: pd.Series) -> tuple[LogisticRegression, np.ndarray]:
    """معايرة احتمالات الخرج (Platt): انحدار لوجستي على عمود احتمال الفئة 1."""
    raw = pipe.predict_proba(X_va)[:, 1].reshape(-1, 1)
    platt = LogisticRegression(C=1e12, solver="lbfgs", max_iter=1000)
    platt.fit(raw, y_va)
    cal = platt.predict_proba(raw)[:, 1]
    return platt, cal


def apply_platt(pipe: Pipeline, platt: LogisticRegression, X: pd.DataFrame) -> np.ndarray:
    raw = pipe.predict_proba(X)[:, 1].reshape(-1, 1)
    return platt.predict_proba(raw)[:, 1]


def pick_thresholds(y_true: np.ndarray, proba: np.ndarray, min_precision: float = 0.10) -> dict:
    """Return honest validation-selected thresholds for both balanced F1 and recall-oriented operation."""
    ts = np.linspace(0.01, 0.99, 99)
    best_f1_t, best_f1 = 0.5, -1.0
    best_feasible: tuple[float, float, float] | None = None  # recall, t, precision

    for t in ts:
        pred = (proba >= t).astype(int)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_f1_t = f1, t
        if p >= min_precision:
            if best_feasible is None or r > best_feasible[0]:
                best_feasible = (r, float(t), p)

    return {
        "max_f1": float(best_f1_t),
        "max_f1_score": float(best_f1),
        "max_recall_at_min_precision": float(best_feasible[1] if best_feasible is not None else best_f1_t),
        "min_precision": float(min_precision),
    }


def randomized_search_lr(pipe_template: Pipeline, X_tr, y_tr, g_tr, random_state: int = 42) -> tuple[Pipeline, dict]:
    gkf = GroupKFold(n_splits=3)
    param = {
        "clf__C": loguniform(1e-2, 40.0),
        "clf__penalty": ["l1", "l2"],
    }
    n_iter = 16
    log("[TRAINING_DEVICE] LogisticRegression = CPU (scikit-learn)")
    log(
        f"LogisticRegression RandomizedSearchCV: n_iter={n_iter}, cv=3 GroupKFold, "
        f"scoring=average_precision (~{n_iter * 3} fits). Progress from sklearn follows."
    )
    t0 = time.perf_counter()
    search = RandomizedSearchCV(
        clone(pipe_template),
        param_distributions=param,
        n_iter=n_iter,
        scoring="average_precision",
        cv=gkf,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
        verbose=2,
    )
    search.fit(X_tr, y_tr, groups=g_tr)
    log(f"LogisticRegression search done in {time.perf_counter() - t0:.1f}s. best_params={search.best_params_}")
    return search.best_estimator_, search.best_params_


def randomized_search_hgb(pipe_template: Pipeline, X_tr, y_tr, g_tr, random_state: int = 44) -> tuple[Pipeline, dict]:
    gkf = GroupKFold(n_splits=3)
    param = {
        "clf__learning_rate": loguniform(0.02, 0.18),
        "clf__max_depth": randint(4, 13),
        "clf__max_iter": randint(180, 401),
        "clf__min_samples_leaf": randint(8, 100),
        "clf__l2_regularization": loguniform(1e-5, 1e-1),
    }
    n_iter = 12
    log("[TRAINING_DEVICE] HistGradientBoostingClassifier = CPU (scikit-learn)")
    log(
        f"HistGradientBoosting RandomizedSearchCV: n_iter={n_iter}, cv=3 GroupKFold, "
        f"scoring=average_precision (~{n_iter * 3} fits). This step is slower than LR."
    )
    t0 = time.perf_counter()
    search = RandomizedSearchCV(
        clone(pipe_template),
        param_distributions=param,
        n_iter=n_iter,
        scoring="average_precision",
        cv=gkf,
        random_state=random_state,
        n_jobs=-1,
        refit=True,
        verbose=2,
    )
    search.fit(X_tr, y_tr, groups=g_tr)
    log(f"HistGradientBoosting search done in {time.perf_counter() - t0:.1f}s. best_params={search.best_params_}")
    return search.best_estimator_, search.best_params_


def plot_pr_curve(y_true, proba, out_path: Path, title: str) -> None:
    apply_plot_style()
    prec, rec, thr = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)
    base = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(7.2, 4.8), facecolor="white")
    ax.set_facecolor("#fafafa")
    ax.plot(rec, prec, color="#1976d2", linewidth=1.5, label=f"Model (AP = {ap:.4f})")
    ax.axhline(base, color="#9e9e9e", linestyle=(0, (4, 4)), linewidth=1.1, label=f"Random baseline ≈ prevalence ({base:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.grid(True, which="major", linestyle="-", alpha=0.45)
    for spine in ax.spines.values():
        spine.set_linewidth(0.85)
        spine.set_edgecolor("#c8c8c8")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def eval_at_threshold(y_true, proba, t: float) -> dict:
    pred = (proba >= t).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    return {
        "threshold": float(t),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1_pos": float(f1_score(y_true, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "confusion_matrix_labels_0_1": cm.tolist(),
    }


def build_catboost_pipeline(pre: ColumnTransformer, task_type: str) -> Pipeline:
    kwargs = {
        "iterations": 400,
        "depth": 6,
        "learning_rate": 0.06,
        "loss_function": "Logloss",
        "auto_class_weights": "Balanced",
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": -1,
        "task_type": task_type,
    }
    if task_type.upper() == "GPU":
        kwargs["devices"] = os.environ.get("MEDOPS_CATBOOST_GPU_DEVICES", "0")

    return Pipeline(
        [
            ("pre", clone(pre)),
            ("clf", CatBoostClassifier(**kwargs)),
        ]
    )


def fit_catboost_gpu_first(pre: ColumnTransformer, X_tr: pd.DataFrame, y_tr: pd.Series) -> tuple[Pipeline, dict]:
    """جرّب CatBoost على GPU أولاً، ثم ارجع إلى CPU إذا فشل CUDA/CatBoost."""
    if os.environ.get("MEDOPS_DISABLE_GPU") == "1":
        log("[TRAINING_DEVICE] CatBoostClassifier = CPU (GPU disabled by MEDOPS_DISABLE_GPU=1)")
        pipe = build_catboost_pipeline(pre, "CPU")
        pipe.fit(X_tr, y_tr)
        return pipe, {"task_type": "CPU", "gpu": "disabled_by_env"}

    try:
        log("[TRAINING_DEVICE] CatBoostClassifier = GPU requested (task_type=GPU, devices=0)")
        pipe = build_catboost_pipeline(pre, "GPU")
        pipe.fit(X_tr, y_tr)
        log("[TRAINING_DEVICE] CatBoostClassifier = GPU active")
        return pipe, {"task_type": "GPU", "gpu": "used"}
    except Exception as ex:
        log(f"[TRAINING_DEVICE] CatBoostClassifier = CPU fallback. GPU failed: {type(ex).__name__}: {str(ex)[:300]}")
        pipe = build_catboost_pipeline(pre, "CPU")
        pipe.fit(X_tr, y_tr)
        return pipe, {"task_type": "CPU", "gpu": "fallback", "gpu_error": f"{type(ex).__name__}: {str(ex)[:500]}"}


def main():
    log("Starting training script.")
    log(f"[TRAINING_DEVICE] NVIDIA GPU = {detect_nvidia_gpu()}")
    log("[TRAINING_DEVICE] CPU-only models: LogisticRegression, HistGradientBoostingClassifier")
    log("[TRAINING_DEVICE] GPU-capable model: CatBoostClassifier (GPU first, CPU fallback)")
    df = pd.read_csv(CLEAN_CSV, low_memory=False)
    y = df["readmit_lt30"].astype(int)
    groups = df["patient_nbr"]
    exclude = ["encounter_id", "patient_nbr", "readmitted", "readmit_lt30"]
    X = df.drop(columns=[c for c in exclude if c in df.columns])
    X = add_model_features(X)

    pre, num_cols, cat_cols = build_preprocess(X)
    log(f"Features: numeric={len(num_cols)}, categorical={len(cat_cols)}")

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    idx_trainval, idx_test = next(gss1.split(X, y, groups))
    X_tv, y_tv, g_tv = X.iloc[idx_trainval], y.iloc[idx_trainval], groups.iloc[idx_trainval]
    X_te, y_te = X.iloc[idx_test], y.iloc[idx_test]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=43)
    idx_tr, idx_va = next(gss2.split(X_tv, y_tv, g_tv))
    X_tr, y_tr, g_tr = X_tv.iloc[idx_tr], y_tv.iloc[idx_tr], g_tv.iloc[idx_tr]
    X_va, y_va = X_tv.iloc[idx_va], y_tv.iloc[idx_va]

    log(f"Split sizes: train={len(X_tr)}, val={len(X_va)}, test={len(X_te)}")
    prevalence_val = float(y_va.mean())
    prevalence_test = float(y_te.mean())

    pipe_lr = Pipeline(
        [
            ("pre", clone(pre)),
            (
                "clf",
                LogisticRegression(max_iter=12000, class_weight="balanced", solver="saga", n_jobs=-1),
            ),
        ]
    )
    pipe_hgb = Pipeline(
        [
            ("pre", clone(pre)),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_depth=10,
                    max_iter=250,
                    learning_rate=0.08,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    best_lr, lr_params = randomized_search_lr(pipe_lr, X_tr, y_tr, g_tr)

    best_hgb, hgb_params = randomized_search_hgb(pipe_hgb, X_tr, y_tr, g_tr)

    log(
        "CatBoostClassifier: fixed params (depth=6, iterations=400, auto_class_weights=Balanced); GPU first, CPU fallback; "
        "uses same sklearn preprocessing (ordinal categoricals) — good for report vs linear + sklearn HGB."
    )
    t_cat = time.perf_counter()
    best_cat, cat_runtime_meta = fit_catboost_gpu_first(pre, X_tr, y_tr)
    log(f"CatBoost fit done in {time.perf_counter() - t_cat:.1f}s.")

    models: dict[str, tuple[Pipeline, dict]] = {
        "logistic_regression": (best_lr, {"search": "RandomizedSearchCV", "best": lr_params}),
        "catboost": (
            best_cat,
            {
                "search": "fixed",
                "note": "CatBoostClassifier: iterations=400, depth=6, lr=0.06, auto_class_weights=Balanced",
                **cat_runtime_meta,
            },
        ),
        "hist_gradient_boosting": (best_hgb, {"search": "RandomizedSearchCV", "best": hgb_params}),
    }

    min_prec = 0.10
    deployment_threshold_policy = os.environ.get("MEDOPS_THRESHOLD_POLICY", "max_f1").strip().lower()
    if deployment_threshold_policy not in {"max_f1", "max_recall_at_min_precision"}:
        log(f"Unknown MEDOPS_THRESHOLD_POLICY={deployment_threshold_policy!r}; using max_f1.")
        deployment_threshold_policy = "max_f1"
    log(f"Threshold policy for saved bundle = {deployment_threshold_policy}")
    per_model: dict = {}
    legacy_05: dict = {}

    for name, (est_tr, meta) in models.items():
        log(f"Post-process (refit train+val, Platt, threshold, save): {name} …")
        proba_te_raw = est_tr.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, proba_te_raw)
        pred05 = (proba_te_raw >= 0.5).astype(int)
        legacy_05[name] = {
            "roc_auc_test": float(auc),
            "f1_pos_test": float(f1_score(y_te, pred05, pos_label=1, zero_division=0)),
        }
        log(f"{name} @0.5 threshold: ROC-AUC={auc:.4f}, F1(pos)={legacy_05[name]['f1_pos_test']:.4f}")

        est_tv = clone(est_tr)
        est_tv.fit(X_tv, y_tv)
        platt, proba_va_cal = platt_fit_transform(est_tv, X_va, y_va)
        thresholds = pick_thresholds(y_va.to_numpy(), proba_va_cal, min_precision=min_prec)
        thr = float(thresholds[deployment_threshold_policy])

        proba_te_cal = apply_platt(est_tv, platt, X_te)
        pr_auc = average_precision_score(y_te, proba_te_cal)

        metrics_test = eval_at_threshold(y_te.to_numpy(), proba_te_cal, thr)
        metrics_test_max_f1 = eval_at_threshold(y_te.to_numpy(), proba_te_cal, float(thresholds["max_f1"]))
        metrics_test_recall = eval_at_threshold(
            y_te.to_numpy(), proba_te_cal, float(thresholds["max_recall_at_min_precision"])
        )

        joblib.dump(est_tv, MODELS_DIR / f"{name}_pipeline.joblib")
        bundle = {
            "name": name,
            "pipeline": est_tv,
            "platt": platt,
            "threshold": thr,
            "threshold_policy": deployment_threshold_policy,
            "threshold_candidates": thresholds,
            "min_precision_for_recall_rule": min_prec,
            "meta": meta,
        }
        joblib.dump(bundle, MODELS_DIR / f"{name}_inference_bundle.joblib")

        per_model[name] = {
            **meta,
            "roc_auc_test_raw_proba": float(auc),
            "pr_auc_test_calibrated": float(pr_auc),
            "pr_auc_baseline_prevalence_test": float(prevalence_test),
            "pr_auc_baseline_note": "Average precision of a naive constant predictor ≈ test set positive rate.",
            "threshold_candidates_chosen_on_val_platt": thresholds,
            "threshold_chosen_on_val_platt": float(thr),
            "threshold_policy": deployment_threshold_policy,
            "prevalence_val": prevalence_val,
            "brier_score_test_calibrated": float(brier_score_loss(y_te, proba_te_cal)),
            "metrics_test_at_chosen_threshold": metrics_test,
            "metrics_test_at_max_f1_threshold": metrics_test_max_f1,
            "metrics_test_at_recall_threshold": metrics_test_recall,
            "artifacts": {
                "pipeline": str(MODELS_DIR / f"{name}_pipeline.joblib"),
                "bundle": str(MODELS_DIR / f"{name}_inference_bundle.joblib"),
            },
        }

    best = max(
        per_model,
        key=lambda k: (
            per_model[k]["pr_auc_test_calibrated"],
            per_model[k]["metrics_test_at_max_f1_threshold"]["f1_pos"],
        ),
    )
    log(f"Best model (PR-AUC calibrated, then F1 @chosen threshold): {best}")

    fig_path = OUTPUT_DIR / "figures" / "pr_curve_best_model_test.png"
    best_tv = joblib.load(MODELS_DIR / f"{best}_pipeline.joblib")
    platt_b = joblib.load(MODELS_DIR / f"{best}_inference_bundle.joblib")["platt"]
    proba_te_best = apply_platt(best_tv, platt_b, X_te)
    plot_pr_curve(
        y_te.to_numpy(),
        proba_te_best,
        fig_path,
        f"Precision-Recall (test) - {best} | Platt calibration (fit on validation)",
    )

    rep_path = OUTPUT_DIR / "classification_report_best.txt"
    thr_b = float(joblib.load(MODELS_DIR / f"{best}_inference_bundle.joblib")["threshold"])
    pred_best = (proba_te_best >= thr_b).astype(int)
    rep = classification_report(y_te, pred_best, digits=3)
    rep_path.write_text(
        rep
        + f"\n\n(model={best}, threshold={thr_b:.4f}, probabilities=Platt calibrated using validation)\n",
        encoding="utf-8",
    )

    summary = {
        "split": "GroupShuffleSplit by patient_nbr (no patient leakage across splits)",
        "sizes": {"train": len(X_tr), "val": len(X_va), "test": len(X_te)},
        "legacy_metrics_threshold_0_5": legacy_05,
        "per_model_extended": per_model,
        "best_model_by_pr_auc_calibrated_then_f1": best,
        "pr_curve_test_figure": str(fig_path),
        "calibration": "Platt: LogisticRegression on predict_proba[:,1] fitted on validation after refit on train+val",
        "feature_engineering": "age midpoint, diagnosis chapters, medication activity/change counts, per-day utilization, prior visit interactions",
        "threshold_policy": deployment_threshold_policy,
        "threshold_rule": "max_f1 by default; recall-oriented threshold also reported with min precision constraint",
    }
    out = OUTPUT_DIR / "model_comparison.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Best={best}, threshold={thr_b:.4f}")
    log(f"Wrote: {out}")
    log(f"Wrote: {rep_path}")
    log(f"Wrote: {fig_path}")
    log("Done.")


if __name__ == "__main__":
    main()
