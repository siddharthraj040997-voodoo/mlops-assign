"""Train, tune, track (MLflow) and package the heart-disease classifier.

Run:  python -m src.train
"""
import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,  # noqa: E402
                             accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split  # noqa: E402

from src.config import (CV_FOLDS, EXPERIMENT_NAME, FIG_DIR, MLFLOW_ARTIFACTS,  # noqa: E402
                        MLFLOW_DB, METRICS_PATH, MODEL_DIR, MODEL_PATH,
                        RANDOM_STATE, REGISTERED_MODEL_NAME, TEST_SIZE)
from src.data import load_clean, split_xy  # noqa: E402
from src.preprocessing import build_pipeline  # noqa: E402

SCORING = {"roc_auc": "roc_auc", "accuracy": "accuracy",
           "precision": "precision", "recall": "recall"}

MODEL_ZOO = {
    "logistic_regression": (
        LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
        {"clf__C": [0.01, 0.1, 1, 10], "clf__class_weight": [None, "balanced"]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        {"clf__n_estimators": [200, 400], "clf__max_depth": [3, 5, None],
         "clf__min_samples_leaf": [1, 3]},
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1],
         "clf__max_depth": [2, 3]},
    ),
}


def setup_mlflow():
    uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB}")
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME,
                                 artifact_location=MLFLOW_ARTIFACTS.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)
    return uri


def evaluate(model, X, y):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }, proba, pred


def plot_roc_cm(model, X, y, title):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    RocCurveDisplay.from_estimator(model, X, y, ax=ax, color="#e76f51")
    ax.plot([0, 1], [0, 1], "k--", lw=.8)
    ax.set_title(f"ROC - {title}")
    fig_cm, ax_cm = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X, y, ax=ax_cm, cmap="Blues",
        display_labels=["No disease", "Disease"])
    ax_cm.set_title(f"Confusion matrix - {title}")
    return fig, fig_cm


def plot_importance(model, title):
    names = model.named_steps["prep"].get_feature_names_out()
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        vals = clf.feature_importances_
    else:
        vals = clf.coef_.ravel()
    order = np.argsort(np.abs(vals))[-12:]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(np.array(names)[order], vals[order], color="#264653")
    ax.set_title(f"Top features - {title}")
    return fig


def main():
    t0 = time.time()
    uri = setup_mlflow()
    print(f"MLflow tracking URI: {uri}")
    df = load_clean()
    X, y = split_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    summary = {}
    for name, (estimator, grid) in MODEL_ZOO.items():
        with mlflow.start_run(run_name=name) as parent:
            mlflow.set_tags({"stage": "model-selection", "model_family": name})
            mlflow.log_params({"cv_folds": CV_FOLDS, "test_size": TEST_SIZE,
                               "n_train": len(X_tr), "n_test": len(X_te),
                               "sklearn_version": sklearn.__version__})
            gs = GridSearchCV(build_pipeline(estimator), grid, scoring=SCORING,
                              refit="roc_auc", cv=cv, n_jobs=-1,
                              return_train_score=False)
            gs.fit(X_tr, y_tr)
            res = gs.cv_results_

            # one nested run per hyper-parameter candidate
            for i, params in enumerate(res["params"]):
                with mlflow.start_run(run_name=f"{name}-cand{i:02d}", nested=True):
                    mlflow.set_tag("stage", "tuning")
                    mlflow.log_params({k.replace("clf__", ""): str(v)
                                       for k, v in params.items()})
                    for m in SCORING:
                        mlflow.log_metric(f"cv_{m}_mean", res[f"mean_test_{m}"][i])
                        mlflow.log_metric(f"cv_{m}_std", res[f"std_test_{m}"][i])

            best = gs.best_estimator_
            bi = gs.best_index_
            cv_metrics = {f"cv_{m}": float(res[f"mean_test_{m}"][bi]) for m in SCORING}
            cv_metrics.update({f"cv_{m}_std": float(res[f"std_test_{m}"][bi])
                               for m in SCORING})
            test_metrics, _, _ = evaluate(best, X_te, y_te)
            test_metrics = {f"test_{k}": float(v) for k, v in test_metrics.items()}

            mlflow.log_params({k.replace("clf__", "best_"): str(v)
                               for k, v in gs.best_params_.items()})
            mlflow.log_metrics({**cv_metrics, **test_metrics})
            fig_roc, fig_cm = plot_roc_cm(best, X_te, y_te, name)
            mlflow.log_figure(fig_roc, "plots/roc_curve.png")
            mlflow.log_figure(fig_cm, "plots/confusion_matrix.png")
            fig_roc.savefig(FIG_DIR / f"roc_{name}.png", dpi=130, bbox_inches="tight")
            fig_cm.savefig(FIG_DIR / f"cm_{name}.png", dpi=130, bbox_inches="tight")
            fig_imp = plot_importance(best, name)
            mlflow.log_figure(fig_imp, "plots/feature_importance.png")
            fig_imp.savefig(FIG_DIR / f"importance_{name}.png", dpi=130,
                            bbox_inches="tight")
            plt.close("all")
            sig = infer_signature(X_te, best.predict_proba(X_te)[:, 1])
            mlflow.sklearn.log_model(best, name="model", signature=sig,
                                     serialization_format="cloudpickle",
                                     input_example=X_te.head(3))
            summary[name] = {"best_params": {k.replace("clf__", ""): v
                                             for k, v in gs.best_params_.items()},
                             **cv_metrics, **test_metrics,
                             "run_id": parent.info.run_id}
            print(f"{name:22s} cv_auc={cv_metrics['cv_roc_auc']:.3f} "
                  f"test_auc={test_metrics['test_roc_auc']:.3f} "
                  f"test_acc={test_metrics['test_accuracy']:.3f}")

    # model selection on CV ROC-AUC only (the test set is never used to choose)
    winner = max(summary, key=lambda k: summary[k]["cv_roc_auc"])
    print(f"Selected model: {winner}")
    estimator, _ = MODEL_ZOO[winner]
    final = build_pipeline(estimator.set_params(
        **{k: v for k, v in summary[winner]["best_params"].items()}))
    final.fit(X, y)  # refit on all labelled data for deployment

    with mlflow.start_run(run_name=f"final-{winner}"):
        mlflow.set_tags({"stage": "final", "selected_model": winner})
        mlflow.log_params(summary[winner]["best_params"])
        mlflow.log_metrics({k: v for k, v in summary[winner].items()
                            if k.startswith(("cv_", "test_"))})
        sig = infer_signature(X, final.predict_proba(X)[:, 1])
        mlflow.sklearn.log_model(final, name="model", signature=sig,
                                 serialization_format="cloudpickle",
                                 input_example=X.head(3),
                                 registered_model_name=REGISTERED_MODEL_NAME)
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(final, MODEL_PATH)
        meta = {"selected_model": winner, "sklearn_version": sklearn.__version__,
                "n_samples": int(len(X)), "models": summary}
        METRICS_PATH.write_text(json.dumps(meta, indent=2))
        mlflow.log_artifact(str(METRICS_PATH))
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="joblib")
    print(f"Saved {MODEL_PATH}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
