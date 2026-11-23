"""Train and evaluate EnergyGuard models without test-set leakage."""
from __future__ import annotations

import argparse
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, precision_score,
    recall_score, f1_score, average_precision_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay, precision_recall_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from .config import PROCESSED_DIR, MODEL_DIR, REPORT_DIR, UCI_FILE, SGCC_FILE
from .data_pipeline import prepare_uci, prepare_sgcc

FORECAST_FEATURES = [
    "energy_kwh", "hour", "weekday", "lag_1", "lag_2", "lag_24",
    "rolling_mean_3", "rolling_mean_24",
]


def _save_json(data, name):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / name).write_text(json.dumps(data, indent=2, default=float))


def train_forecasting():
    if not UCI_FILE.exists():
        raise FileNotFoundError("Add UCI data to data/raw/household_power_consumption.txt first.")
    df = prepare_uci()
    df.to_csv(PROCESSED_DIR / "uci_hourly_features.csv", index=False)
    # Chronological 70/15/15 split; no random shuffling.
    n = len(df)
    train_end, val_end = int(n * .70), int(n * .85)
    train, validation, test = df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]
    X_train, y_train = train[FORECAST_FEATURES], train["target_next_hour_kwh"]
    X_val, y_val = validation[FORECAST_FEATURES], validation["target_next_hour_kwh"]
    X_test, y_test = test[FORECAST_FEATURES], test["target_next_hour_kwh"]
    models = {
        "linear_regression": Pipeline([("imputer", SimpleImputer(strategy="median")),
                                       ("scaler", StandardScaler()),
                                       ("model", LinearRegression())]),
        "random_forest_regressor": Pipeline([("imputer", SimpleImputer(strategy="median")),
                                              ("model", RandomForestRegressor(
                                                  n_estimators=250, min_samples_leaf=2,
                                                  random_state=42, n_jobs=-1))]),
    }
    # Fit each candidate on training only; validation is used for comparison.
    val_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        val_scores[name] = {"MAE": mean_absolute_error(y_val, pred),
                            "RMSE": float(np.sqrt(mean_squared_error(y_val, pred))),
                            "R2": r2_score(y_val, pred)}
    best_name = min(val_scores, key=lambda k: val_scores[k]["MAE"])
    best = models[best_name]
    # Test is evaluated once after model selection; no refit on validation.
    prediction = best.predict(X_test)
    metrics = {
        "selected_model": best_name,
        "validation": val_scores,
        "test": {"MAE": mean_absolute_error(y_test, prediction),
                 "RMSE": float(np.sqrt(mean_squared_error(y_test, prediction))),
                 "R2": r2_score(y_test, prediction)},
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "test_start": str(test["timestamp"].min()),
        "test_end": str(test["timestamp"].max()),
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODEL_DIR / "forecast_model.joblib")
    joblib.dump(models, MODEL_DIR / "forecast_candidates.joblib")
    test_output = test[["timestamp", "target_next_hour_kwh"]].copy()
    test_output["prediction_kwh"] = prediction
    test_output.to_csv(REPORT_DIR / "forecast_test_predictions.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 4))
    view = test_output.head(500)
    ax.plot(pd.to_datetime(view["timestamp"]), view["target_next_hour_kwh"], label="Actual")
    ax.plot(pd.to_datetime(view["timestamp"]), view["prediction_kwh"], label="Predicted")
    ax.set(title="Next-hour household energy: actual vs predicted", ylabel="kWh")
    ax.legend(); fig.tight_layout(); fig.savefig(REPORT_DIR / "forecast_test.png", dpi=150); plt.close(fig)
    _save_json(metrics, "forecast_metrics.json")
    return metrics


def train_anomaly():
    if not UCI_FILE.exists():
        raise FileNotFoundError("Add UCI data to data/raw/household_power_consumption.txt first.")
    df = prepare_uci()
    features = ["energy_kwh", "hour", "weekday", "lag_1", "lag_24", "rolling_mean_24"]
    split = int(len(df) * .85)
    train, review = df.iloc[:split], df.iloc[split:]
    model = Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler()),
                      ("model", IsolationForest(
                          n_estimators=250, contamination="auto",
                          random_state=42, n_jobs=-1))])
    model.fit(train[features])
    # IsolationForest: -1 unusual, +1 inlier. Score is oriented so higher = more unusual.
    labels = model.predict(review[features])
    scores = -model.decision_function(review[features])
    output = review[["timestamp", "energy_kwh"]].copy()
    output["is_anomaly"] = labels == -1
    output["anomaly_score"] = scores
    output["recent_context_lag_1_kwh"] = review["lag_1"].values
    output.to_csv(REPORT_DIR / "anomaly_review.csv", index=False)
    metrics = {"training_rows": len(train), "review_rows": len(review),
               "flagged_count": int((labels == -1).sum()),
               "flagged_percentage": float((labels == -1).mean() * 100),
               "note": "Unusual observations only; UCI has no ground-truth anomaly labels."}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "anomaly_model.joblib")
    _save_json(metrics, "anomaly_metrics.json")
    return metrics


def train_theft():
    if not SGCC_FILE.exists():
        raise FileNotFoundError("Add extracted SGCC CSV to data/raw/sgcc.csv first.")
    df, features = prepare_sgcc()
    # Split consumers, not daily readings. Identifiers are retained only for traceability.
    train_val_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=.15, random_state=42, stratify=df["FLAG"])
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=(.15/.85), random_state=42,
        stratify=df.iloc[train_val_idx]["FLAG"])
    train, validation, test = df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]
    X_train, y_train = train[features], train["FLAG"]
    X_val, y_val = validation[features], validation["FLAG"]
    X_test, y_test = test[features], test["FLAG"]
    model = Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("classifier", RandomForestClassifier(
                          n_estimators=350, min_samples_leaf=2,
                          class_weight="balanced_subsample",
                          random_state=42, n_jobs=-1))])
    model.fit(X_train, y_train)
    val_probability = model.predict_proba(X_val)[:, 1]
    # Select threshold on validation only, maximizing F1; default to 0.5 if no positive F1.
    from sklearn.metrics import f1_score as _f1
    thresholds = np.linspace(.10, .90, 81)
    threshold = max(thresholds, key=lambda t: _f1(y_val, val_probability >= t, zero_division=0))
    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability >= threshold).astype(int)
    metrics = {
        "threshold_selected_on_validation": float(threshold),
        "test": {
            "precision": precision_score(y_test, prediction, zero_division=0),
            "recall": recall_score(y_test, prediction, zero_division=0),
            "f1": f1_score(y_test, prediction, zero_division=0),
            "pr_auc": average_precision_score(y_test, probability),
            "confusion_matrix": confusion_matrix(y_test, prediction).tolist(),
            "classification_report": classification_report(y_test, prediction, output_dict=True, zero_division=0),
        },
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "class_counts": df["FLAG"].value_counts().sort_index().to_dict(),
        "feature_names": features,
        "note": "Risk screening only; a positive prediction is not proof of electricity theft.",
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "features": features, "threshold": float(threshold)},
                MODEL_DIR / "theft_model.joblib")
    out = test[["CONS_NO", "FLAG"]].copy()
    out["risk_probability"] = probability
    out["predicted_flag"] = prediction
    out.to_csv(REPORT_DIR / "theft_test_predictions.csv", index=False)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, prediction))
    disp.plot(); plt.title("SGCC theft-risk screening — held-out consumers")
    plt.tight_layout(); plt.savefig(REPORT_DIR / "theft_confusion_matrix.png", dpi=150); plt.close()
    _save_json(metrics, "theft_metrics.json")
    return {k: v for k, v in metrics.items() if k != "test"} | {"test": {
        k: v for k, v in metrics["test"].items() if k != "classification_report"
    }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Train all available tasks")
    parser.add_argument("--forecast", action="store_true")
    parser.add_argument("--anomaly", action="store_true")
    parser.add_argument("--theft", action="store_true")
    args = parser.parse_args()
    run_all = args.all or not any((args.forecast, args.anomaly, args.theft))
    results = {}
    if run_all or args.forecast: results["forecast"] = train_forecasting()
    if run_all or args.anomaly: results["anomaly"] = train_anomaly()
    if run_all or args.theft: results["theft"] = train_theft()
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
