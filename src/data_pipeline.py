"""Dataset auditing and preprocessing for EnergyGuard AI."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import UCI_FILE, SGCC_FILE, PROCESSED_DIR, REPORT_DIR

UCI_COLUMNS = [
    "Date", "Time", "Global_active_power", "Global_reactive_power", "Voltage",
    "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3",
]


def audit_uci(path: Path = UCI_FILE) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"UCI file not found: {path}. Download the dataset and place "
            "household_power_consumption.txt in data/raw/."
        )
    df = pd.read_csv(path, sep=";", na_values=["?", " ", ""], low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    missing_columns = set(UCI_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"UCI file is missing columns: {sorted(missing_columns)}")
    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        dayfirst=True, errors="coerce"
    )
    numeric = [c for c in UCI_COLUMNS[2:] if c in df]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    report = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "date_min": str(df["timestamp"].min()),
        "date_max": str(df["timestamp"].max()),
        "missing_by_column": df[numeric].isna().sum().astype(int).to_dict(),
        "duplicate_timestamps": int(df["timestamp"].duplicated().sum()),
        "invalid_timestamps": int(df["timestamp"].isna().sum()),
        "zero_active_power": int((df["Global_active_power"] == 0).sum()),
    }
    return report


def prepare_uci(path: Path = UCI_FILE) -> pd.DataFrame:
    """Return complete hourly energy and leak-safe time features."""
    if not path.exists():
        raise FileNotFoundError(f"Missing UCI data file: {path}")
    df = pd.read_csv(path, sep=";", na_values=["?", " ", ""], low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        dayfirst=True, errors="coerce"
    )
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.drop_duplicates("timestamp").set_index("timestamp")
    active = pd.to_numeric(df["Global_active_power"], errors="coerce")
    # Interpolate only short internal gaps; never fill dataset edges or long gaps.
    active = active.interpolate(method="time", limit=3, limit_area="inside")
    minute = active.to_frame("active_power_kw")
    grouped = minute["active_power_kw"].resample("h")
    hourly = pd.DataFrame({
        "energy_kwh": grouped.sum(min_count=1) / 60.0,
        "valid_readings": grouped.count(),
    })
    hourly = hourly.loc[hourly["valid_readings"] >= 54].copy()
    hourly = hourly.dropna(subset=["energy_kwh"])
    hourly.index.name = "timestamp"
    hourly["hour"] = hourly.index.hour
    hourly["weekday"] = hourly.index.dayofweek
    hourly["lag_1"] = hourly["energy_kwh"].shift(1)
    hourly["lag_2"] = hourly["energy_kwh"].shift(2)
    hourly["lag_24"] = hourly["energy_kwh"].shift(24)
    hourly["rolling_mean_3"] = hourly["energy_kwh"].shift(1).rolling(3).mean()
    hourly["rolling_mean_24"] = hourly["energy_kwh"].shift(1).rolling(24).mean()
    hourly["target_next_hour_kwh"] = hourly["energy_kwh"].shift(-1)
    hourly = hourly.dropna().reset_index()
    return hourly


def _looks_like_date(value: str) -> bool:
    return bool(re.search(r"(19|20)\d{2}[-_/]?(0[1-9]|1[0-2])", str(value)))


def _ordered_daily_columns(columns: list[str]) -> list[str]:
    # Parse common YYYYMMDD / YYYY-MM-DD / YYYY/MM/DD formats when possible.
    dated = []
    for col in columns:
        raw = str(col).strip()
        parsed = pd.to_datetime(raw, errors="coerce")
        if pd.notna(parsed):
            dated.append((parsed, col))
        else:
            digits = re.sub(r"\D", "", raw)
            if len(digits) == 8 and digits[:2] in ("19", "20"):
                parsed = pd.to_datetime(digits, format="%Y%m%d", errors="coerce")
                if pd.notna(parsed):
                    dated.append((parsed, col))
    if len(dated) >= max(3, int(0.5 * len(columns))):
        return [c for _, c in sorted(dated)]
    return columns


def prepare_sgcc(path: Path = SGCC_FILE) -> tuple[pd.DataFrame, list[str]]:
    """Create one row per consumer with aggregate and daily-pattern features."""
    if not path.exists():
        raise FileNotFoundError(
            f"SGCC file not found: {path}. Place the extracted consumer CSV at data/raw/sgcc.csv."
        )
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    if "FLAG" not in df.columns:
        raise ValueError("SGCC CSV must contain the FLAG target column.")
    if "CONS_NO" not in df.columns:
        raise ValueError("SGCC CSV must contain CONS_NO consumer identifiers.")
    df["FLAG"] = pd.to_numeric(df["FLAG"], errors="coerce")
    df = df.dropna(subset=["FLAG"]).copy()
    df["FLAG"] = df["FLAG"].astype(int)
    if not set(df["FLAG"].unique()).issubset({0, 1}):
        raise ValueError("FLAG must contain only binary labels 0 and 1.")
    daily_cols = [c for c in df.columns if c not in ("CONS_NO", "FLAG")]
    daily_cols = _ordered_daily_columns(daily_cols)
    daily = df[daily_cols].apply(pd.to_numeric, errors="coerce")
    # Interpolate only small internal gaps within each consumer's daily sequence.
    daily = daily.interpolate(axis=1, limit=3, limit_area="inside")
    features = pd.DataFrame(index=df.index)
    features["mean_daily"] = daily.mean(axis=1)
    features["std_daily"] = daily.std(axis=1)
    features["median_daily"] = daily.median(axis=1)
    features["min_daily"] = daily.min(axis=1)
    features["max_daily"] = daily.max(axis=1)
    features["zero_day_ratio"] = (daily == 0).sum(axis=1) / daily.notna().sum(axis=1).replace(0, np.nan)
    features["missing_day_ratio"] = daily.isna().sum(axis=1) / max(1, len(daily_cols))
    features["first_quarter_mean"] = daily.iloc[:, :max(1, len(daily_cols)//4)].mean(axis=1)
    features["last_quarter_mean"] = daily.iloc[:, -max(1, len(daily_cols)//4):].mean(axis=1)
    features["consumption_cv"] = features["std_daily"] / features["mean_daily"].replace(0, np.nan)
    result = features.replace([np.inf, -np.inf], np.nan)
    result["FLAG"] = df["FLAG"].values
    result["CONS_NO"] = df["CONS_NO"].astype(str).values
    return result.reset_index(drop=True), list(features.columns)


def run_audit() -> dict:
    audit = {}
    if UCI_FILE.exists():
        audit["uci"] = audit_uci()
    else:
        audit["uci"] = {"status": "missing", "expected_path": str(UCI_FILE)}
    if SGCC_FILE.exists():
        sgcc, features = prepare_sgcc()
        audit["sgcc"] = {
            "rows": int(len(sgcc)), "feature_count": len(features),
            "class_counts": sgcc["FLAG"].value_counts().sort_index().to_dict(),
            "missing_feature_cells": int(sgcc[features].isna().sum().sum()),
            "consumer_ids_unique": int(sgcc["CONS_NO"].nunique()),
        }
    else:
        audit["sgcc"] = {"status": "missing", "expected_path": str(SGCC_FILE)}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "dataset_audit.json").write_text(json.dumps(audit, indent=2, default=str))
    return audit


if __name__ == "__main__":
    print(json.dumps(run_audit(), indent=2, default=str))
    if UCI_FILE.exists():
        hourly = prepare_uci()
        hourly.to_csv(PROCESSED_DIR / "uci_hourly_features.csv", index=False)
        print(f"Saved {len(hourly):,} hourly rows.")
    if SGCC_FILE.exists():
        sgcc, _ = prepare_sgcc()
        sgcc.to_csv(PROCESSED_DIR / "sgcc_consumer_features.csv", index=False)
        print(f"Saved {len(sgcc):,} SGCC consumers.")
