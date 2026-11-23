from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

UCI_FILE = RAW_DIR / "household_power_consumption.txt"
SGCC_FILE = RAW_DIR / "sgcc.csv"

for directory in (RAW_DIR, PROCESSED_DIR, MODEL_DIR, REPORT_DIR):
    directory.mkdir(parents=True, exist_ok=True)
