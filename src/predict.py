from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

try:
    from prepare_data import build_array_daily_dataset
    from train_model import FEATURES, train_models
except ImportError:  # pragma: no cover
    from src.prepare_data import build_array_daily_dataset
    from src.train_model import FEATURES, train_models


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"
MODEL_NAMES = ["random_forest", "gradient_boosting", "ridge_regression"]


def _normalize_panel_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "panel_id" not in normalized.columns and "plant_id" in normalized.columns:
        normalized = normalized.rename(columns={"plant_id": "panel_id"})
    if "panel_number" not in normalized.columns and "plant_number" in normalized.columns:
        normalized = normalized.rename(columns={"plant_number": "panel_number"})
    return normalized


def _load_model(name: str):
    model_path = MODELS_DIR / f"{name}.joblib"
    if not model_path.exists():
        train_models()
    artifact = joblib.load(model_path)
    features = artifact.get("features", FEATURES)
    if "plant_id" in features or "plant_number" in features:
        train_models()
        artifact = joblib.load(model_path)
        features = artifact.get("features", FEATURES)
    return artifact["model"], features


def _best_model_name() -> str:
    metrics_path = OUTPUT_DIR / "model_metrics.csv"
    if not metrics_path.exists():
        train_models()
    metrics = pd.read_csv(metrics_path)
    return str(metrics.sort_values("rmse").iloc[0]["model"])


def make_predictions() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(exist_ok=True)
    daily = _normalize_panel_columns(build_array_daily_dataset()).sort_values(["panel_id", "array_id", "date"])
    latest_rows = daily.groupby(["panel_id", "array_id"], as_index=False).tail(1).copy()
    latest_rows["prediction_date"] = latest_rows["date"] + pd.Timedelta(days=1)
    latest_rows["day_of_year"] = latest_rows["prediction_date"].dt.dayofyear
    latest_rows["month"] = latest_rows["prediction_date"].dt.month
    latest_rows["day"] = latest_rows["prediction_date"].dt.day
    latest_rows["prior_day_output_kwh"] = latest_rows["daily_output_kwh"]

    output = latest_rows[
        [
            "panel_id",
            "panel_number",
            "array_id",
            "prediction_date",
            "daily_output_kwh",
            "solar_irradiance",
            "cloud_cover_pct",
            "estimated_cloud_cover_pct",
            "temperature",
            "module_temperature",
            "daylight_hours",
            "maintenance_flag",
            "maintenance_priority_score",
        ]
    ].rename(columns={"daily_output_kwh": "prior_day_output_kwh"})

    prediction_columns = []
    for model_name in MODEL_NAMES:
        model, features = _load_model(model_name)
        column = f"{model_name}_prediction_kwh"
        output[column] = model.predict(latest_rows[features]).clip(min=0)
        prediction_columns.append(column)

    best_model = _best_model_name()
    output["selected_model"] = best_model
    output["predicted_next_day_output_kwh"] = output[f"{best_model}_prediction_kwh"]
    output["prediction_date"] = output["prediction_date"].dt.strftime("%Y-%m-%d")
    return output.sort_values(["panel_id", "array_id"]).reset_index(drop=True)


def main() -> None:
    submission = make_predictions()
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved submission to {SUBMISSION_PATH}")
    print(submission.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
