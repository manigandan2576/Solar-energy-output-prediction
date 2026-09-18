from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from prepare_data import build_array_daily_dataset
except ImportError:  # pragma: no cover
    from src.prepare_data import build_array_daily_dataset


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
PREPARED_DATA_PATH = OUTPUT_DIR / "prepared_array_daily_data.csv"
METRICS_PATH = OUTPUT_DIR / "model_metrics.csv"

TARGET = "next_day_output_kwh"

CATEGORICAL_FEATURES = ["panel_id", "array_id"]
NUMERIC_FEATURES = [
    "panel_number",
    "daily_output_kwh",
    "dc_power_mean",
    "dc_power_max",
    "ac_power_mean",
    "ac_power_max",
    "total_yield_end",
    "solar_irradiance",
    "irradiance_mean",
    "temperature",
    "temperature_max",
    "module_temperature",
    "module_temperature_max",
    "daylight_hours",
    "generation_readings",
    "prior_day_output_kwh",
    "day_of_year",
    "month",
    "day",
    "panel_age_years",
    "cloud_cover_pct",
    "estimated_cloud_cover_pct",
    "dust_accumulation_index",
    "irradiance_cloud_interaction",
    "temperature_irradiance_interaction",
    "module_temperature_delta",
    "output_per_irradiance",
    "maintenance_flag",
    "maintenance_priority_score",
    "humidity",
    "panel_tilt_angle",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def _ensure_panel_compatibility_columns(data: pd.DataFrame) -> pd.DataFrame:
    normalized = data.copy()
    if "panel_id" in normalized.columns and "plant_id" not in normalized.columns:
        normalized["plant_id"] = normalized["panel_id"]
    if "panel_number" in normalized.columns and "plant_number" not in normalized.columns:
        normalized["plant_number"] = normalized["panel_number"]
    return normalized


def load_training_data() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = _ensure_panel_compatibility_columns(build_array_daily_dataset())
    data.to_csv(PREPARED_DATA_PATH, index=False)
    return data


def _tree_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def _linear_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def _model_specs() -> dict[str, Pipeline]:
    return {
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", _tree_preprocessor()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=350,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                ("preprocessor", _tree_preprocessor()),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=300,
                        learning_rate=0.04,
                        max_depth=3,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "ridge_regression": Pipeline(
            steps=[
                ("preprocessor", _linear_preprocessor()),
                ("model", Ridge(alpha=5.0)),
            ]
        ),
    }


def _safe_mape(y_true: pd.Series, y_pred: np.ndarray) -> float:
    mask = y_true > 0
    if not mask.any():
        return float("nan")
    return float(mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100)


def train_models() -> tuple[dict[str, Pipeline], pd.DataFrame]:
    data = load_training_data()
    trainable = (
        data.dropna(subset=[TARGET, "prior_day_output_kwh"])
        .sort_values(["date", "panel_id", "array_id"])
        .copy()
    )
    if len(trainable) < 50:
        raise ValueError("Not enough array-date rows with next-day targets to train models.")

    split_date_index = max(1, int(trainable["date"].nunique() * 0.8))
    split_date = sorted(trainable["date"].unique())[split_date_index - 1]
    train_df = trainable[trainable["date"] <= split_date]
    test_df = trainable[trainable["date"] > split_date]
    if test_df.empty:
        raise ValueError("Date-based split produced no test rows.")

    x_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    x_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    metrics = []
    fitted_models: dict[str, Pipeline] = {}
    MODELS_DIR.mkdir(exist_ok=True)

    for name, model in _model_specs().items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        metrics.append(
            {
                "model": name,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "mae": mean_absolute_error(y_test, predictions),
                "rmse": mean_squared_error(y_test, predictions) ** 0.5,
                "mape_pct": _safe_mape(y_test, predictions),
                "r2": r2_score(y_test, predictions) if len(test_df) > 1 else float("nan"),
            }
        )

        final_model = _model_specs()[name]
        final_model.fit(trainable[FEATURES], trainable[TARGET])
        fitted_models[name] = final_model
        joblib.dump(
            {
                "model": final_model,
                "features": FEATURES,
                "numeric_features": NUMERIC_FEATURES,
                "categorical_features": CATEGORICAL_FEATURES,
                "target": TARGET,
            },
            MODELS_DIR / f"{name}.joblib",
        )

    metrics_df = pd.DataFrame(metrics).sort_values("rmse")
    metrics_df.to_csv(METRICS_PATH, index=False)
    return fitted_models, metrics_df


def main() -> None:
    _, metrics = train_models()
    print(f"Saved prepared data to {PREPARED_DATA_PATH}")
    print(f"Saved models to {MODELS_DIR}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
