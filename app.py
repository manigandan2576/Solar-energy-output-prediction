from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st  # type: ignore[import-not-found]


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
MODELS_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from prepare_data import build_array_daily_dataset, identify_damaged_panels  # noqa: E402
from predict import MODEL_NAMES, make_predictions  # noqa: E402
from train_model import FEATURES, METRICS_PATH, train_models  # noqa: E402

ENSEMBLE_MODEL_NAMES = ["random_forest", "gradient_boosting"]

st.set_page_config(
    page_title="Solar energy output prediction",
    page_icon=":material/solar_power:",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_daily_data() -> pd.DataFrame:
    return build_array_daily_dataset()


@st.cache_data(show_spinner=False, ttl=30)
def load_damaged_panels() -> pd.DataFrame:
    return identify_damaged_panels(load_daily_data())


@st.cache_data(show_spinner=False, ttl=30)
def load_metrics() -> pd.DataFrame:
    if not METRICS_PATH.exists():
        train_models()
    return pd.read_csv(METRICS_PATH)


@st.cache_resource(show_spinner=False)
def load_models() -> dict[str, dict]:
    missing = [name for name in MODEL_NAMES if not (MODELS_DIR / f"{name}.joblib").exists()]
    if missing:
        train_models()
    return {name: joblib.load(MODELS_DIR / f"{name}.joblib") for name in MODEL_NAMES}


def load_submission() -> pd.DataFrame:
    submission_path = OUTPUT_DIR / "submission.csv"
    if not submission_path.exists():
        make_predictions().to_csv(submission_path, index=False)
    return pd.read_csv(submission_path)


def predict_one(row: pd.Series, models: dict[str, dict], model_choice: str) -> dict[str, float]:
    frame = pd.DataFrame([row])
    predictions = {
        name: float(artifact["model"].predict(frame[artifact.get("features", FEATURES)])[0])
        for name, artifact in models.items()
    }
    predictions = {name: max(0.0, value) for name, value in predictions.items()}
    if model_choice == "average":
        predictions["average"] = sum(predictions[name] for name in ENSEMBLE_MODEL_NAMES) / len(
            ENSEMBLE_MODEL_NAMES
        )
    return predictions


def model_label(name: str) -> str:
    labels = {
        "average": "Average",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
        "ridge_regression": "Ridge Regression",
    }
    if name == "best_model":
        best_name = globals().get("best_model_name")
        if not isinstance(best_name, str):
            return "Best model"
        best_label = labels.get(best_name)
        return f"Best model ({best_label})" if best_label else "Best model"
    return labels.get(name, name)


def build_overall_prediction_view(submission_df: pd.DataFrame, grain: str) -> pd.DataFrame:
    predictions = submission_df.copy()
    predictions["prediction_date"] = pd.to_datetime(predictions["prediction_date"])

    if grain == "Daily":
        daily_view = predictions.sort_values(["prediction_date", "plant_id", "array_id"]).copy()
        daily_view.insert(0, "prediction_grain", "daily")
        daily_view["prediction_date"] = daily_view["prediction_date"].dt.strftime("%Y-%m-%d")
        return daily_view

    predictions["prediction_month"] = predictions["prediction_date"].dt.to_period("M").astype(str)
    monthly_view = (
        predictions.groupby(["prediction_month", "plant_id", "plant_number", "selected_model"], as_index=False)
        .agg(
            array_count=("array_id", "nunique"),
            total_prior_day_output_kwh=("prior_day_output_kwh", "sum"),
            total_predicted_output_kwh=("predicted_next_day_output_kwh", "sum"),
            avg_predicted_output_per_array_kwh=("predicted_next_day_output_kwh", "mean"),
            avg_solar_irradiance=("solar_irradiance", "mean"),
            avg_cloud_cover_pct=("cloud_cover_pct", "mean"),
            avg_temperature=("temperature", "mean"),
            avg_module_temperature=("module_temperature", "mean"),
            avg_daylight_hours=("daylight_hours", "mean"),
            maintenance_flag_count=("maintenance_flag", "sum"),
            avg_maintenance_priority_score=("maintenance_priority_score", "mean"),
        )
        .sort_values(["prediction_month", "plant_id"])
    )
    monthly_view.insert(0, "prediction_grain", "monthly")
    return monthly_view


st.title("Solar energy output prediction")
st.caption("Next-day kWh forecasts per panel array using the trained project models.")

daily = load_daily_data()
metrics = load_metrics()
models = load_models()
submission = load_submission()
damaged_panels = load_damaged_panels()
best_model_name = str(metrics.sort_values("rmse").iloc[0]["model"])

with st.sidebar:
    st.header("Prediction setup")
    array_options = (
        daily[["plant_id", "array_id"]]
        .drop_duplicates()
        .sort_values(["plant_id", "array_id"])
        .itertuples(index=False, name=None)
    )
    panel_id_col = "panel_id" if "panel_id" in daily.columns else "plant_id"
    panel_options = daily[[panel_id_col, "array_id"]].drop_duplicates().sort_values([panel_id_col, "array_id"]).itertuples(index=False, name=None)
    selected_plant, selected_array = st.selectbox(
        "Panel and array",
        list(panel_options),
        format_func=lambda value: f"{value[0]} | {value[1]}",
    )
    model_choice = st.segmented_control(
        "Model",
        ["best_model", "average", *MODEL_NAMES],
        default="best_model",
        format_func=model_label,
        required=True,
        width="stretch",
        wrap=True,
    )

array_history = daily[
    (daily["plant_id"] == selected_plant) & (daily["array_id"] == selected_array)
].sort_values("date")
latest = array_history.iloc[-1].copy()
default_prediction_date = latest["date"] + pd.Timedelta(days=1)

st.subheader("Forecast controls")
with st.form("forecast_form"):
    with st.container(horizontal=True):
        prediction_date = st.date_input(
            "Prediction date",
            value=default_prediction_date.date(),
        )
        solar_irradiance = st.number_input(
            "Solar irradiance",
            min_value=0.0,
            value=float(latest["solar_irradiance"]),
            step=0.1,
        )
        cloud_cover_pct = st.number_input(
            "Cloud cover (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(latest["cloud_cover_pct"]),
            step=1.0,
        )
        daylight_hours = st.number_input(
            "Daylight hours",
            min_value=0.0,
            max_value=24.0,
            value=float(latest["daylight_hours"]),
            step=0.25,
        )

    with st.container(horizontal=True):
        temperature = st.number_input(
            "Temperature",
            value=float(latest["temperature"]),
            step=0.5,
        )
        module_temperature = st.number_input(
            "Module temperature",
            value=float(latest["module_temperature"]),
            step=0.5,
        )
        prior_day_output = st.number_input(
            "Prior-day output (kWh)",
            min_value=0.0,
            value=float(latest["daily_output_kwh"]),
            step=10.0,
        )
        maintenance_flag = st.selectbox(
            "Maintenance flag",
            [0, 1],
            index=int(latest["maintenance_flag"]),
            format_func=lambda value: "Yes" if value else "No",
        )

    with st.expander("Advanced equipment and weather inputs"):
        with st.container(horizontal=True):
            dust_accumulation_index = st.number_input(
                "Dust accumulation index",
                min_value=0.0,
                max_value=1.0,
                value=float(latest["dust_accumulation_index"]),
                step=0.05,
            )
            humidity = st.number_input(
                "Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(latest["humidity"]),
                step=1.0,
            )
            panel_tilt_angle = st.number_input(
                "Panel tilt angle",
                min_value=0.0,
                max_value=90.0,
                value=float(latest["panel_tilt_angle"]),
                step=1.0,
            )
            maintenance_priority_score = st.number_input(
                "Maintenance priority score",
                min_value=0.0,
                max_value=100.0,
                value=float(latest["maintenance_priority_score"]),
                step=1.0,
            )

    submitted = st.form_submit_button("Predict output", icon=":material/bolt:", type="primary")

forecast_row = latest.copy()
forecast_row["date"] = pd.Timestamp(prediction_date)
forecast_row["solar_irradiance"] = solar_irradiance
forecast_row["irradiance_mean"] = solar_irradiance / max(float(latest["generation_readings"]), 1.0)
forecast_row["cloud_cover_pct"] = cloud_cover_pct
forecast_row["estimated_cloud_cover_pct"] = cloud_cover_pct
forecast_row["temperature"] = temperature
forecast_row["temperature_max"] = max(float(latest["temperature_max"]), temperature)
forecast_row["module_temperature"] = module_temperature
forecast_row["module_temperature_max"] = max(float(latest["module_temperature_max"]), module_temperature)
forecast_row["daylight_hours"] = daylight_hours
forecast_row["daily_output_kwh"] = prior_day_output
forecast_row["prior_day_output_kwh"] = prior_day_output
forecast_row["maintenance_flag"] = maintenance_flag
forecast_row["dust_accumulation_index"] = dust_accumulation_index
forecast_row["humidity"] = humidity
forecast_row["panel_tilt_angle"] = panel_tilt_angle
forecast_row["maintenance_priority_score"] = maintenance_priority_score
forecast_row["day_of_year"] = pd.Timestamp(prediction_date).dayofyear
forecast_row["month"] = pd.Timestamp(prediction_date).month
forecast_row["day"] = pd.Timestamp(prediction_date).day
forecast_row["panel_age_years"] = float(latest["panel_age_years"]) + (
    pd.Timestamp(prediction_date) - latest["date"]
).days / 365.25
forecast_row["irradiance_cloud_interaction"] = solar_irradiance * cloud_cover_pct
forecast_row["temperature_irradiance_interaction"] = temperature * solar_irradiance
forecast_row["module_temperature_delta"] = module_temperature - temperature
forecast_row["output_per_irradiance"] = prior_day_output / solar_irradiance if solar_irradiance else 0.0

effective_model_choice = best_model_name if model_choice == "best_model" else (model_choice or "average")
predictions = predict_one(forecast_row, models, effective_model_choice)
if submitted:
    st.session_state["latest_prediction"] = {
        "model_choice": effective_model_choice,
        "predictions": predictions,
        "prior_day_output": prior_day_output,
        "array_id": selected_array,
        "plant_id": selected_plant,
    }

latest_prediction = st.session_state.get("latest_prediction")
shown_prediction = None
if latest_prediction:
    shown_prediction = latest_prediction["predictions"][latest_prediction["model_choice"]]

with st.container(horizontal=True):
    if shown_prediction is None:
        st.metric("Predicted next-day output", "Run prediction", border=True)
    else:
        st.metric(
            "Predicted next-day output",
            f"{shown_prediction:,.0f} kWh",
            border=True,
        )
    st.metric(
        "Prior-day output",
        f"{prior_day_output:,.0f} kWh",
        border=True,
    )
    st.metric(
        "Selected array",
        f"{selected_plant} | {selected_array}",
        border=True,
    )

damage_id_column = "panel_id" if "panel_id" in damaged_panels.columns else "plant_id"
selected_damage = damaged_panels[
    (damaged_panels[damage_id_column] == selected_plant)
    & (damaged_panels["array_id"] == selected_array)
]

with st.container(border=True):
    st.subheader("Panel health check")
    if selected_damage.empty:
        st.success(f"No likely damaged panel detected for Panel {selected_plant} | Array {selected_array}.")
    else:
        damaged_row = selected_damage.iloc[0]
        st.warning(
            "Likely damaged panel detected: "
            f"Panel {selected_plant} | Array {selected_array} | "
            f"Last flagged {pd.to_datetime(damaged_row['last_detected_date']).strftime('%Y-%m-%d')} | "
            f"Priority score {float(damaged_row['maintenance_priority_score']):.0f}/100"
        )

with st.container(border=True):
    st.subheader("Damaged panels detected")
    if damaged_panels.empty:
        st.info("No damaged panel has been detected in the current dataset.")
    else:
        display_damaged = damaged_panels.copy()
        display_damaged["last_detected_date"] = pd.to_datetime(display_damaged["last_detected_date"]).dt.strftime("%Y-%m-%d")
        display_damaged["maintenance_priority_score"] = display_damaged["maintenance_priority_score"].round(1)
        st.dataframe(display_damaged, hide_index=True)

left, right = st.columns(2)
with left:
    with st.container(border=True):
        st.subheader("Array output history")
        chart_data = array_history[["date", "daily_output_kwh"]].rename(
            columns={"daily_output_kwh": "Daily output (kWh)"}
        )
        st.line_chart(chart_data, x="date", y="Daily output (kWh)")

with right:
    with st.container(border=True):
        st.subheader("Model metrics")
        st.dataframe(
            metrics,
            hide_index=True,
            column_config={
                "mae": st.column_config.NumberColumn("MAE", format="%.2f"),
                "rmse": st.column_config.NumberColumn("RMSE", format="%.2f"),
                "mape_pct": st.column_config.NumberColumn("MAPE (%)", format="%.2f"),
                "r2": st.column_config.NumberColumn("R2", format="%.3f"),
            },
        )

with st.container(border=True):
    st.subheader("Prediction details")
    if latest_prediction is None:
        st.info("Choose inputs and click Predict output to calculate a custom forecast.")
    else:
        prediction_table = pd.DataFrame(
            {
                "model": [model_label(name) for name in latest_prediction["predictions"]],
                "predicted_next_day_output_kwh": [
                    latest_prediction["predictions"][name]
                    for name in latest_prediction["predictions"]
                ],
            }
        )
        st.dataframe(
            prediction_table,
            hide_index=True,
            column_config={
                "predicted_next_day_output_kwh": st.column_config.NumberColumn(
                    "Predicted next-day output (kWh)",
                    format="%.2f",
                )
            },
        )

with st.container(border=True):
    st.subheader("Latest generated submission")
    st.dataframe(submission.head(50), hide_index=True)
    st.download_button(
        "Download submission.csv",
        data=submission.to_csv(index=False).encode("utf-8"),
        file_name="submission.csv",
        mime="text/csv",
        icon=":material/download:",
    )

with st.container(border=True):
    st.subheader("Overall prediction download")
    prediction_grain = st.segmented_control(
        "Prediction data",
        ["Daily", "Monthly"],
        default="Daily",
        required=True,
        width="stretch",
    )
    overall_predictions = build_overall_prediction_view(submission, prediction_grain or "Daily")
    st.dataframe(
        overall_predictions,
        hide_index=True,
        column_config={
            "predicted_next_day_output_kwh": st.column_config.NumberColumn(
                "Predicted output (kWh)",
                format="%.2f",
            ),
            "total_predicted_output_kwh": st.column_config.NumberColumn(
                "Total predicted output (kWh)",
                format="%.2f",
            ),
            "avg_predicted_output_per_array_kwh": st.column_config.NumberColumn(
                "Avg predicted output per array (kWh)",
                format="%.2f",
            ),
        },
    )
    file_stem = "overall_daily_predictions" if prediction_grain == "Daily" else "overall_monthly_predictions"
    st.download_button(
        f"Download {prediction_grain.lower()} predictions CSV",
        data=overall_predictions.to_csv(index=False).encode("utf-8"),
        file_name=f"{file_stem}.csv",
        mime="text/csv",
        icon=":material/download:",
    )

with st.expander("Data note"):
    st.write(
        "The provided CSV files contain generation, ambient temperature, module "
        "temperature, and irradiation. The problem statement fields for humidity, "
        "panel tilt angle, dust, cloud cover, maintenance, and daylight hours are "
        "not all directly measured, so the pipeline uses neutral defaults or "
        "engineered proxies where source values are unavailable."
    )
