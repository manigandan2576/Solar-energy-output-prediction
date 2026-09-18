from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

GENERATION_FILES = {
    1: "Plant_1_Generation_Data.csv",
    2: "Plant_2_Generation_Data.csv",
}

WEATHER_FILES = {
    1: "Plant_1_Weather_Sensor_Data.csv",
    2: "Plant_2_Weather_Sensor_Data.csv",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input file: {path}")
    return pd.read_csv(path)


def _parse_date_time(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(
            series.loc[missing],
            format="%d-%m-%Y %H:%M",
            errors="coerce",
        )
    return parsed


def load_plant_data(plant_number: int) -> pd.DataFrame:
    """Read and merge generation with plant-level weather sensor readings."""
    generation = _read_csv(DATA_DIR / GENERATION_FILES[plant_number])
    weather = _read_csv(DATA_DIR / WEATHER_FILES[plant_number])

    generation["DATE_TIME"] = _parse_date_time(generation["DATE_TIME"])
    weather["DATE_TIME"] = _parse_date_time(weather["DATE_TIME"])

    generation = generation.dropna(subset=["DATE_TIME"]).copy()
    weather = weather.dropna(subset=["DATE_TIME"]).copy()

    weather = weather.drop(columns=["SOURCE_KEY"], errors="ignore")
    merged = generation.merge(
        weather,
        on=["DATE_TIME", "PLANT_ID"],
        how="left",
        validate="many_to_one",
    )
    merged["plant_number"] = plant_number
    merged["date"] = merged["DATE_TIME"].dt.normalize()
    return merged


def _add_problem_features(daily: pd.DataFrame) -> pd.DataFrame:
    array_keys = ["plant_id", "array_id"]
    daily = daily.sort_values([*array_keys, "date"]).reset_index(drop=True)

    daily["prior_day_output_kwh"] = daily.groupby(array_keys)["daily_output_kwh"].shift(1)
    daily["next_day_output_kwh"] = daily.groupby(array_keys)["daily_output_kwh"].shift(-1)
    daily["day_of_year"] = daily["date"].dt.dayofyear
    daily["month"] = daily["date"].dt.month
    daily["day"] = daily["date"].dt.day

    first_date = daily.groupby(array_keys)["date"].transform("min")
    daily["panel_age_years"] = (daily["date"] - first_date).dt.days / 365.25

    daily["solar_irradiance_7d_max"] = daily.groupby(array_keys)[
        "solar_irradiance"
    ].transform(lambda value: value.rolling(7, min_periods=1).max())
    daily["estimated_cloud_cover_pct"] = (
        100
        * (1 - daily["solar_irradiance"] / daily["solar_irradiance_7d_max"].replace(0, np.nan))
    ).clip(lower=0, upper=100)
    daily["cloud_cover_pct"] = daily["estimated_cloud_cover_pct"]

    daily["dust_accumulation_index"] = (
        daily.groupby(array_keys).cumcount() % 30
    ) / 30
    daily["irradiance_cloud_interaction"] = (
        daily["solar_irradiance"] * daily["cloud_cover_pct"]
    )
    daily["temperature_irradiance_interaction"] = (
        daily["temperature"] * daily["solar_irradiance"]
    )
    daily["module_temperature_delta"] = daily["module_temperature"] - daily["temperature"]
    daily["output_per_irradiance"] = (
        daily["daily_output_kwh"] / daily["solar_irradiance"].replace(0, np.nan)
    )

    rolling_perf = daily.groupby(array_keys)["output_per_irradiance"].transform(
        lambda value: value.rolling(7, min_periods=3).median()
    )
    daily["maintenance_flag"] = (
        (daily["output_per_irradiance"] < rolling_perf * 0.75)
        & daily["solar_irradiance"].gt(0)
    ).astype(int)
    daily["maintenance_priority_score"] = (
        100 * (1 - daily["output_per_irradiance"] / rolling_perf.replace(0, np.nan))
    ).clip(lower=0, upper=100)

    # Present in the printed problem statement but unavailable in these CSVs.
    # Neutral constants make the app/form schema match the problem statement,
    # while the README documents that these were not measured in the source data.
    daily["humidity"] = 50.0
    daily["panel_tilt_angle"] = 25.0
    return daily


def build_array_daily_dataset() -> pd.DataFrame:
    """Create one training row per panel array per date."""
    merged = pd.concat(
        [load_plant_data(plant_number) for plant_number in GENERATION_FILES],
        ignore_index=True,
    )

    daily = (
        merged.groupby(["PLANT_ID", "plant_number", "SOURCE_KEY", "date"], as_index=False)
        .agg(
            daily_output_kwh=("DAILY_YIELD", "max"),
            dc_power_mean=("DC_POWER", "mean"),
            dc_power_max=("DC_POWER", "max"),
            ac_power_mean=("AC_POWER", "mean"),
            ac_power_max=("AC_POWER", "max"),
            total_yield_end=("TOTAL_YIELD", "max"),
            solar_irradiance=("IRRADIATION", "sum"),
            irradiance_mean=("IRRADIATION", "mean"),
            temperature=("AMBIENT_TEMPERATURE", "mean"),
            temperature_max=("AMBIENT_TEMPERATURE", "max"),
            module_temperature=("MODULE_TEMPERATURE", "mean"),
            module_temperature_max=("MODULE_TEMPERATURE", "max"),
            daylight_hours=("IRRADIATION", lambda value: float(value.gt(0).sum() * 0.25)),
            generation_readings=("DATE_TIME", "count"),
        )
        .rename(columns={"PLANT_ID": "plant_id", "SOURCE_KEY": "array_id"})
    )

    return _add_problem_features(daily)


def identify_damaged_panels(daily: pd.DataFrame) -> pd.DataFrame:
    """Return the most recent array-level panel flags that look damaged."""
    required_columns = [
        "plant_id",
        "array_id",
        "date",
        "maintenance_flag",
        "maintenance_priority_score",
        "daily_output_kwh",
        "solar_irradiance",
    ]
    missing = [name for name in required_columns if name not in daily.columns]
    if missing:
        raise ValueError(f"Missing required columns for damage detection: {missing}")

    flagged = daily.loc[daily["maintenance_flag"] == 1, required_columns].copy()
    if flagged.empty:
        return pd.DataFrame(
            columns=[
                "plant_id",
                "array_id",
                "last_detected_date",
                "maintenance_priority_score",
                "daily_output_kwh",
                "solar_irradiance",
                "issue",
            ]
        )

    flagged["date"] = pd.to_datetime(flagged["date"])
    flagged = flagged.sort_values(["plant_id", "array_id", "date"], ascending=[True, True, False])
    latest = flagged.drop_duplicates(subset=["plant_id", "array_id"], keep="first").copy()
    latest = latest.rename(columns={"date": "last_detected_date"})
    latest["issue"] = "Likely damaged panel"
    latest["maintenance_priority_score"] = latest["maintenance_priority_score"].clip(lower=0, upper=100)
    return latest[
        [
            "plant_id",
            "array_id",
            "last_detected_date",
            "maintenance_priority_score",
            "daily_output_kwh",
            "solar_irradiance",
            "issue",
        ]
    ].reset_index(drop=True)


def build_daily_dataset() -> pd.DataFrame:
    """Backward-compatible alias for older scripts."""
    return build_array_daily_dataset()


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    daily = build_array_daily_dataset()
    output_path = OUTPUT_DIR / "prepared_array_daily_data.csv"
    daily.to_csv(output_path, index=False)
    print(f"Saved {len(daily)} array-date rows to {output_path}")
    print(f"Training rows with target: {daily['next_day_output_kwh'].notna().sum()}")


if __name__ == "__main__":
    main()
