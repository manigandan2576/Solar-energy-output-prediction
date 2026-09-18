Solar Energy Output Prediction
==============================

This project forecasts next-day solar energy output in kWh for each panel array.
It uses the four supplied panel CSV files, merges generation readings with
weather sensor readings, creates an array-date training table, trains regression
models, and serves an interactive Streamlit predictor.

Problem Alignment
-----------------

The photographed problem statement asks for:

- next-day output prediction per array-date
- regression models such as Random Forest, Gradient Boosting, and linear models
- engineered interaction features such as irradiance x cloud cover
- RMSE and MAPE evaluation
- a prediction output file

The supplied CSV files include generation, ambient temperature, module
temperature, and irradiation. They do not directly include humidity, panel tilt
angle, dust, cloud cover, maintenance flags, or daylight hours. The pipeline
therefore uses available measured fields and engineered proxies or neutral
defaults for the missing problem-statement fields.

Run the Pipeline
----------------

From the project root:

```powershell
.venv\Scripts\python.exe src\prepare_data.py
.venv\Scripts\python.exe src\train_model.py
.venv\Scripts\python.exe src\predict.py
```

Run the Website
---------------

```powershell
.venv\Scripts\streamlit.exe run app.py
```

Outputs
-------

- `outputs/prepared_array_daily_data.csv`: merged array-date data with
  `next_day_output_kwh`.
- `models/random_forest.joblib`: trained Random Forest model.
- `models/gradient_boosting.joblib`: trained Gradient Boosting model.
- `models/ridge_regression.joblib`: trained linear baseline model.
- `outputs/model_metrics.csv`: MAE, RMSE, MAPE, and R2 scores.
- `outputs/submission.csv`: next-day predictions per array-date. The final
  `predicted_next_day_output_kwh` column uses the model with the lowest RMSE in
  `outputs/model_metrics.csv`.

Project Hygiene
---------------

Generated models, output CSVs, Python caches, and virtual environments are
ignored by `.gitignore`. Keep the four source CSV files in `data/`.
