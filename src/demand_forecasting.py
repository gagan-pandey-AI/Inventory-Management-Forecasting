import mysql.connector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from getpass import getpass
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

FEATURES = [
    "product_id",
    "day_of_week",
    "month",
    "day_of_year",
    "is_weekend",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28"
]


# ==========================================
# 2. MYSQL CONNECTION
# ==========================================

password = getpass("Enter MySQL root password: ")

db = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password=password,
    database="inventoryforecastdb"
)

print("\nMySQL connected successfully!")

query = """
SELECT
    s.sale_date,
    s.product_id,
    p.product_name,
    SUM(s.quantity_sold) AS quantity_sold
FROM sales s
JOIN products p
    ON s.product_id = p.product_id
GROUP BY
    s.sale_date,
    s.product_id,
    p.product_name
ORDER BY s.sale_date
"""

df = pd.read_sql(query, db)

products = (
    df[["product_id", "product_name"]]
    .drop_duplicates()
    .sort_values("product_id")
)

db.close()


# ==========================================
# 3. PREPARE DAILY SALES DATA
# ==========================================

df["sale_date"] = pd.to_datetime(df["sale_date"])

start_date = df["sale_date"].min()
end_date = df["sale_date"].max()

all_dates = pd.date_range(start_date, end_date, freq="D")

grid = pd.MultiIndex.from_product(
    [products["product_id"], all_dates],
    names=["product_id", "sale_date"]
).to_frame(index=False)

daily_sales = (
    df.groupby(["product_id", "sale_date"])["quantity_sold"]
    .sum()
    .reset_index()
)

daily = grid.merge(
    daily_sales,
    on=["product_id", "sale_date"],
    how="left"
)

daily["quantity_sold"] = (
    daily["quantity_sold"].fillna(0)
)

daily = daily.sort_values(
    ["product_id", "sale_date"]
).reset_index(drop=True)


# ==========================================
# 4. FEATURE ENGINEERING
# ==========================================

def create_features(data):

    data = data.copy()

    data["day_of_week"] = data["sale_date"].dt.dayofweek
    data["month"] = data["sale_date"].dt.month
    data["day_of_year"] = data["sale_date"].dt.dayofyear

    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    grouped = data.groupby("product_id")["quantity_sold"]

    data["lag_1"] = grouped.shift(1)
    data["lag_7"] = grouped.shift(7)
    data["lag_14"] = grouped.shift(14)
    data["lag_28"] = grouped.shift(28)

    data["rolling_mean_7"] = (
        grouped.transform(
            lambda x: x.shift(1).rolling(7).mean()
        )
    )

    data["rolling_mean_28"] = (
        grouped.transform(
            lambda x: x.shift(1).rolling(28).mean()
        )
    )

    return data


featured = create_features(daily).dropna().copy()

X = featured[FEATURES]
y = featured["quantity_sold"]


# ==========================================
# 5. TIME-BASED TRAIN/TEST SPLIT
# ==========================================

test_start = end_date - pd.Timedelta(days=29)

train_mask = featured["sale_date"] < test_start
test_mask = featured["sale_date"] >= test_start

X_train = X.loc[train_mask]
y_train = y.loc[train_mask]

X_test = X.loc[test_mask]
y_test = y.loc[test_mask]

if len(X_train) == 0 or len(X_test) == 0:
    raise ValueError("Not enough historical data for model evaluation.")


# ==========================================
# 6. TRAIN RANDOM FOREST MODEL
# ==========================================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# ==========================================
# 7. MODEL EVALUATION
# ==========================================

test_predictions = model.predict(X_test)

model_mae = mean_absolute_error(
    y_test,
    test_predictions
)

baseline_predictions = X_test["lag_7"]

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

print("\n========== MODEL EVALUATION ==========")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Random Forest MAE:", round(model_mae, 3))
print("7-day baseline MAE:", round(baseline_mae, 3))

# ==========================================
# 7.1 FORECAST PERFORMANCE ANALYSIS
# ==========================================

evaluation_df = featured.loc[
    test_mask,
    ["sale_date", "product_id", "quantity_sold"]
].copy()

evaluation_df["actual_sales"] = y_test.to_numpy()
evaluation_df["random_forest_prediction"] = test_predictions
evaluation_df["baseline_prediction"] = baseline_predictions.to_numpy()

evaluation_df["product_name"] = evaluation_df["product_id"].map(
    products.set_index("product_id")["product_name"]
)

evaluation_df = evaluation_df.rename(
    columns={"sale_date": "evaluation_date"}
)

# Save row-level actual vs predicted data
evaluation_df.to_csv(
    REPORT_DIR / "forecast_evaluation.csv",
    index=False
)

# Product-wise accuracy
product_accuracy = (
    evaluation_df.groupby("product_name")
    .apply(
        lambda group: pd.Series({
            "actual_units": group["actual_sales"].sum(),
            "predicted_units": group["random_forest_prediction"].sum(),
            "random_forest_mae": mean_absolute_error(
                group["actual_sales"],
                group["random_forest_prediction"]
            ),
            "baseline_mae": mean_absolute_error(
                group["actual_sales"],
                group["baseline_prediction"]
            )
        }),
        include_groups=False
    )
    .reset_index()
)

product_accuracy.to_csv(
    REPORT_DIR / "product_forecast_accuracy.csv",
    index=False
)

# Overall model comparison
improvement = (
    (baseline_mae - model_mae) / baseline_mae * 100
    if baseline_mae != 0 else 0
)

model_performance = pd.DataFrame([
    {
        "model": "Random Forest",
        "MAE": model_mae
    },
    {
        "model": "7-Day Baseline",
        "MAE": baseline_mae
    }
])

model_performance.to_csv(
    REPORT_DIR / "model_performance.csv",
    index=False
)

# Daily actual vs predicted comparison
daily_comparison = (
    evaluation_df.groupby("evaluation_date")[
        ["actual_sales", "random_forest_prediction", "baseline_prediction"]
    ]
    .sum()
)

plt.figure(figsize=(13, 6))

plt.plot(
    daily_comparison.index,
    daily_comparison["actual_sales"],
    label="Actual Sales"
)

plt.plot(
    daily_comparison.index,
    daily_comparison["random_forest_prediction"],
    label="Random Forest Prediction"
)

plt.plot(
    daily_comparison.index,
    daily_comparison["baseline_prediction"],
    label="7-Day Baseline"
)

plt.title("Actual vs Predicted Sales - Model Evaluation")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "forecast_actual_comparison.png"
)

plt.close()

print("\n========== FORECAST PERFORMANCE ==========")
print("Random Forest MAE:", round(model_mae, 3))
print("Baseline MAE:", round(baseline_mae, 3))
print("MAE Improvement:", round(improvement, 2), "%")

print("\nProduct-wise Forecast Accuracy:")
print(product_accuracy.to_string(index=False))

print("\nEvaluation reports saved successfully!")


# ==========================================
# 7.1 FORECAST PERFORMANCE ANALYSIS
# ==========================================

evaluation_df = featured.loc[
    test_mask,
    ["sale_date", "product_id", "quantity_sold"]
].copy()

evaluation_df["actual_sales"] = y_test.to_numpy()
evaluation_df["random_forest_prediction"] = test_predictions
evaluation_df["baseline_prediction"] = baseline_predictions.to_numpy()

evaluation_df["product_name"] = evaluation_df["product_id"].map(
    products.set_index("product_id")["product_name"]
)

evaluation_df = evaluation_df.rename(
    columns={"sale_date": "evaluation_date"}
)

evaluation_df.to_csv(
    REPORT_DIR / "forecast_evaluation.csv",
    index=False
)

product_accuracy = (
    evaluation_df.groupby("product_name")
    .apply(
        lambda group: pd.Series({
            "actual_units": group["actual_sales"].sum(),
            "predicted_units": group["random_forest_prediction"].sum(),
            "random_forest_mae": mean_absolute_error(
                group["actual_sales"],
                group["random_forest_prediction"]
            ),
            "baseline_mae": mean_absolute_error(
                group["actual_sales"],
                group["baseline_prediction"]
            )
        }),
        include_groups=False
    )
    .reset_index()
)

product_accuracy.to_csv(
    REPORT_DIR / "product_forecast_accuracy.csv",
    index=False
)

improvement = (
    (baseline_mae - model_mae) / baseline_mae * 100
    if baseline_mae != 0 else 0
)

model_performance = pd.DataFrame([
    {"model": "Random Forest", "MAE": model_mae},
    {"model": "7-Day Baseline", "MAE": baseline_mae}
])

model_performance.to_csv(
    REPORT_DIR / "model_performance.csv",
    index=False
)

daily_comparison = (
    evaluation_df.groupby("evaluation_date")[
        [
            "actual_sales",
            "random_forest_prediction",
            "baseline_prediction"
        ]
    ].sum()
)

plt.figure(figsize=(13, 6))

plt.plot(
    daily_comparison.index,
    daily_comparison["actual_sales"],
    label="Actual Sales"
)

plt.plot(
    daily_comparison.index,
    daily_comparison["random_forest_prediction"],
    label="Random Forest Prediction"
)

plt.plot(
    daily_comparison.index,
    daily_comparison["baseline_prediction"],
    label="7-Day Baseline"
)

plt.title("Actual vs Predicted Sales - Model Evaluation")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "forecast_actual_comparison.png"
)

plt.close()

print("\n========== FORECAST PERFORMANCE ==========")
print("Random Forest MAE:", round(model_mae, 3))
print("Baseline MAE:", round(baseline_mae, 3))
print("MAE Improvement:", round(improvement, 2), "%")

print("\nProduct-wise Forecast Accuracy:")
print(product_accuracy.to_string(index=False))

print("\nEvaluation reports saved successfully!")


# ==========================================
# 8. TRAIN FINAL MODEL
# ==========================================

final_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

final_model.fit(X, y)


# ==========================================
# 9. FORECAST NEXT 30 DAYS
# ==========================================

forecast_records = []

forecast_dates = pd.date_range(
    end_date + pd.Timedelta(days=1),
    periods=30,
    freq="D"
)

for _, product in products.iterrows():

    product_id = int(product["product_id"])
    product_name = product["product_name"]

    history = daily.loc[
        daily["product_id"] == product_id,
        "quantity_sold"
    ].tolist()

    for forecast_date in forecast_dates:

        weekday = forecast_date.dayofweek

        row = pd.DataFrame([{
            "product_id": product_id,
            "day_of_week": weekday,
            "month": forecast_date.month,
            "day_of_year": forecast_date.dayofyear,
            "is_weekend": int(weekday >= 5),
            "lag_1": history[-1],
            "lag_7": history[-7],
            "lag_14": history[-14],
            "lag_28": history[-28],
            "rolling_mean_7": np.mean(history[-7:]),
            "rolling_mean_28": np.mean(history[-28:])
        }])

        prediction = final_model.predict(row[FEATURES])[0]

        prediction = max(0, int(round(prediction)))

        history.append(prediction)

        forecast_records.append({
            "product_id": product_id,
            "product_name": product_name,
            "forecast_date": forecast_date,
            "predicted_demand": prediction
        })


forecast_df = pd.DataFrame(forecast_records)


# ==========================================
# 10. SAVE FORECAST REPORT
# ==========================================

forecast_df.to_csv(
    REPORT_DIR / "demand_forecast.csv",
    index=False
)

summary = (
    forecast_df.groupby("product_name")["predicted_demand"]
    .sum()
    .reset_index()
    .sort_values("predicted_demand", ascending=False)
)

summary.to_csv(
    REPORT_DIR / "forecast_summary.csv",
    index=False
)


# ==========================================
# 11. FORECAST VISUALIZATION
# ==========================================

historical_total = (
    daily.groupby("sale_date")["quantity_sold"]
    .sum()
)

forecast_total = (
    forecast_df.groupby("forecast_date")["predicted_demand"]
    .sum()
)

plt.figure(figsize=(12, 6))

plt.plot(
    historical_total.index,
    historical_total.values,
    label="Historical Sales"
)

plt.plot(
    forecast_total.index,
    forecast_total.values,
    label="Predicted Demand"
)

plt.title("Historical Sales vs 30-Day Demand Forecast")
plt.xlabel("Date")
plt.ylabel("Units")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "forecast_comparison.png"
)

plt.close()


# ==========================================
# 12. PRINT FINAL RESULTS
# ==========================================

print("\n========== 30-DAY DEMAND FORECAST ==========")

print(
    summary.to_string(index=False)
)

print(
    "\nTotal predicted demand:",
    forecast_df["predicted_demand"].sum(),
    "units"
)

print("\nForecast period: November 1–30, 2026")

print("\nReports saved in:", REPORT_DIR)

print("\nDEMAND FORECASTING COMPLETED SUCCESSFULLY!")