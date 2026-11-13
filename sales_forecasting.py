"""
Sales Forecasting Capstone - reproducible example
Dataset: illustrative synthetic daily retail sales (not real business records).
Run: pip install pandas numpy matplotlib scikit-learn
     python src/sales_forecasting.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "dataset" / "daily_sales.csv", parse_dates=["date"])
df = df.sort_values("date").copy()

df["day_of_week"] = df.date.dt.dayofweek
df["month"] = df.date.dt.month
df["day_of_year"] = df.date.dt.dayofyear
df["day_of_month"] = df.date.dt.day
for lag in [1, 7, 14, 28]:
    df[f"lag_{lag}"] = df.sales.shift(lag)
df["rolling_mean_7"] = df.sales.shift(1).rolling(7).mean()
df["rolling_mean_28"] = df.sales.shift(1).rolling(28).mean()
df = df.dropna().reset_index(drop=True)

features = ["promotion", "day_of_week", "month", "day_of_year", "day_of_month",
            "lag_1", "lag_7", "lag_14", "lag_28", "rolling_mean_7", "rolling_mean_28"]
split = int(len(df) * 0.8)
train, test = df.iloc[:split], df.iloc[split:]

model = RandomForestRegressor(n_estimators=250, max_depth=12,
                              min_samples_leaf=3, random_state=42, n_jobs=-1)
model.fit(train[features], train["sales"])
prediction = model.predict(test[features])
baseline = test["lag_7"].to_numpy()

for name, actual, forecast in [
    ("Random Forest", test.sales, prediction),
    ("7-day seasonal-naive baseline", test.sales, baseline)
]:
    print(name)
    print("MAE:", round(mean_absolute_error(actual, forecast), 3))
    print("RMSE:", round(mean_squared_error(actual, forecast) ** 0.5, 3))
    print("R2:", round(r2_score(actual, forecast), 3))

plt.figure(figsize=(10, 4))
plt.plot(test.date, test.sales, label="Actual")
plt.plot(test.date, prediction, label="Random Forest")
plt.legend()
plt.xlabel("Date"); plt.ylabel("Sales units")
plt.title("Actual vs predicted sales")
plt.tight_layout()
plt.savefig(ROOT / "graphs" / "actual_vs_predicted.png", dpi=150)
print("Saved graph to graphs/actual_vs_predicted.png")
