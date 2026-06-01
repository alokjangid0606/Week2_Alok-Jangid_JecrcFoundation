# Generated from: tesla-sales-forecasting (1).ipynb
# Converted at: 2026-05-31T16:47:31.606Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os

for root, dirs, files in os.walk("/kaggle/input"):
    
    for file in files:
        
        if file.endswith(".csv"):
            print(os.path.join(root, file))

# loading dataset
df = pd.read_csv("/kaggle/input/datasets/nalisha/tesla-ea-deliveries-and-production-data20152025/tesla_deliveries_dataset_2015_2025.csv")

# first 5 rows

df.head()

# last 5 rows

df.tail()

df.shape

df.info()

# Missing Values Check
missing_values = df.isnull().sum()

print(missing_values)

# Duplicate Rows Check
duplicate_rows = df.duplicated().sum()

print("Duplicate Rows :", duplicate_rows)

# Data Quality Report
report = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.values,
    "Missing Values": df.isnull().sum().values,
    "Unique Values": df.nunique().values
})

report

df.describe()

# Separate Numerical and Categorical Features
num_cols = []
cat_cols = []

for col in df.columns:

    if df[col].dtype == "object":
        cat_cols.append(col)
    else:
        num_cols.append(col)

print("Numerical Features")
print(num_cols)

print()

print("Categorical Features")
print(cat_cols)

# # EDA Phase 1


# Numerical Features
num_cols = [
    "Year",
    "Month",
    "Production_Units",
    "Avg_Price_USD",
    "Battery_Capacity_kWh",
    "Range_km",
    "CO2_Saved_tons",
    "Charging_Stations"
]

# Categorical Features
cat_cols = [
    "Region",
    "Model",
    "Source_Type"
]

plt.figure(figsize=(8,5))

sns.histplot(
    df["Estimated_Deliveries"],
    kde=True
)

plt.title("Estimated Deliveries Distribution")

plt.show()

print(df["Estimated_Deliveries"].describe())

# Numerical Features Distribution
for col in num_cols:

    plt.figure(figsize=(6,4))

    sns.histplot(
        df[col],
        kde=True
    )

    plt.title(col)

    plt.show()

# Outlier Detection
for col in num_cols:

    plt.figure(figsize=(6,2))

    sns.boxplot(
        x=df[col]
    )

    plt.title(col)

    plt.show()

# Correlation Analysis
corr_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(10,8))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm"
)

plt.show()

# Deliveries Correlation
corr_matrix["Estimated_Deliveries"]\
.sort_values(
    ascending=False
)

# Categorical Analysis
# Deliveries by Region
plt.figure(figsize=(10,5))

sns.barplot(
    data=df,
    x="Region",
    y="Estimated_Deliveries"
)

plt.xticks(rotation=45)

plt.show()

# Deliveries by Model
plt.figure(figsize=(10,5))

sns.barplot(
    data=df,
    x="Model",
    y="Estimated_Deliveries"
)

plt.xticks(rotation=45)

plt.show()

# Time Trend
monthly_sales = (
    df.groupby(["Year"])
    ["Estimated_Deliveries"]
    .mean()
)

plt.figure(figsize=(10,5))

monthly_sales.plot(
    marker="o"
)

plt.ylabel("Average Deliveries")

plt.show()

corr_matrix["Estimated_Deliveries"]\
.sort_values(
    ascending=False
)

# # Feature Engineering


df_ml = df.copy()

 # Feature 1: Year-Month
df_ml["Year_Month"] = (
    df_ml["Year"].astype(str)
    + "-"
    + df_ml["Month"].astype(str)
)

# quarter feature

df_ml["Quarter"] = ((df_ml["Month"] - 1) // 3) + 1

# month cyclic encoding

import numpy as np

df_ml["Month_sin"] = np.sin(
    2 * np.pi * df_ml["Month"] / 12
)

df_ml["Month_cos"] = np.cos(
    2 * np.pi * df_ml["Month"] / 12
)

# price per km

df_ml["Price_Per_KM"] = (
    df_ml["Avg_Price_USD"] /
    df_ml["Range_km"]
)

# co2 saved per charging station

# co2 saved per charging station

df_ml["CO2_Per_Station"] = (
    df_ml["CO2_Saved_tons"] /
    df_ml["Charging_Stations"]
)

df_ml.head()

df_ml.columns.tolist()

# # Encoding


cat_cols = [
    "Region",
    "Model",
    "Source_Type"
]

 # One Hot Encoding
df_encoded = pd.get_dummies(
    df_ml,
    columns=cat_cols,
    drop_first=True
)

print(df_encoded.shape)

df_encoded.columns.tolist()

# Target aur Features
X = df_encoded.drop("Estimated_Deliveries", axis=1)

y = df_encoded["Estimated_Deliveries"]

print(X.select_dtypes(include="object").columns.tolist())

# remove string column

X = X.drop("Year_Month", axis=1)

# Train Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

# # Evaluation


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

print("MAE :", round(mae, 2))
print("RMSE :", round(rmse, 2))
print("R2 Score :", round(r2, 4))

# Random Forest
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

# Evaluation
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

rf_mae = mean_absolute_error(y_test, rf_pred)

rf_rmse = np.sqrt(
    mean_squared_error(y_test, rf_pred)
)

rf_r2 = r2_score(y_test, rf_pred)

print("\nRandom Forest Results")
print("                      ")
print("MAE :", round(rf_mae, 2))
print("RMSE :", round(rf_rmse, 2))
print("R2 Score :", round(rf_r2, 4))

# # Hyperparameter Tuning


from sklearn.model_selection import GridSearchCV

params = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15]
}

grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    params,
    cv=3,
    scoring="r2"
)

grid.fit(X_train, y_train)

print("\nBest Parameters :", grid.best_params_)
print("Best CV Score :", round(grid.best_score_, 4))
print("\nModel Comparison")
print("----------------------")
print("Linear Regression R2 :", round(r2, 4))
print("Random Forest R2 :", round(rf_r2, 4))

# # Time Series Forecasting


monthly_data = (
    df.groupby(["Year", "Month"])
    ["Estimated_Deliveries"]
    .mean()
    .reset_index()
)

monthly_data.head()

# Create Data Column
monthly_data["Date"] = pd.to_datetime(
    monthly_data["Year"].astype(str)
    + "-"
    + monthly_data["Month"].astype(str)
    + "-01"
)

monthly_data = monthly_data.sort_values("Date")

plt.figure(figsize=(12,5))

plt.plot(
    monthly_data["Date"],
    monthly_data["Estimated_Deliveries"]
)

plt.title("Tesla Deliveries Trend")

plt.show()