# %% Libraries
import pandas as pd
import numpy as np
from utils import *

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# %% Import test data

# define data types for columns
col_dtype = {
    "Date": str,
    "Store": "Int64",
    "DayOfWeek": "Int64",
    "Sales": "float64",
    "Customers": "Int64",
    "Open": str,
    "Promo": str,
    "StateHoliday": str,
    "SchoolHoliday": str,
}

# read CSV
data = pd.read_csv("./data/train.csv", dtype=col_dtype, parse_dates=["Date"])

# sample data -- NOTE: temporary addition to improve performance
data = data.sample(frac=0.01, replace=False, random_state=42)

# clean df
data_clean = clean_cols(data)

data_clean.head(3)
data_clean.dtypes
# %% Encodeders

# one-hot encoding
enc_cols = ["SchoolHoliday", "StateHoliday"]
data_clean = pd.get_dummies(data_clean, columns=enc_cols, drop_first=True)

# Expand Date
data_clean = date_expand(data_clean)
data_clean = data_clean.drop("Date", axis=1)

# Append Attributes
enrich_df = pd.read_csv("./data/enrich_data.csv")  # import CSV
data_merged = pd.merge(data_clean, enrich_df, how="left", on="Store")


# # %% --- Train / Test Split -- DELETE FOR SUBMISSION

# drop NA's
data_merged = data_merged.dropna()

y = data_merged.loc[:, "Sales"]
X = data_merged.drop(columns="Sales")

# train,test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=0
)

# %% --- Baseline 1: Mean

model_name = "Mean"
#  make pipe
pipe = make_pipeline(StandardScaler(), DummyClassifier())
pipe.fit(X_train, y_train)  # apply scaling on training data
score = pipe.score(X_test, y_test)

y_pred = pipe.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmspe = metric(y_pred, y_test.values)

# print results
print(f"{model_name}:\nScore (RMSPE) = {rmspe}\nR2 = {score}\nRMSE = {rmse}\n\n")


# %% --- Baseline 2: Linear Regression

model_name = "Linear Regression"
#  make pipe
pipe = make_pipeline(StandardScaler(), LinearRegression())
pipe.fit(X_train, y_train)  # apply scaling on training data
score = pipe.score(X_test, y_test)

y_pred = pipe.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmspe = metric(y_pred, y_test.to_numpy())

# print results
print(f"{model_name}:\nScore (RMSPE) = {rmspe}\nR2 = {score}\nRMSE = {rmse}\n\n")

# %% --- Random Forest

model_name = "Random Forest"
# make pipe
pipe = make_pipeline(StandardScaler(), RandomForestRegressor())
pipe.fit(X_train, y_train)  # apply scaling on training data
score = pipe.score(X_test, y_test)

y_pred = pipe.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmspe = metric(y_pred, y_test.to_numpy())

# print results
print(f"{model_name}:\nScore (RMSPE) = {rmspe}\nR2 = {score}\nRMSE = {rmse}\n\n")

# %% --- Gradient Boosting

model_name = "Gradient Boosting"
# make pipe
pipe = make_pipeline(StandardScaler(), GradientBoostingRegressor())
pipe.fit(X_train, y_train)  # apply scaling on training data
score = pipe.score(X_test, y_test)

y_pred = pipe.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmspe = metric(y_pred, y_test.to_numpy())

# print results
print(f"{model_name}:\nScore (RMSPE) = {rmspe}\nR2 = {score}\nRMSE = {rmse}\n\n")
