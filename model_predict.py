# %% Libraries
import pandas as pd
import numpy as np
import pickle
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
data = pd.read_csv("./data/holdout_b29.csv", dtype=col_dtype, parse_dates=["Date"])

# sample for testing
data = data.sample(frac=0.1, replace=False, random_state=42)


# clean df
data_clean = clean_cols_test(data)

# drop customers col
data_clean = data_clean.drop("Customers", axis=1)


# %% Encodeders

# one-hot encoding
enc_cols = ["SchoolHoliday", "StateHoliday"]
data_clean = pd.get_dummies(data_clean, columns=enc_cols, drop_first=True)

# Expand Date
data_clean = date_expand(data_clean)
data_clean = data_clean.drop("Date", axis=1)

# cols_to_drop = ['Customers', 'SchoolHoliday_missing', 'StateHoliday_missing']
# data_clean = data_clean.drop(columns=cols_to_drop)

# Append Attributes
enrich_df = pd.read_csv("./data/enrich_data.csv")  # import CSV
data_merged = pd.merge(data_clean, enrich_df, how="left", on="Store")


# %% --- Predict

filename = "./models/rossman_linear_reg_model.sav"
loaded_model = pickle.load(open(filename, "rb"))

result = loaded_model.predict(data_merged)

# export read_csv
result.to_csv("./predictions/holdout_prediction.csv", index=False)

# %%
