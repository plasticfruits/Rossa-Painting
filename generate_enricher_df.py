# %% Libraries
import pandas as pd
import numpy as np
from utils import *


from sklearn.pipeline import make_pipeline
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler


# %%

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


data = pd.read_csv("./data/train.csv", dtype=col_dtype, parse_dates=["Date"])
data_clean = clean_cols(data)


# Date Expand
data_clean = date_expand(data_clean)


# %% Import Store static data

stor_col_dtype = {
    "Store": "Int64",
    "StoreType": str,
    "Assortment": str,
    "CompetitionDistance": "Int64",
}
# Add store static data
store = pd.read_csv("./data/store.csv", dtype=stor_col_dtype)

store_cols = ["Store", "StoreType", "Assortment", "CompetitionDistance"]
store_clean = store.loc[:, store_cols]

store_clean.head(3)
store_clean.dtypes

# # %% --- Enrich

data_enrich = pd.merge(data_clean, store_clean, how="left", on="Store")

data_enrich["avg_wday_sales"] = data_enrich.groupby(["Store", "DayOfWeek"])[
    "Sales"
].transform("mean")
data_enrich["avg_wyear_sales"] = data_enrich.groupby(["Store", "WeekOfYear"])[
    "Sales"
].transform("mean")
data_enrich["avg_monthly_sales"] = data_enrich.groupby(["Store", "Month"])[
    "Sales"
].transform("mean")
data_enrich["avg_wday_customes"] = data_enrich.groupby(["Store", "DayOfWeek"])[
    "Customers"
].transform("mean")
data_enrich["avg_wyear_customes"] = data_enrich.groupby(["Store", "WeekOfYear"])[
    "Customers"
].transform("mean")
data_enrich["avg_monthly_customes"] = data_enrich.groupby(["Store", "Month"])[
    "Customers"
].transform("mean")
data_enrich["avg_sales_school_holiday"] = data_enrich.groupby(
    ["Store", "SchoolHoliday"]
)["Sales"].transform("mean")
data_enrich["avg_sales_state_holiday"] = data_enrich.groupby(["Store", "StateHoliday"])[
    "Sales"
].transform("mean")
data_enrich["avg_sales_store_type"] = data_enrich.groupby(["StoreType"])[
    "Sales"
].transform("mean")
data_enrich.head(2)


enrich_cols = [
    "Store",
    # "StoreType",
    "Assortment",
    "CompetitionDistance",
    "avg_wday_sales",
    "avg_wyear_sales",
    "avg_monthly_sales",
    "avg_wday_customes",
    "avg_wyear_customes",
    "avg_monthly_customes",
    "avg_sales_school_holiday",
    "avg_sales_state_holiday",
    "avg_sales_store_type",
]

enrich_df = data_enrich.loc[:, enrich_cols]
enrich_df = pd.get_dummies(enrich_df, columns=["Assortment"], drop_first=True)


# export read_csv
enrich_df.to_csv("./data/enrich_data.csv", index=False)
