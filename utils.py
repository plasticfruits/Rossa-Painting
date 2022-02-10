import numpy as np


def clean_cols(df):
    """
    input:
    Output:
    """
    # Clean Store
    df = df.dropna(subset=["Store"])
    # DayOfWeek
    df = df.drop("DayOfWeek", axis=1)
    # Open Col
    index_drop = df[(df["Sales"] < 1) & (df["Open"] != "1")].index
    df.drop(index_drop, inplace=True)
    df["Open"] = df["Open"].fillna("1")
    df = df.drop("Open", axis=1)
    # Promo
    df["Promo"] = df["Promo"].fillna("3")
    df["Promo"] = df["Promo"].astype(int)
    # StateHoliday
    df["StateHoliday"] = df["StateHoliday"].replace({"0": "d"})
    df["StateHoliday"] = df["StateHoliday"].fillna("missing")
    # SchoolHoliday
    df["SchoolHoliday"] = df["SchoolHoliday"].fillna("missing")

    return df


def date_expand(df):
    """
    Input: a df with column "Date"
    Out: expands multiple datetime features to columns
    """
    df["Year"] = df.Date.dt.year
    df["Month"] = df.Date.dt.month
    df["DayOfWeek"] = df.Date.dt.dayofweek
    df["WeekOfYear"] = df.Date.dt.isocalendar().week
    return df


def metric(preds, actuals):
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])
