import pandas as pd
import numpy as np

def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create calendar-based time series features from the datetime index.

    Args:
        df: time series dataframe with a datetime index

    Returns:
        df: copy of the input dataframe with added calendar features
    """
    df = df.copy()
    df["day_of_month"] = df.index.day
    df["day_of_week"] = df.index.day_of_week
    df["day_name"] = df.index.day_name()
    df["hour"] = df.index.hour
    df["day_of_year"] = df.index.day_of_year
    df["quarter"] = df.index.quarter
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["week_of_year"] = df.index.isocalendar().week.astype("int64")
    return df

def add_cyclical_encoding(df: pd.DataFrame, column: str, period: int):
    """Add sin/cos encoding for a cyclical column (e.g. hour, day_of_week, month)."""
    df = df.copy()
    df[f"{column}_sin"] = np.sin(2 * np.pi * df[column] / period)
    df[f"{column}_cos"] = np.cos(2 * np.pi * df[column] / period)
    return df