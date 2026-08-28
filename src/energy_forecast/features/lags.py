# lags.py
import pandas as pd

def create_lag_features(df: pd.DataFrame, lags: list[int], column: str = "nd") -> pd.DataFrame:
    """Create lag features for the given column at the specified lags (in steps)."""
    df = df.copy()
    for lag in lags:
        df[f"{column}_lag_{lag}"] = df[column].shift(lag)
    return df


def create_rolling_features(df: pd.DataFrame, windows: list[int], column: str = "nd") -> pd.DataFrame:
    """Create rolling mean/std features for the given column over the specified windows (in steps)."""
    df = df.copy()
    for window in windows:
        df[f"{column}_rolling_mean_{window}"] = df[column].shift(1).rolling(window).mean()
        df[f"{column}_rolling_std_{window}"] = df[column].shift(1).rolling(window).std()
    return df