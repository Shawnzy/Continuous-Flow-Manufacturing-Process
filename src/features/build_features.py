import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_pickle("../../data/interim/data_processed.pkl")

# df["Stage1.Output.Measurement0.U.Actual"].plot()


def clean_time_series(series, threshold_factor=1.5):
    """
    Remove extreme values from a time series and fill missing values with linear interpolation.

    Args:
        series (pd.Series): Time series data
        threshold_factor (float): Factor to determine extreme values based on the interquartile range (IQR)

    Returns:
        pd.Series: Cleaned time series
    """
    cleaned_series = series.copy()

    # Calculate the interquartile range (IQR)
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - threshold_factor * IQR
    upper_bound = Q3 + threshold_factor * IQR

    # Replace extreme values with NaN
    cleaned_series[
        (cleaned_series < lower_bound) | (cleaned_series > upper_bound)
    ] = np.nan

    # Fill missing values using linear interpolation
    cleaned_series.interpolate(method="linear", inplace=True)

    return cleaned_series


clean_series = clean_time_series(df["Stage1.Output.Measurement0.U.Actual"])
clean_series.head(100).plot()


def clean_time_series_rolling_window(series, window_size=5, num_std_dev=2):
    """
    Remove extreme values from a time series using a rolling window and fill missing values with linear interpolation.

    Args:
        series (pd.Series): Time series data
        window_size (int): Size of the rolling window
        threshold_factor (float): Factor to determine extreme values based on the rolling window mean and standard deviation

    Returns:
        pd.Series: Cleaned time series
    """
    cleaned_series = series.copy()

    # Calculate the rolling window mean and standard deviation
    rolling_mean = series.rolling(window=window_size).mean()
    rolling_std = series.rolling(window=window_size).std()

    # Define the lower and upper bounds for outliers
    lower_bound = rolling_mean - threshold_factor * rolling_std
    upper_bound = rolling_mean + threshold_factor * rolling_std

    # Replace extreme values with NaN
    cleaned_series[
        (cleaned_series < lower_bound)
        | (cleaned_series > upper_bound)
        | (cleaned_series == 0)
    ] = np.nan

    # Fill missing values using linear interpolation
    cleaned_series.interpolate(method="linear", inplace=True)

    return cleaned_series


cleaned_series = clean_time_series_rolling_window(
    df["Stage1.Output.Measurement0.U.Actual"], window_size=300, threshold_factor=3
)
cleaned_series.plot()
