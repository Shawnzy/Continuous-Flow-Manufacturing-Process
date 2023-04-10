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


def clean_series(series, window_size=10, num_std_dev=3):
    """
    Clean a time series by removing extreme values and filling missing values using linear interpolation.

    Args:
        series (pd.Series): Time series data
        window_size (int): Size of the rolling window
        num_std_dev (int): The number of standard deviations to consider for identifying outliers

    Returns:
        pd.Series: Cleaned time series
    """

    # Calculate the rolling window mean and standard deviation
    moving_avg = series.rolling(window=window_size).mean()
    moving_std = series.rolling(window=window_size).std()

    # Identify outliers
    outliers = np.abs(series - moving_avg) > num_std_dev * moving_std

    # Replace outliers with NaN values
    cleaned_series = series.copy()
    cleaned_series[outliers] = np.nan

    # Replace 0 values with NaN values
    cleaned_series[cleaned_series == 0] = np.nan

    # Fill missing values using linear interpolation
    cleaned_series = cleaned_series.interpolate(method="linear")

    return cleaned_series


cleaned_series = clean_series(
    df["Stage1.Output.Measurement0.U.Actual"], window_size=100
)
cleaned_series.plot()

df["Stage1.Output.Measurement0.U.Actual"] = cleaned_series
df = df.iloc[:, :42]


def engineer_features(df, lag_features, window_size):
    """
    Engineer lag features along with their rolling mean, rolling standard deviation,
    rolling minimum, and rolling maximum for the specified columns and window size.

    Parameters:
    df (DataFrame): Input DataFrame containing the original data.
    lag_features (list): List of column names for which to generate lag features.
    window_size (int): Window size for calculating rolling statistics.

    Returns:
    DataFrame: DataFrame with engineered features and NaN rows dropped.
    """
    # Create a copy of the input DataFrame
    df_eng = df.copy()

    # Lag features
    for feature in lag_features:
        # Pass is the column is 'Stage1.Output.Measurement0.U.Actual'
        if feature != "Stage1.Output.Measurement0.U.Actual":
            for lag in range(1, window_size + 1):
                df_eng[f"{feature}_lag{lag}"] = df[feature].shift(lag)

    for feature in lag_features:
        # Pass is the column is 'Stage1.Output.Measurement0.U.Actual'
        if feature != "Stage1.Output.Measurement0.U.Actual":
            df_eng[f"{feature}_rolling_mean"] = (
                df[feature].rolling(window=window_size).mean()
            )
            df_eng[f"{feature}_rolling_std"] = (
                df[feature].rolling(window=window_size).std()
            )
            df_eng[f"{feature}_rolling_min"] = (
                df[feature].rolling(window=window_size).min()
            )
            df_eng[f"{feature}_rolling_max"] = (
                df[feature].rolling(window=window_size).max()
            )

    # Drop all rows with NaN values
    df_eng = df_eng.dropna()

    return df_eng


lag_features = df.columns.tolist()
window_size = 60

df_eng = engineer_features(df, lag_features, window_size)

df_eng.to_pickle("../../data/interim/data_engineered.pkl")
