import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Import interim data
df = pd.read_pickle("../../data/interim/data_processed.pkl")


# Remove extreme values and fill missing values
def clean_series(
    series: pd.Series, window_size: int = 10, num_std_dev: int = 3
) -> pd.Series:
    """
    Clean a time series by removing extreme values and filling missing values using linear interpolation.

    Args:
        series (pd.Series): Time series data
        window_size (int): Size of the rolling window
        num_std_dev (int): The number of standard deviations to consider for identifying outliers

    Returns:
        pd.Series: Cleaned time series
    """

    # Move copy of the input series
    cleaned_series = series.copy()

    # Calculate the rolling window mean and standard deviation
    moving_avg = cleaned_series.rolling(window=window_size).mean()
    moving_std = cleaned_series.rolling(window=window_size).std()

    # Identify outliers
    outliers = np.abs(cleaned_series - moving_avg) > num_std_dev * moving_std

    # Replace outliers with NaN values
    cleaned_series[outliers] = np.nan

    # Replace 0 values with NaN values
    cleaned_series[cleaned_series == 0] = np.nan

    # Fill missing values using linear interpolation
    cleaned_series = cleaned_series.interpolate(method="linear")

    return cleaned_series


cleaned_series = clean_series(
    df["Stage1.Output.Measurement0.U.Actual"], window_size=100
)

df["Stage1.Output.Measurement0.U.Actual"] = cleaned_series
df = df.iloc[:, :42]


# Engineer new features including lag features and rolling statistics
def engineer_features(
    df: pd.DataFrame, lag_features: list, window_size: int
) -> pd.DataFrame:
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
        # Pass if the column is 'Stage1.Output.Measurement0.U.Actual'
        if feature != "Stage1.Output.Measurement0.U.Actual":
            for lag in range(1, window_size + 1):
                df_eng[f"{feature}_lag{lag}"] = df[feature].shift(lag)

    # Lag features rolling statistics
    for feature in lag_features:
        # Pass if the column is 'Stage1.Output.Measurement0.U.Actual'
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


# Create engineered features
lag_features = df.columns.tolist()
window_size = 60
df_eng = engineer_features(df, lag_features, window_size)

# Save the engineered features to disk
df_eng.to_pickle("../../data/interim/data_engineered.pkl")
