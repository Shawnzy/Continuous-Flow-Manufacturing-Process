import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import raw data
df = pd.read_csv("../../data/raw/continuous_factory_process.csv")


# perform basic data processing and create a dataframe datetime index
def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the input DataFrame by selecting relevant columns, removing 'Setpoint' columns, and setting the 'time_stamp' as index.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be processed

    Returns:
    pd.DataFrame: The processed DataFrame
    """
    # Select the first 72 columns
    selected_columns = df.columns[:72]
    df = df[selected_columns]

    # Remove columns containing 'Setpoint' in their name
    df = df.loc[:, ~df.columns.str.contains("Setpoint")]

    # Convert 'time_stamp' column to datetime and set as index
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    df = df.set_index("time_stamp")

    return df


df_processed = process_dataframe(df)

# Save processed data to disk
df_processed.to_pickle("../../data/interim/data_processed.pkl")
