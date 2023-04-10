import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_regression

df = pd.read_pickle("../../data/interim/data_engineered.pkl")


def select_best_features(df, target_column, k=10):
    """
    Select the k best features with regards to the target column using the mutual_info_regression scoring function.

    Parameters:
    df (DataFrame): Input DataFrame containing the original data.
    target_column (str): Name of the target column.
    k (int): Number of best features to select. Default is 10.

    Returns:
    DataFrame: DataFrame containing only the k best features.
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    selector = SelectKBest(mutual_info_regression, k=k)
    selector.fit(X, y)

    selected_features = X.columns[selector.get_support()]

    return df[selected_features]


best_features_df = select_best_features(df, "Stage1.Output.Measurement0.U.Actual", k=50)
best_features_df.columns

best_features_df = pd.concat(
    [best_features_df, df["Stage1.Output.Measurement0.U.Actual"]], axis=1
)

best_features_df.to_pickle("../../data/processed/best_features.pkl")
