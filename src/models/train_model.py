import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# Load the data from the pickle file
df = pd.read_pickle("../../data/processed/best_features.pkl")
# df_basic = pd.read_pickle("../../data/interim/data_processed.pkl")


def experiment_models(
    df: pd.DataFrame, target_column: str, test_size: float = 0.2, scale: bool = True
) -> pd.DataFrame:
    """Trains and evaluates regression models and plots the actual vs. predicted values.

    Args:
        df (pd.DataFrame): Dataframe containing the data.
        target_column (str): Name of the target column.
        test_size (float, optional): Ratio of the test data. Defaults to 0.2.
        scale (bool, optional): Whether to scale the data. Defaults to True.

    Returns:
        pd.DataFrame: Dataframe containing the results of the experiments.
    """
    # Split data into Train and Test sets
    split_index = int(len(df) * (1 - test_size))
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]

    # Define features and target
    X_train = train.drop(target_column, axis=1)
    y_train = train[target_column]
    X_test = test.drop(target_column, axis=1)
    y_test = test[target_column]

    # Scale the data
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Define models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42),
        "LightGBM": LGBMRegressor(random_state=42),
    }
    results = []

    for model_name, model in models.items():
        # Fit the model to the training data
        model.fit(X_train, y_train)
        # Predict the target column using the test data
        y_pred = model.predict(X_test)

        # Calculate the MAE and R2 scores
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append({"Model": model_name, "MAE": mae, "R2": r2})

        # Plot the actual vs. predicted values
        plt.plot(y_test.values, label="Actual")
        plt.plot(y_pred, label="Predicted")
        plt.title(f"{model_name} - Actual Vs. Predicted - MAE: {mae:.4f}, R2: {r2:.4f}")
        plt.xlabel("Time")
        plt.ylabel(target_column)
        plt.legend()
        plt.show()

    # Display the results as a DataFrame
    results_df = pd.DataFrame(results).set_index("Model")
    return results


# Run the experiment and visualize the results
target_column = "Stage1.Output.Measurement0.U.Actual"
results = experiment_models(df, target_column)
print(results)
