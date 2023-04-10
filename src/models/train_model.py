import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score


df = pd.read_pickle("../../data/processed/best_features.pkl")
df_basic = pd.read_pickle("../../data/interim/data_processed.pkl")


def experiment_models(df, target_column, test_size=0.2, scale=True):
    # Split data into Train and Test sets
    split_index = int(len(df) * (1 - test_size))
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]

    X_train = train.drop(target_column, axis=1)
    y_train = train[target_column]
    X_test = test.drop(target_column, axis=1)
    y_test = test[target_column]

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
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

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
