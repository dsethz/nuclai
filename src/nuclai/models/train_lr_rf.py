# Author: Daniel Schirmacher                                                                                           #
# Date: 10.02.2021                                                                                                     #
# Python: 3.8.6                                                                                                        #
########################################################################################################################
import argparse
import json
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid


def args_parse():
    """
    Catches user input.


    Parameters
    ----------

    -


    Return
    ------

    Returns a namespace from `argparse.parse_args()`.
    """
    desc = "Program to plot traces and image crops."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--data", type=str, required=True, help="Path to training data JSON."
    )

    parser.add_argument(
        "--output_base_dir",
        type=str,
        required=True,
        help="Path to output directory.",
    )

    return parser.parse_args()


def main():
    args = args_parse()
    path_data = args.data
    path_out = args.output_base_dir

    # create output directory
    os.makedirs(path_out, exist_ok=True)

    # Read data
    with open(path_data) as f:
        data = json.load(f)

    # compile training data
    tmp = np.load(data["0"]["path"])
    data_train = np.zeros((len(data), tmp.shape[1]))
    labels_train = np.zeros(len(data))

    for i, k in enumerate(data.keys()):
        tmp = np.load(data[k]["path"])
        data_train[i] = tmp
        labels_train[i] = data[k]["label"]

    # Define hyperparameter grid for Logistic Regression
    param_grid_lr = {
        "penalty": ["l1", "l2"],
        "C": [0.1, 1, 10, 100],
        "class_weight": [None, "balanced"],
        "solver": ["saga"],
    }

    # Define hyperparameter grid for Random Forest
    param_grid_rf = {
        "n_estimators": [20, 100, 500],
        "max_depth": [None, 10, 20],
        "class_weight": [None, "balanced"],
    }

    # Train and save Logistic Regression models
    print("Training Logistic Regression models...")
    grid_lr = ParameterGrid(param_grid_lr)
    for params in grid_lr:
        try:
            lr = LogisticRegression(max_iter=10000, **params)
            lr.fit(data_train, labels_train)
            # Save the model
            penalty = params["penalty"]
            C = params["C"]
            class_weight = (
                "imbalanced" if params["class_weight"] is None else "balanced"
            )
            model_filename = f"lr_{penalty}_{C}_{class_weight}.joblib"
            joblib.dump(lr, os.path.join(path_out, model_filename))
            print(f"Saved Logistic Regression model with parameters: {params}")
        except Exception as e:  # noqa BLE001
            print(
                f"Failed to train Logistic Regression model with parameters: {params}"
            )
            print(f"Error: {e}")

    # Train and save Random Forest models
    print("\nTraining Random Forest models...")
    grid_rf = ParameterGrid(param_grid_rf)
    for params in grid_rf:
        try:
            rf = RandomForestClassifier(**params)
            rf.fit(data_train, labels_train)
            # Save the model
            n_estimators = params["n_estimators"]
            max_depth = (
                "none" if params["max_depth"] is None else params["max_depth"]
            )
            class_weight = (
                "imbalanced" if params["class_weight"] is None else "balanced"
            )
            model_filename = (
                f"rf_{n_estimators}_{max_depth}_{class_weight}.joblib"
            )
            joblib.dump(rf, os.path.join(path_out, model_filename))
            print(f"Saved Random Forest model with parameters: {params}")
        except Exception as e:  # noqa BLE001
            print(
                f"Failed to train Random Forest model with parameters: {params}"
            )
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
