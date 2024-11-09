# Author: Daniel Schirmacher                                                                                           #
# Date: 10.02.2021                                                                                                     #
# Python: 3.8.6                                                                                                        #
########################################################################################################################
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


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
        "--data", type=str, required=True, help="Path to testing data JSON."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model JOBLIB file.",
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
    path_model = args.model
    path_out = args.output_base_dir

    # create output directory
    os.makedirs(path_out, exist_ok=True)

    # Read data
    with open(path_data) as f:
        data = json.load(f)

    # compile training data
    tmp = np.load(data[0]["path"])
    data_test = np.zeros((len(data), tmp.shape[1]))
    labels_test = np.zeros(len(data))
    ids = []

    for i, k in enumerate(data.keys()):
        tmp = np.load(data[k]["path"])
        data_test[i] = tmp
        labels_test[i] = data[k]["label"]
        ids.append(os.path.basename(data[k]["path"]))

    # Load model
    model = joblib.load(path_model)

    # predict labels
    labels_pred = model.predict(data_test)

    # calculate F1 score
    f1 = f1_score(labels_test, labels_pred)

    # save results
    results_f1 = pd.DataFrame({"f1": [f1]})
    results_f1.to_csv(os.path.join(path_out, "f1_score.csv"), index=False)

    predictions = pd.DataFrame(
        {"id": ids, "prediction": labels_pred, "label": labels_test}
    )
    predictions.to_csv(os.path.join(path_out, "predictions.csv"), index=False)


if __name__ == "__main__":
    main()
