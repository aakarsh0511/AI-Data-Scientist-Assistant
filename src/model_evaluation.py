
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np



def evaluate_classification_model(
        model,
        X_test,
        y_test
):

    """
    Complete classification evaluation
    """


    predictions = model.predict(
        X_test
    )


    results = {}


    results["Accuracy"] = round(
        accuracy_score(
            y_test,
            predictions
        ),
        4
    )


    results["Precision"] = round(
        precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),
        4
    )


    results["Recall"] = round(
        recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),
        4
    )


    results["F1 Score"] = round(
        f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),
        4
    )


    # ROC-AUC

    try:

        probabilities = model.predict_proba(
            X_test
        )[:,1]


        results["ROC-AUC"] = round(
            roc_auc_score(
                y_test,
                probabilities
            ),
            4
        )

    except:

        results["ROC-AUC"] = "Not Available"



    results["Confusion Matrix"] = (
        confusion_matrix(
            y_test,
            predictions
        )
    )


    results["Classification Report"] = (
        classification_report(
            y_test,
            predictions
        )
    )


    return results





def evaluate_regression_model(
        model,
        X_test,
        y_test
):

    """
    Complete regression evaluation
    """


    predictions = model.predict(
        X_test
    )


    results = {}


    results["R2 Score"] = round(
        r2_score(
            y_test,
            predictions
        ),
        4
    )


    results["MAE"] = round(
        mean_absolute_error(
            y_test,
            predictions
        ),
        4
    )


    results["MSE"] = round(
        mean_squared_error(
            y_test,
            predictions
        ),
        4
    )


    results["RMSE"] = round(
        np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        ),
        4
    )


    return results