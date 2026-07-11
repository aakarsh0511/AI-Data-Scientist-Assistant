
import pandas as pd


def detect_problem_type(df, target_column):

    """
    Detect ML problem type
    """

    target = df[target_column]


    unique_values = target.nunique()


    datatype = target.dtype



    # Classification

    if datatype == "object":

        return "Classification"



    elif unique_values <= 20:

        return "Classification"



    # Regression

    elif datatype in ["int64", "float64"]:

        return "Regression"



    else:

        return "Regression"