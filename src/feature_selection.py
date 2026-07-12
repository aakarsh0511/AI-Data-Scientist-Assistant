import pandas as pd
import numpy as np


def select_features_using_shap(
        shap_values,
        X_processed,
        original_X,
        top_features=20
):


    # Calculate SHAP importance

    importance = (
        np.abs(
            shap_values.values
        )
        .mean(axis=0)
    )


    feature_importance = pd.DataFrame(
        {
            "Feature": X_processed.columns,
            "Importance": importance
        }
    )


    feature_importance = (
        feature_importance
        .sort_values(
            by="Importance",
            ascending=False
        )
    )


    # Select top N features

    important_features = (
        feature_importance
        .head(top_features)
        ["Feature"]
        .tolist()
    )


    selected_original_features = []


    for feature in important_features:


        # remove transformer prefix

        if "__" in feature:

            original_feature = (
                feature.split("__")[-1]
            )

        else:

            original_feature = feature



        if original_feature in original_X.columns:

            selected_original_features.append(
                original_feature
            )


    selected_original_features = list(
        set(selected_original_features)
    )


    return (
        selected_original_features,
        feature_importance
    )