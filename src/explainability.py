import shap
import pandas as pd
import numpy as np


def explain_model(model, X):

    """
    Generate SHAP explanation
    """

    trained_model = model.named_steps["model"]

    preprocessor = model.named_steps["preprocessor"]


    # Transform data

    X_processed = preprocessor.transform(X)


    # Convert sparse matrix to dense

    if hasattr(X_processed, "toarray"):

        X_processed = X_processed.toarray()



    # Get feature names

    try:

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )


    except Exception:

        feature_names = [
            f"Feature_{i}"
            for i in range(
                X_processed.shape[1]
            )
        ]



    # Create dataframe

    X_processed = pd.DataFrame(
        X_processed,
        columns=feature_names
    )


    # Reduce rows for SHAP

    if len(X_processed) > 200:

        X_processed = X_processed.sample(
            200,
            random_state=42
        )



    model_name = trained_model.__class__.__name__



    tree_models = [
        "RandomForestClassifier",
        "RandomForestRegressor",
        "XGBClassifier",
        "XGBRegressor",
        "LGBMClassifier",
        "LGBMRegressor"
    ]



    if model_name in tree_models:

        explainer = shap.TreeExplainer(
            trained_model
        )

    else:

        explainer = shap.Explainer(
            trained_model,
            X_processed
        )



    shap_values = explainer(
        X_processed
    )


    return shap_values, X_processed