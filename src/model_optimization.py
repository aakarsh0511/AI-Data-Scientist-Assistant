from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from src.preprocessing import create_preprocessor


def retrain_optimized_model(
        X,
        y,
        selected_features,
        best_model,
        problem_type
):

    """
    Retrain model using SHAP selected features
    """


    # Select important features

    X_selected = X[selected_features]


    X_train, X_test, y_train, y_test = train_test_split(
        X_selected,
        y,
        test_size=0.2,
        random_state=42
    )


    # Create NEW preprocessor
    # because number of features changed

    preprocessor = create_preprocessor(
        X_selected
    )


    optimized_pipeline = Pipeline(
        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                best_model.named_steps["model"]
            )

        ]
    )


    optimized_pipeline.fit(
        X_train,
        y_train
    )


    predictions = optimized_pipeline.predict(
        X_test
    )


    if problem_type == "Classification":

        score = accuracy_score(
            y_test,
            predictions
        )

    else:

        score = r2_score(
            y_test,
            predictions
        )


    return (
        optimized_pipeline,
        score,
        X_selected
    )