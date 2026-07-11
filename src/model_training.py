
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score
)


from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression
)


from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)


from xgboost import (
    XGBClassifier,
    XGBRegressor
)



def train_models(
        X,
        y,
        preprocessor,
        problem_type
):


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )



    results = {}



    if problem_type == "Classification":


        models = {


            "Logistic Regression":
            LogisticRegression(
                max_iter=1000
            ),


            "Random Forest":
            RandomForestClassifier(),


            "XGBoost":
            XGBClassifier(
                eval_metric="logloss"
            )

        }



    else:


        models = {


            "Linear Regression":
            LinearRegression(),


            "Random Forest":
            RandomForestRegressor(),


            "XGBoost":
            XGBRegressor()

        }



    best_model = None

    best_score = 0



    for name, model in models.items():


        pipeline = Pipeline(

            steps=[

                (
                    "preprocessor",
                    preprocessor
                ),

                (
                    "model",
                    model
                )

            ]

        )


        pipeline.fit(
            X_train,
            y_train
        )



        predictions = pipeline.predict(
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



        results[name] = score



        if score > best_score:

            best_score = score

            best_model = pipeline



    return results, best_model