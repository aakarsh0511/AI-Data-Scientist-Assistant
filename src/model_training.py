from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    r2_score
)


from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso
)


from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor
)


from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor
)


from sklearn.svm import (
    SVC,
    SVR
)


from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor
)


from sklearn.naive_bayes import GaussianNB


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


            "Decision Tree":
            DecisionTreeClassifier(),


            "Random Forest":
            RandomForestClassifier(),


            "Gradient Boosting":
            GradientBoostingClassifier(),


            "XGBoost":
            XGBClassifier(
                eval_metric="logloss"
            ),


            "SVM":
            SVC(),


            "KNN":
            KNeighborsClassifier(),


            "Naive Bayes":
            GaussianNB()

        }



    else:


        models = {


            "Linear Regression":
            LinearRegression(),


            "Ridge Regression":
            Ridge(),


            "Lasso Regression":
            Lasso(),


            "Decision Tree":
            DecisionTreeRegressor(),


            "Random Forest":
            RandomForestRegressor(),


            "Gradient Boosting":
            GradientBoostingRegressor(),


            "XGBoost":
            XGBRegressor(),


            "SVR":
            SVR(),


            "KNN":
            KNeighborsRegressor()

        }



    best_model = None


    best_score = float("-inf")



    for name, model in models.items():


        try:


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



            results[name] = round(
                score,
                4
            )



            if score > best_score:

                best_score = score

                best_model = pipeline



        except Exception as e:

            results[name] = -1

    results = dict(
    sorted(
        results.items(),
        key=lambda x: x[1],
        reverse=True
    )
)


    return results, best_model

   