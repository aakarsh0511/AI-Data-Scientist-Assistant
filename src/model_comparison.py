
import pandas as pd


def compare_models(
        original_score,
        optimized_score,
        original_features,
        selected_features
):

    comparison = pd.DataFrame({

        "Metric":[
            "Number of Features",
            "Model Score"
        ],

        "Before SHAP":[
            original_features,
            original_score
        ],

        "After SHAP":[
            selected_features,
            optimized_score
        ]

    })


    return comparison