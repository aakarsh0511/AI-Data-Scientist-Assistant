
import pandas as pd
import numpy as np


def create_ratio_features(df):
    """
    Create useful numerical ratio features automatically.
    """

    df = df.copy()

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns


    # Create ratio features between meaningful numerical columns
    if len(numeric_columns) >= 2:

        for i in range(len(numeric_columns)):

            for j in range(i + 1, len(numeric_columns)):

                col1 = numeric_columns[i]
                col2 = numeric_columns[j]


                # Avoid division by zero
                new_feature = f"{col1}_to_{col2}_ratio"


                df[new_feature] = (
                    df[col1] /
                    (df[col2] + 1e-6)
                )


    return df



def remove_high_correlation_features(
        df,
        threshold=0.98
):
    """
    Remove highly correlated numerical features.
    """

    df = df.copy()


    numeric_df = df.select_dtypes(
        include=np.number
    )


    correlation_matrix = (
        numeric_df.corr()
    )


    upper_triangle = (
        correlation_matrix
        .where(
            np.triu(
                np.ones(
                    correlation_matrix.shape
                ),
                k=1
            ).astype(bool)
        )
    )


    drop_columns = [

        column

        for column in upper_triangle.columns

        if any(
            upper_triangle[column] > threshold
        )

    ]


    df = df.drop(
        columns=drop_columns,
        errors="ignore"
    )


    return df, drop_columns



def feature_engineering_pipeline(df):
    """
    Complete automatic feature engineering pipeline.
    """

    original_features = df.shape[1]


    # Step 1: Create new features

    df = create_ratio_features(
        df
    )


    after_creation = df.shape[1]


    # Step 2: Remove correlated features

    df, removed_features = (
        remove_high_correlation_features(
            df
        )
    )


    final_features = df.shape[1]


    feature_report = {

        "Original Features":
        original_features,

        "After Feature Creation":
        after_creation,

        "Removed Correlated Features":
        len(removed_features),

        "Final Features":
        final_features,

        "Removed Columns":
        removed_features

    }


    return df, feature_report