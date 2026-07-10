
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff



def missing_value_analysis(df):

    """
    Create missing value visualization
    """

    missing = (
        df.isnull()
        .sum()
        .reset_index()
    )

    missing.columns = [
        "Column",
        "Missing Values"
    ]


    missing = missing[
        missing["Missing Values"] > 0
    ]


    if len(missing) == 0:

        return None


    fig = px.bar(
        missing,
        x="Column",
        y="Missing Values",
        title="Missing Value Analysis"
    )


    return fig



def numerical_distribution(df):

    """
    Create distribution plots
    """

    numerical_columns = (
        df.select_dtypes(
            include=["int64","float64"]
        )
        .columns
    )


    figures = []


    for column in numerical_columns:


        fig = px.histogram(
            df,
            x=column,
            title=f"Distribution of {column}",
            marginal="box"
        )


        figures.append(fig)


    return figures



def correlation_heatmap(df):

    """
    Generate correlation heatmap
    """


    numerical_df = (
        df.select_dtypes(
            include=["int64","float64"]
        )
    )


    if numerical_df.shape[1] < 2:

        return None


    correlation = (
        numerical_df
        .corr()
    )


    fig = px.imshow(
        correlation,
        text_auto=True,
        title="Correlation Heatmap"
    )


    return fig



def categorical_analysis(df):

    """
    Analyze categorical columns
    """

    categorical_columns = (
        df.select_dtypes(
            include=["object","category"]
        )
        .columns
    )


    figures = []


    for column in categorical_columns:


        if df[column].nunique() <= 20:


            count_df = (
                df[column]
                .value_counts()
                .reset_index()
            )


            count_df.columns = [
                column,
                "Count"
            ]


            fig = px.bar(
                count_df,
                x=column,
                y="Count",
                title=f"{column} Distribution"
            )


            figures.append(fig)


    return figures



def generate_basic_insights(df):

    """
    Generate automatic observations
    """

    insights = []


    rows, cols = df.shape


    insights.append(
        f"The dataset contains {rows} rows and {cols} columns."
    )


    missing = (
        df.isnull()
        .sum()
        .sum()
    )


    if missing > 0:

        insights.append(
            f"The dataset contains {missing} missing values."
        )

    else:

        insights.append(
            "The dataset does not contain missing values."
        )


    numerical_columns = (
        df.select_dtypes(
            include=["int64","float64"]
        )
        .columns
    )


    insights.append(
        f"There are {len(numerical_columns)} numerical features."
    )


    categorical_columns = (
        df.select_dtypes(
            include=["object","category"]
        )
        .columns
    )


    insights.append(
        f"There are {len(categorical_columns)} categorical features."
    )


    return insights