import streamlit as st

from src.data_processing import (
    load_dataset,
    get_dataset_summary,
    get_column_information,
    get_statistics
)
from src.eda import (
    missing_value_analysis,
    numerical_distribution,
    correlation_heatmap,
    categorical_analysis,
    generate_basic_insights
)


# Page Configuration

st.set_page_config(
    page_title="AI Data Scientist Assistant",
    page_icon="🤖",
    layout="wide"
)



st.title(
    "🤖 AI Data Scientist Assistant"
)


st.write(
    """
    Upload your dataset and automatically analyze it.
    """
)



uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)



if uploaded_file is not None:


    # Load Dataset

    df = load_dataset(uploaded_file)


    st.success(
        "Dataset loaded successfully!"
    )


    # Dataset Summary

    st.header(
        "📊 Dataset Overview"
    )


    summary = get_dataset_summary(df)


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Rows",
            summary["Number of Rows"]
        )


    with col2:

        st.metric(
            "Columns",
            summary["Number of Columns"]
        )


    with col3:

        st.metric(
            "Missing Values",
            summary["Total Missing Values"]
        )


    with col4:

        st.metric(
            "Duplicates",
            summary["Duplicate Rows"]
        )



    # Dataset Preview


    st.subheader(
        "First 5 Rows"
    )


    st.dataframe(
        df.head()
    )



    # Column Information


    st.subheader(
        "Column Information"
    )


    column_info = get_column_information(df)


    st.dataframe(
        column_info,
        use_container_width=True
    )



    # Statistics


    st.subheader(
        "Statistical Summary"
    )
    st.dataframe(
        get_statistics(df),
        use_container_width=True
    )
else:
    st.info(
        "Upload a dataset to start analysis."
    )






# -----------------------------
# Exploratory Data Analysis
# -----------------------------


st.header(
    "📈 Exploratory Data Analysis"
)



# Insights

st.subheader(
    "Automated Insights"
)


insights = generate_basic_insights(df)


for insight in insights:

    st.write(
        "• " + insight
    )



# Missing Values

st.subheader(
    "Missing Value Analysis"
)


missing_fig = missing_value_analysis(df)


if missing_fig:

    st.plotly_chart(
        missing_fig,
        use_container_width=True
    )

else:

    st.success(
        "No missing values detected"
    )



# Numerical Distribution

st.subheader(
    "Numerical Feature Distribution"
)


distribution_figures = numerical_distribution(df)


for fig in distribution_figures:

    st.plotly_chart(
        fig,
        use_container_width=True
    )



# Correlation

st.subheader(
    "Feature Correlation"
)


corr_fig = correlation_heatmap(df)


if corr_fig:

    st.plotly_chart(
        corr_fig,
        use_container_width=True
    )



# Categorical Analysis

st.subheader(
    "Categorical Feature Analysis"
)


cat_figures = categorical_analysis(df)


for fig in cat_figures:

    st.plotly_chart(
        fig,
        use_container_width=True
    )