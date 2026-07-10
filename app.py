import streamlit as st

from src.data_processing import (
    load_dataset,
    get_dataset_summary,
    get_column_information,
    get_statistics
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