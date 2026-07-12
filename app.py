import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from src.explainability import explain_model
from src.llm import generate_business_insights
from src.report_generator import create_pdf_report
from src.feature_engineering import (feature_engineering_pipeline)
from src.feature_selection import (select_features_using_shap)
from src.model_optimization import (retrain_optimized_model)
from src.model_comparison import compare_models
from src.model_evaluation import ( evaluate_classification_model, evaluate_regression_model)

if "best_model" not in st.session_state:
    st.session_state.best_model = None
if "X_data" not in st.session_state:
    st.session_state.X_data = None
if "problem_type" not in st.session_state:
    st.session_state.problem_type = None
if "shap_values" not in st.session_state:
    st.session_state.shap_values = None
if "X_processed" not in st.session_state:
    st.session_state.X_processed = None
if "model_results" not in st.session_state:
    st.session_state.model_results = None
if "shap_summary" not in st.session_state:
    st.session_state.shap_summary = None
if "ai_report" not in st.session_state:
    st.session_state.ai_report = None
if "df" not in st.session_state:
    st.session_state.df = None
if "selected_features" not in st.session_state:
    st.session_state.selected_features = None
if "feature_importance" not in st.session_state:
    st.session_state.feature_importance = None
if "optimized_model" not in st.session_state:
    st.session_state.optimized_model = None
if "optimized_score" not in st.session_state:
    st.session_state.optimized_score = None
if "y_data" not in st.session_state:
    st.session_state.y_data = None
if "optimized_model_results" not in st.session_state:
    st.session_state.optimized_model_results = None
if "optimized_features" not in st.session_state:
    st.session_state.optimized_features = None
if "optimized_model_results" not in st.session_state:
    st.session_state.optimized_model_results = None
if "X_test" not in st.session_state:
    st.session_state.X_test = None
if "y_test" not in st.session_state:
    st.session_state.y_test = None
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
from src.problem_detection import detect_problem_type
from src.preprocessing import create_preprocessor
from src.model_training import train_models
# Page Configuration
st.set_page_config(
    page_title="AI Data Scientist",
    page_icon="🤖",
    layout="wide"
)
st.title(
    "🤖 AI Data Scientist Assistant"
)
st.write(
    """
Upload any dataset and automatically perform:

- Data Analysis
- ML Model Training
- Model Comparison
- Explainable AI
- AI Generated Business Insights
"""
)
st.divider()
uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)
df=None
if uploaded_file is not None:
    # Load Dataset
    df = load_dataset(uploaded_file)
    st.session_state.df = df
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

df = st.session_state.df

if df is not None:
    st.header("📈 Exploratory Data Analysis")


    if df is not None:

        st.subheader("Automated Insights")

        insights = generate_basic_insights(df)

        for insight in insights:
            st.write("• " + insight)


    else:

        st.info("Please upload a dataset first.")



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

else:

    st.info("Upload dataset first")



st.header("🤖 Machine Learning")

if df is not None:

    target_column = st.selectbox(
        "Select Target Column",
        df.columns
    )
    st.write(df.columns)
else:
    st.warning("Please upload a dataset first")



if st.button("Train Models"):


# Remove rows where target is missing

    df_ml = df.dropna(
        subset=[target_column]
    )


    X = df_ml.drop(
        target_column,
        axis=1
    )


    y = df_ml[target_column]


    # Feature Engineering

    X, feature_report = feature_engineering_pipeline(
        X
    )


    st.session_state.feature_report = feature_report


    st.subheader(
        "⚙️ Feature Engineering Report"
    )

    st.write(
        feature_report
    )


    problem_type = detect_problem_type(
        df,
        target_column
    )


    st.write(
        "Problem Type:",
        problem_type
    )



    preprocessor = create_preprocessor(
        X
    )



    (
    results,
    best_model,
    X_test,
    y_test
) = train_models(
    X,
    y,
    preprocessor,
    problem_type
)


    st.session_state.best_model = best_model

    st.session_state.X_data = X
    st.session_state.y_data = y
    st.session_state.X_test = X_test

    st.session_state.y_test = y_test
    st.session_state.problem_type = problem_type

    st.session_state.model_results = results

    st.subheader(
        "Model Performance"
    )


    result_df = pd.DataFrame(
        results.items(),
        columns=[
            "Model",
            "Score"
        ]
    )


    st.dataframe(
        result_df
    )


    st.success(
        "Best Model Selected Automatically"
    )
    st.info(
    "Model trained using automatically engineered features."
)
    
st.header(
    "📊 Detailed Model Evaluation"
)


if st.button(
    "Generate Evaluation Report"
):


    if st.session_state.best_model is not None:


        if st.session_state.problem_type == "Classification":


            evaluation = evaluate_classification_model(

                st.session_state.best_model,

                st.session_state.X_test,

                st.session_state.y_test

            )


        else:


            evaluation = evaluate_regression_model(

                st.session_state.best_model,

                st.session_state.X_test,

                st.session_state.y_test

            )


        evaluation_df = pd.DataFrame(

            evaluation.items(),

            columns=[
                "Metric",
                "Value"
            ]

        )


        st.dataframe(
            evaluation_df,
            use_container_width=True
        )


    else:

        st.warning(
            "Train model first."
        )



    st.header(
    "🔍 Model Explainability"
)


if st.button("Generate SHAP Explanation"):


    if st.session_state.best_model is not None:

        shap_values, X_processed = explain_model(
            st.session_state.best_model,
            st.session_state.X_data
        )
        st.session_state.shap_values = shap_values
        st.session_state.X_processed = X_processed

        selected_features, feature_importance = (
            select_features_using_shap(
                shap_values,
                X_processed,
                st.session_state.X_data
            )
        )
        st.session_state.selected_features = (
            selected_features)
        st.session_state.feature_importance = feature_importance
        st.success(
            "SHAP values generated successfully")
        st.subheader(
            "Feature Importance")
        fig, ax = plt.subplots()
        shap.summary_plot(
            shap_values,
            X_processed,
            show=False
        )


        st.pyplot(
            fig,
            clear_figure=True
        )


    else:

        st.warning(
            "Please train the model first."
        )






shap_importance = None

if (
    st.session_state.X_processed is not None
    and st.session_state.shap_values is not None
):

    shap_importance = pd.DataFrame(
        {
            "Feature": st.session_state.X_processed.columns,
            "Importance": abs(
                st.session_state.shap_values.values
            ).mean(axis=0)
        }
    )


    shap_importance = shap_importance.sort_values(
        by="Importance",
        ascending=False
    ).head(10)


    st.dataframe(
        shap_importance
    )


else:

    st.warning(
        "Please generate SHAP explanation first."
    )


if shap_importance is not None:
    st.session_state.shap_summary = shap_importance.to_string()
else:
    st.session_state.shap_summary = "SHAP importance not generated yet."

if st.session_state.selected_features:

    st.subheader(
        "⭐ SHAP Selected Important Features"
    )


    selected_df = pd.DataFrame(
        {
            "Selected Features":
            st.session_state.selected_features
        }
    )


    st.dataframe(
        selected_df,
        use_container_width=True
    )
st.header(
    "⚡ Model Optimization"
)


if st.button(
    "Retrain Using Important Features"
):

    if st.session_state.selected_features is not None:


        optimized_model, optimized_score, X_selected = (
            retrain_optimized_model(

                st.session_state.X_data,

                st.session_state.y_data,

                st.session_state.selected_features,

                st.session_state.best_model,

                st.session_state.problem_type
            )
        )


        # Store optimized results

        st.session_state.optimized_model = optimized_model

        st.session_state.optimized_score = optimized_score

        st.session_state.optimized_model_results = optimized_score

        st.session_state.optimized_features = (
            st.session_state.selected_features
        )


        st.success(
            "Optimized Model Retrained Successfully"
        )


        st.metric(
            "Optimized Model Score",
            optimized_score
        )


    else:

        st.warning(
            "Generate SHAP feature selection first."
        )
st.header(
    "📊 Feature Optimization Results"
)


if (
    st.session_state.optimized_model_results is not None
    and
    st.session_state.optimized_features is not None
):


    comparison = compare_models(

        original_score=max(
            st.session_state.model_results.values()
        ),

        optimized_score=
        st.session_state.optimized_model_results,


        original_features=
        len(st.session_state.X_data.columns),


        selected_features=
        len(st.session_state.optimized_features)

    )


    st.dataframe(
        comparison,
        use_container_width=True
    )


else:

    st.info(
        "Retrain optimized model to see comparison."
    )
st.header(
    "🤖 AI Business Analyst Report"
)


if st.button(
    "Generate AI Insights"
):

    if (
        st.session_state.model_results is not None
        and st.session_state.shap_summary is not None
    ):

        report = generate_business_insights(

            dataset_summary=str(
                st.session_state.X_data.describe()
            ),

            model_results=str(
                st.session_state.model_results
            ),

            shap_summary=
            st.session_state.shap_summary

        )


        # Store Gemini output in session state
        st.session_state.ai_report = report


        st.success(
            "AI Business Report Generated Successfully"
        )


        st.divider()


        # Display report

        st.markdown(
            st.session_state.ai_report
        )


    else:

        st.warning(
            "Please train model and generate SHAP explanation first."
        )

st.header(
    "📄 Generate PDF Report"
)


if st.button(
    "Create Report"
):

    if (
        st.session_state.shap_summary
        and
        st.session_state.model_results
    ):


        pdf_file = create_pdf_report(

            dataset_summary=str(
                st.session_state.X_data.describe()
            ),


            model_results=str(
                st.session_state.model_results
            ),


            shap_summary=
            st.session_state.shap_summary,

            ai_report=st.session_state.ai_report
        )


        with open(
            pdf_file,
            "rb"
        ) as file:


            st.download_button(

                label="Download AI Report PDF",

                data=file,

                file_name=
                "AI_Data_Scientist_Report.pdf",

                mime=
                "application/pdf"

            )


    else:

        pass

