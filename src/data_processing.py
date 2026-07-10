
import pandas as pd

def load_dataset(uploaded_file):
    """
    Load CSV or Excel file
    """
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError(
            "Unsupported file format"
        )
    return df

def get_dataset_summary(df):
    """
    Generate basic dataset information
    """
    summary = {
        "Number of Rows": df.shape[0],
        "Number of Columns": df.shape[1],
        "Total Missing Values": df.isnull().sum().sum(),
        "Duplicate Rows": df.duplicated().sum()
    }
    return summary

def get_column_information(df):
    """
    Generate column-level information
    """
    column_info = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum().values,
        "Missing Percentage":
            (
                df.isnull().sum()
                /
                len(df)
                *
                100
            ).round(2).values,
        "Unique Values":
            [
                df[col].nunique()
                for col in df.columns
            ]
    })
    return column_info

def get_statistics(df):
    """
    Generate statistical summary
    """
    return df.describe(include="all")