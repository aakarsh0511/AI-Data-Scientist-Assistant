
import pandas as pd

from src.feature_engineering import (
    feature_engineering_pipeline
)


df = pd.read_csv(
    "Sales_Records_Final.csv"
)


new_df, report = feature_engineering_pipeline(
    df
)


print(new_df.head())

print(report)