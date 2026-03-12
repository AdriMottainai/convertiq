import pandas as pd

from google.cloud import bigquery
from colorama import Fore, Style
from pathlib import Path

from convertiq.params import *

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by
    - assigning correct dtypes to each column
    - removing buggy or irrelevant transactions
    """

    # Clean the event time, remove UTC
    df["event_time"] = pd.to_datetime(
        df["event_time"].str.replace(" UTC", "", regex=False),
        format="%Y-%m-%d %H:%M:%S",
    )

    # Impute the "brand" feature = "has_brand" feature
    df["brand"] = df["brand"].astype(str).replace("nan", "unknown").astype("category")
    df["has_brand"] = (df["brand"] != "unknown").astype("int8")


    # Drop the duplicates
    df = df.drop_duplicates()

    # New feature "has_valid_price"
    df["has_valid_price"] = (df["price"] > 0).astype("int8")

    print("✅ Data cleaned")

    return df
