import pandas as pd

from pathlib import Path

import kagglehub
from kagglehub import KaggleDatasetAdapter

from convertiq.params import LOCAL_DATA_PATH, COLUMN_NAMES_RAW, DTYPES_RAW, DATA_SIZE, DATA_SIZE_MAP, CHUNK_SIZE

def load_data_kaggle_raw(nrows:int | None = None) -> pd.DataFrame:
    if nrows is None:
        nrows = DATA_SIZE_MAP[DATA_SIZE]
        
    file_path = "2019-Oct.csv"
    
    df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS,"mkechinov/ecommerce-behavior-data-from-multi-category-store",
    "2019-Oct.csv",pandas_kwargs={"usecols": COLUMN_NAMES_RAW,
            "nrows": nrows,
            "dtype": DTYPES_RAW,
}
)
    return df

def load_data_in_chunks(csv_path: str | Path):
    
    return pd.read_csv(
        csv_path,
        usecols=COLUMN_NAMES_RAW,
        dtype=DTYPES_RAW,
        chunksize=CHUNK_SIZE,
    )

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
