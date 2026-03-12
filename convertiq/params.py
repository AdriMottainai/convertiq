import os
import numpy as np

##################  VARIABLES  ##################
#DATA_SIZE = "1k" # ["1k", "200k", "all"]
#CHUNK_SIZE = 200
#GCP_PROJECT = "<your project id>" # TO COMPLETE
#GCP_PROJECT_WAGON = "wagon-public-datasets"
#BQ_DATASET = "taxifare"
#BQ_REGION = "EU"
MODEL_TARGET = "local"
##################  CONSTANTS  #####################
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DATA_PATH = str(PROJECT_ROOT / "data")
LOCAL_REGISTRY_PATH = str(PROJECT_ROOT / "training_outputs")
COLUMN_NAMES_RAW = [
    "user_id",
    "event_type",
    "event_time",
    "product_id",
    "category_id",
    "brand",
    "price",
    "category_code",
    "user_session",
]
DTYPES_RAW = {
    "user_id": "int32",
    "event_type": "category",
    "event_time": "object",
    "product_id": "int32",
    "category_id": "int64",
    "brand": "object",
    "price": "float32",
    "category_code": "category",
    "user_session": "object",
}

DTYPES_PROCESSED = np.float32