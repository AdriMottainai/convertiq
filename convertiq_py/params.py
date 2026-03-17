import os
import numpy as np
from pathlib import Path
##################  VARIABLES  ##################
DATA_SIZE = "1M"  # ["1k", "200k", "1M", "all"]

DATA_SIZE_MAP = {
    "200k": 200_000,
    "1M": 1_000_000,
    "10M": 10_000_000,
    "all": None
}

# Time interval for preprocessing
OBSERVATION_END = "2019-10-06"
PREDICTION_END = "2019-10-08"

CHUNK_SIZE = 100_000 #nbr de row qu'on va utiliser [pas encore operationnel en code]
GCP_PROJECT = "convertiq-490009"

MODEL_TARGET = "local"
##################  CONSTANTS  #####################
PROJECT_ROOT = Path(__file__).resolve().parents[0]
LOCAL_DATA_PATH = os.path.join(PROJECT_ROOT,  'convertiq', 'raw_data')
LOCAL_REGISTRY_PATH = os.path.join(PROJECT_ROOT, 'save_models')
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
