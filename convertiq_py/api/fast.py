import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from convertiq_py.ml_logic.preprocessor import preprocess_features, feature_engineering
from convertiq_py.ml_logic.data import load_data_kaggle_raw, load_data_in_chunks, clean_data
from convertiq_py.ml_logic.registry import load_model

app = FastAPI()
app.state.model = load_model()

# Allowing all middleware is optional, but good practice for dev purposes
'''app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)'''

# http://127.0.0.1:8000/predict?pickup_datetime=2014-07-06+19:18:00&pickup_longitude=-73.950655&pickup_latitude=40.783282&dropoff_longitude=-73.984365&dropoff_latitude=40.769802&passenger_count=2
@app.get("/predict")
def predict(
        df,  # Format df initial
    ):
    """
    Make a prediction from a dataset.
    """

    X = clean_data(df)

    X_processed = feature_engineering(X)

    print('Debug X_processed')

    y_pred = app.state.model.predict(X_processed)
    y_pred.apply(lambda x: 'purchase' if x==1 else 'no purchase')

    print('Debug y_pred')
    print(y_pred)

    return y_pred


@app.get("/")
def root():
    root = {
    'greeting': 'Hello'
    }
    return root
