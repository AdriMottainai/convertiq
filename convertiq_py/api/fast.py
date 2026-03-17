import pandas as pd
from fastapi import FastAPI, UploadFile, File

from convertiq_py.ml_logic.preprocessor import preprocess_features, feature_engineering
from convertiq_py.ml_logic.data import load_data_kaggle_raw, load_data_in_chunks, clean_data
from convertiq_py.ml_logic.registry import load_model


app = FastAPI()
app.state.model = load_model()

@app.post("/predict")
async def predict_csv(csv: UploadFile = File(...)):
    df = pd.read_csv(csv.file)

    """
    Make a prediction from a dataset.
    """

    X = clean_data(df)

    X_processed = X_processed.reset_index()
    user_ids = X_processed['user_id']
    X_processed= X_processed.drop(columns=['user_id'])

    print('Debug X_processed')

    y_pred = app.state.model.predict(X_processed)
    y_proba = app.state.model.predict_proba(X_processed)[1]
    #y_pred.apply(lambda x: 'purchase' if x==1 else 'no purchase')

    print('Debug y_pred')
    print(y_pred)


    return {
    'user_id': int(user_ids[0]),
    'prediction': int(y_pred[0]),
    'probability' : float(y_proba[0])
    }


@app.get("/")
def root():
    root = {
    'Welcome to Convertiq'
    }
    return root
