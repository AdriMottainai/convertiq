import glob
import os
import time
import pickle
from convertiq_py.params import *
import joblib
from datetime import datetime



def save_model(model) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = Path(LOCAL_REGISTRY_PATH) / f"model_lgbm_baseline{timestamp}.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")
    return str(model_path)



def load_model():

    model = joblib.load(os.path.join(LOCAL_REGISTRY_PATH, "model_lgbm_baseline.pkl"))

    return model
