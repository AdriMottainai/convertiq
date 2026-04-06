import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

from convertiq_py.ml_logic.data import load_data_kaggle_raw, clean_data
from convertiq_py.ml_logic.preprocessor import preprocess_features
from convertiq_py.ml_logic.model import initialize_model, train_model
from convertiq_py.ml_logic.registry import save_model



def train() -> dict:
    # Chargement et nettoyage des données brutes
    print(":inbox_tray: Chargement des données")
    df = load_data_kaggle_raw()
    df = clean_data(df)

    # Preprocessing : feature engineering + split temporel observation/prediction
    print("Preprocessing")
    X, y = preprocess_features(df)

    # Split train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialisation et entraînement du modèle
    print("Entraînement du modèle LightGBM")
    model = initialize_model(X_train, y_train)
    model = train_model(model, X_train, y_train)

    # Évaluation
    print(":bar_chart: Évaluation sur le jeu de test")
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc":         round(roc_auc_score(y_test, y_proba)),
        "average_precision": round(average_precision_score(y_test, y_proba)),
    }
    print(f"ROC-AUC: {metrics['roc_auc']}")
    print(f"Average Precision: {metrics['average_precision']}")
    print(classification_report(y_test, y_pred))

    # Sauvegarde du modèle
    model_path = save_model(model)

    return model



if __name__ == "__main__":
    train()