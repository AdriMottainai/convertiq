from sklearn.model_selection import train_test_split
import lightgbm as lgb
import pandas as pd


"""def train_test_split(X, y, test_size=0.2, random_state=42) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test"""

def initialize_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # base_spw = base_spw(y_train)

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=200,
        scale_pos_weight=40,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
        num_leaves=40,
        max_depth=5,
        learning_rate=0.04
    )

    print("✅ Model initialized")

    return model

def train_model(model, X, y):
    model.fit(X, y)

    print("✅ Model trained")

    return model

def base_spw(y_train):
    pos = sum(y_train)
    neg = len(y_train-pos)
    base_spw = neg / pos

    return base_spw
