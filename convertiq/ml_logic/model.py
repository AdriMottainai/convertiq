from sklearn.model_selection import train_test_split
import lightgbm as lgb


def train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) -> pd.Dataframe:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    return X_train, X_test, y_train, y_test

def initialize_model(
        objective="binary",
        n_estimators=200,
        scale_pos_weight=base_spw,
        random_state=42,
        verbosity=-1,
        n_jobs=-1,
        num_leaves=40,
        max_depth=5,
        learning_rate=0.04):

    model = lgb.LGBMClassifier(
        objective=objective,
        n_estimators=n_estimators,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        verbosity=verbosity,
        n_jobs=n_jobs,
        num_leaves=num_leaves,
        max_depth=max_depth,
        learning_rate=learning_rate
    )

    print("✅ Model initialized")

    return model

def train_model(model, X, y):
    model.fit(X, y)

    print("✅ Model trained")

    return model
