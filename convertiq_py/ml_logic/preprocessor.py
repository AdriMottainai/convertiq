import pandas as pd

# from google.cloud import bigquery
# from colorama import Fore, Style
# from pathlib import Path

from convertiq_py.params import *
from convertiq_py.ml_logic.data import clean_data

def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    # Clean data
    X = clean_data(df)

    # Observation and prediction set split
    observation_end = pd.Timestamp(OBSERVATION_END)
    prediction_end  = pd.Timestamp(PREDICTION_END)

    # X_pred sert juste à aller récupérer les acheteurs (y) pendant cette période
    X_obs = X[X["event_time"] < observation_end].copy()
    X_pred = X[X["event_time"] >= observation_end].copy()

    # Create y with purchasers from prediction period
    purchasers = set(X_pred.loc[X_pred["event_type"] == "purchase", "user_id"])
    y_purchasers = pd.DataFrame({"user_id": X_obs["user_id"].unique()})
    y_purchasers["label"] = y_purchasers["user_id"].isin(purchasers).astype(int)

    # Feature engineering X_obs
    X_feat_eng = feature_engineering(X_obs)

    # Grouping with y_purchasers
    dataset = y_purchasers.merge(X_feat_eng, on="user_id", how="inner")
    X_processed = dataset.drop(columns="label")
    X_processed = X_processed.drop(columns="user_id")

    y_processed = dataset["label"]

    print("✅ X_processed, with shape", X_processed.shape)
    print("✅ y_processed, with shape", y_processed.shape)

    return X_processed, y_processed


def feature_engineering(X_obs: pd.DataFrame) -> pd.DataFrame:
    # Time features
    X_obs["hour"] = X_obs["event_time"].dt.hour
    X_obs["dayofweek"] = X_obs["event_time"].dt.dayofweek  # 0=Lundi
    X_obs["is_weekend"] = (X_obs["dayofweek"] >= 5).astype("int8")

    # Features de comportement dans l'intervalle d'observation choisi
    # Prend beaucoup de temps les fonctions lambda, optimisable?
    behavior = X_obs.groupby("user_id").agg(
    total_events   = ("event_type", "count"),
    total_views    = ("event_type", lambda x: (x == "view").sum()),
    total_carts    = ("event_type", lambda x: (x == "cart").sum()),
    total_purchases= ("event_type", lambda x: (x == "purchase").sum()),
    n_sessions     = ("user_session", "nunique"),
    n_days_active  = ("event_time", lambda x: x.dt.date.nunique()),
    )
    print(behavior["total_carts"].dtype)
    behavior["has_ever_carted"] = (behavior["total_carts"].astype("int8") > 0).astype("int8")
    behavior["has_ever_purchased"] = (behavior["total_purchases"].astype("int8") > 0).astype("int8")
    behavior["view_to_cart_ratio"] = (
        behavior["total_carts"].astype("int8") / behavior["total_views"].astype("int8").replace(0, 1)
    )
    behavior["cart_to_purchase_ratio"] = (
        behavior["total_purchases"].astype("int8") / behavior["total_carts"].astype("int8").replace(0, 1)
    )

    # Sessions features
    session_stats = X_obs.groupby(["user_id", "user_session"], as_index=False).agg(
        session_start = ("event_time", "min"),
        session_end   = ("event_time", "max"),
        session_events = ("event_time", "count"),
    )
    session_stats["session_duration"] = (
        session_stats["session_end"] - session_stats["session_start"]
    )
    # Grouping Session features per user_id for the observation period
    ## Infos de session par user_ID sur l'intervalle d'observation

    session_user = session_stats.groupby("user_id").agg(
        avg_session_duration    = ("session_duration", "mean"),
        median_session_duration = ("session_duration", "median"),
        max_session_duration    = ("session_duration", "max"),
        avg_events_per_session  = ("session_events", "mean"),
        max_events_per_session  = ("session_events", "max"),
    )

    for col in ["avg_session_duration", "median_session_duration", "max_session_duration"]:
            session_user[col] = session_user[col].dt.total_seconds()

    # Features grouping and X_features_engineering
    X_feat_eng = (behavior.join(session_user, how="left")
        )
    X_feat_eng  = X_feat_eng #.astype("float32")

    return X_feat_eng
