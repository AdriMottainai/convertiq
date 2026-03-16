import pandas as pd

# from google.cloud import bigquery
# from colorama import Fore, Style
# from pathlib import Path

from convertiq.params import *

def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:

    # Filtering X to keep only top 5 category_code
    top5 = ['electronics.smartphone', 'electronics.audio.headphone', 'electronics.video.tv',
        'electronics.clocks', 'computers.notebook']
    X = df[df['category_code'].isin(top5)]

    # Listed in chronological order
    X = X.sort_values("event_time").reset_index(drop=True)

    # Observation and prediction set split
    observation_end = pd.Timestamp(OBSERVATION_END)
    prediction_end  = pd.Timestamp(PREDICTION_END)

    X_obs = X[X["event_time"] < observation_end].copy()
    X_pred = X[X["event_time"] >= observation_end].copy()

    # Create y with purchasers from prediction period
    purchasers = set(X_obs.loc[X_obs["event_type"] == "purchase", "user_id"])
    y_purchasers = pd.DataFrame({"user_id": X_obs["user_id"].unique()})
    y_purchasers["label"] = y_purchasers["user_id"].isin(purchasers).astype(int)

    # Feature engineering X_obs
    feature_engineering(X_obs)

    # Grouping with y_purchasers
    dataset = y_purchasers.merge(user_features, on="user_id", how="inner")
    X_processed = dataset.drop(columns="label")
    X_processed = X_processed.drop(columns="user_id")

    y_processed = dataset["label"]

    print("✅ X_processed, with shape", X_processed.shape)
    print("✅ y_processed, with shape", y_processed.shape)

    return X_processed, y_processed


def feature_engineering(X: pd.DataFrame) -> pd.DataFrame:
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

    behavior["has_ever_carted"] = (behavior["total_carts"] > 0).astype("int8")
    behavior["has_ever_purchased"] = (behavior["total_purchases"] > 0).astype("int8")
    behavior["view_to_cart_ratio"] = (
        behavior["total_carts"] / behavior["total_views"].replace(0, 1)
    )
    behavior["cart_to_purchase_ratio"] = (
        behavior["total_purchases"] / behavior["total_carts"].replace(0, 1)
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
    X_feat_eng  = X_feat_eng.astype("float32")

    return X_feat_eng
