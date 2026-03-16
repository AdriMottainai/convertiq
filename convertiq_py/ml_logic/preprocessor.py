import pandas as pd

# from google.cloud import bigquery
# from colorama import Fore, Style
# from pathlib import Path

from convertiq_py.params import *

def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:

    # Observation and prediction set split
    observation_end = pd.Timestamp(OBSERVATION_END)
    prediction_end  = pd.Timestamp(PREDICTION_END)

    # X_pred sert juste à aller récupérer les acheteurs (y) pendant cette période
    X_obs = X[X["event_time"] < observation_end].copy()
    X_pred = X[(X["event_time"] >= observation_end) & (X["event_time"] < prediction_end)].copy()

    # 1-Create a purchasers set with all the purchasers in the prediction period
    # 2-Create a y_purchasers dataframe with all unique user_id in the observation period
    # 3-Create a label column in y_purchasers, with 1 if the user_id is a purchaser during prediction period else 0 (If 0, the user_id might be inactive in the prediction period, or not a buyer)
    purchasers = set(X_pred.loc[X_pred["event_type"] == "purchase", "user_id"])
    y_purchasers = pd.DataFrame({"user_id": X_obs["user_id"].unique()})
    y_purchasers["label"] = y_purchasers["user_id"].isin(purchasers).astype(int)

    # Feature engineering X_obs
    X_feat_eng, long_session_users = feature_engineering(X_obs)

    #Reset user_id index from the groupby in feature_engineering
    X_feat_eng = X_feat_eng.reset_index()

    # Grouping with y_purchasers
    dataset = y_purchasers.merge(X_feat_eng, on="user_id", how="inner")

    # Filtering out long_session_users
    dataset = dataset[~dataset["user_id"].isin(long_session_users["user_id"])]

    X_processed = dataset.drop(columns="label")
    X_processed = X_processed.drop(columns="user_id")

    y_processed = dataset["label"]

    print("✅ X_processed, with shape", X_processed.shape)
    print("✅ y_processed, with shape", y_processed.shape)

    return X_processed, y_processed


def feature_engineering(X_obs: pd.DataFrame) -> pd.DataFrame:

    # Observation and prediction set split
    observation_end = pd.Timestamp(OBSERVATION_END)
    prediction_end  = pd.Timestamp(PREDICTION_END)

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

    # Capturer les sessions de 18h ou plus
    long_sessions = session_stats[session_stats['session_duration'] >= pd.Timedelta(hours=18)]
    # Créer un dataframe long_session_users avec les user_id concernés (filtré plus tard sur "dataset" dans la fonction preprocess_features)
    long_session_users = long_sessions[['user_id']].drop_duplicates().reset_index(drop=True)

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

    # Features product diversity (Compte le nombre de produits et catégories différentes des events user_id, et si un produit est revisité)
    diversity = X_obs.groupby("user_id").agg(
    unique_products   = ("product_id", "nunique"),
    unique_categories = ("category_id", "nunique"),
    unique_brands     = ("brand", "nunique"),
    )

    diversity["product_revisit_rate"] = 1 - (
    diversity["unique_products"] /
    behavior["total_events"].replace(0, 1)
    )

    # Features de prix, regarde des statistiques sur le prix des produits associés à des évènements d'un user_id sur la période d'observation
    price_feats = X_obs.groupby("user_id")["price"].agg(
    avg_price    = "mean",
    median_price = "median",
    max_price    = "max",
    min_price    = "min",
    )
    price_feats["price_range"] = price_feats["max_price"] - price_feats["min_price"]

    # Features temporelles, calcule la proximité des events avec la fin de la période d'observation
    temporal = X_obs.groupby("user_id").agg(
    last_event_ts  = ("event_time", "max"),
    first_event_ts = ("event_time", "min"),
    weekend_ratio  = ("is_weekend", "mean"),
    )

    # Enlève les colonnes au format datetime64 et en crée des nouvelles numériques
    temporal["recency_seconds"] = (observation_end - temporal["last_event_ts"]).dt.total_seconds()
    temporal["user_lifetime_seconds"] = (temporal["last_event_ts"] - temporal["first_event_ts"]).dt.total_seconds()
    temporal = temporal.drop(columns=["last_event_ts", "first_event_ts"])


    # Features grouping and X_features_engineering
    X_feat_eng = (behavior
                 .join(session_user, how="left")
                 .join(diversity, how="left")
                 .join(price_feats, how="left")
                 .join(temporal, how="left")
    )

    X_feat_eng  = X_feat_eng.astype("float32")

    return X_feat_eng, long_session_users
