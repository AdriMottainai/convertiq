import streamlit as st
import folium
import pandas as pd


#API_PREDICT_URL = #rajouter l'URL de notre /predict de l'API
#API_PREDICT_PROBA_URL = #rajouter l'URL de notre /predict_proba de l'API

st.set_page_config(page_icon="🛒", page_title="convertIQ")

st.title("convertIQ")

st.subheader("Predict based on user behavior a user is going to buy or not on your Ecommerce site")

@st.cache_data
def get_data() -> pd.DataFrame:
    df = pd.read_csv("/Users/glenhellio/code/AdriMottainai/convertiq/convertiq_py/app/dataset_2user.csv") #PATH final a rajouter apres que Dom et Sophie ont finis avec le dataset de demo
    return df

df = get_data()

st.selectbox("Choisir un user_id", sorted(df["user_id"].unique()))

st.button("Lancer la prediction")