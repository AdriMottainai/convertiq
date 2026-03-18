import streamlit as st
import folium
import pandas as pd
import requests
import pylint



API_PREDICT_URL = "http://localhost:8000/predict"
#API_PREDICT_PROBA_URL = #rajouter l'URL de notre /predict_proba de l'API

st.set_page_config(page_icon="🛒", page_title="convertIQ")

st.title("convertIQ")

st.subheader("Predict based on user behavior a user is going to buy or not on your Ecommerce site")

df = pd.read_csv("/Users/glenhellio/code/AdriMottainai/convertiq/raw_data/dataset_2user.csv") #PATH final a rajouter apres que Dom et Sophie ont finis avec le dataset de demo

selected_user = st.selectbox("Choisir un user_id", sorted(df["user_id"].unique()))

if st.button("🔮 Predict & Show Route", type="primary"):
    #params = dict()
    df_selected_user = df[df["user_id"]==selected_user].drop(columns = df.columns[0])
    
    df_selected_user.to_csv("csv_selected_user.csv", index=False)
    
    myfiles = {"file": open("csv_selected_user.csv", "rb")}

    #response = requests.get('https://taxifare.lewagon.ai/predict', params = params)
    with open("csv_selected_user.csv", "rb") as f :
        response = requests.post(API_PREDICT_URL, files={"csv":f})
    
    result = response.json()#['user_id', 'probability', 'prediction']
    
    if (result["prediction"] == 1):
        st.success(f"Based on its recent behavior, this user has a {round(result['probability']*100, 2)}% probability of making a purchase in the next 2 days.")
    else:
        st.success(f"Based on its recent behavior, this user has a {round(result['probability']*100, 2)}% probability of making a purchase in the next 2 days.")
