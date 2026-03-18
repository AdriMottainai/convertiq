import streamlit as st
import pandas as pd
import requests
import os
from convertiq_py.params import *

API_PREDICT_URL = "https://convertiq-docker-2-551516277637.europe-west1.run.app/predict"

st.set_page_config(page_icon="🛒", page_title="convertIQ")

st.image(os.path.join(PROJECT_ROOT,"../data_streamlit/Convertiq_banner.png"))

st.title("convertIQ")

st.subheader("Prediction-based on user behavior...")

df = pd.read_csv(os.path.join(PROJECT_ROOT,"../data_streamlit/dataset_2user.csv"))

selected_user = st.selectbox("Pick a user_id", sorted(df["user_id"].unique()))

st.markdown("""
<style>
div.stButton > button {
    background-color: #ff8fff;
    color: white;
    height: 3em;
    width: 100%;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #ff8fff;
    color: white;
}
</style>
""", unsafe_allow_html=True)
if st.button("🔮 Predict PURCHASE or NOT", type="primary"):
    df_selected_user = df[df["user_id"]==selected_user].drop(columns = df.columns[0])
    
    df_selected_user.to_csv("csv_selected_user.csv", index=False)
    
    myfiles = {"file": open("csv_selected_user.csv", "rb")}

    with open("csv_selected_user.csv", "rb") as f :
        response = requests.post(API_PREDICT_URL, files={"csv":f})
    
    result = response.json()
    
    if (result["prediction"] == 1):
        st.success(f"Based on its recent behavior, this user has a {round(result['probability']*100, 2)}% probability of making a purchase in the next 2 days.", icon="🔥")
    else:
        st.error(f"Based on its recent behavior, this user has a {round(result['probability']*100, 2)}% probability of making a purchase in the next 2 days.", icon="❌")