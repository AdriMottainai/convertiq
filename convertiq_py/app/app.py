import streamlit as st
import pandas as pd
import requests




API_PREDICT_URL = "http://localhost:8000/predict"
#API_PREDICT_PROBA_URL = #rajouter l'URL de notre /predict_proba de l'API

st.set_page_config(page_icon="🛒", page_title="convertIQ")

st.image("/Users/glenhellio/code/AdriMottainai/convertiq/convertiq_py/app/Convertiq_banner.png")

st.title("convertIQ")

st.subheader("Prediction-based on user behavior...")

df = pd.read_csv("/Users/glenhellio/code/AdriMottainai/convertiq/raw_data/dataset_2user.csv") #PATH final a rajouter apres que Dom et Sophie ont finis avec le dataset de demo

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
    #params = dict()
    df_selected_user = df[df["user_id"]==selected_user].drop(columns = df.columns[0])
    
    df_selected_user.to_csv("csv_selected_user.csv", index=False)
    
    myfiles = {"file": open("csv_selected_user.csv", "rb")}

    #response = requests.get('https://taxifare.lewagon.ai/predict', params = params)
    with open("csv_selected_user.csv", "rb") as f :
        response = requests.post(API_PREDICT_URL, files={"csv":f})
    
    result = response.json()#['user_id', 'probability', 'prediction']
    
    if (result["prediction"] == 1):
        st.success(f"Based on its recent behavior, this user has a {round(result['probability']*100, 2)}% probability of making a purchase in the next 2 days.", icon="🔥")
    else:
        st.error(f"Based on its recent behavior, this user has a {round(result['probability']*100, 2)}% probability of making a purchase in the next 2 days.", icon="❌")