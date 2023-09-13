import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


df = pd.read_csv('artifacts\data.csv')

st.title('Crops Prediction App')


nitrogen = st.selectbox('Nirogen', df['nitrogen'].unique())
phosphorus = st.selectbox('Phosphorus', df['phosphorus'].unique())
potassium = st.selectbox('Potassium', df['potassium'].unique())
temperature = st.selectbox('Temperature', df['temperature'].unique())
humidity = st.selectbox('Humidity', df['humidity'].unique())
ph = st.selectbox('Ph', df['ph'].unique())
rainfall = st.selectbox('Rainfall', df['rainfall'].unique())

data = CustomData(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)

if st.button('Predict'):
    pred_df = data.get_data_as_data_frame()
    
    predict_pipeline = PredictPipeline()

    results = predict_pipeline.predict(pred_df)

    st.title('The Predicted Crops to grow: '+ str(float(np.exp(results[0]))))