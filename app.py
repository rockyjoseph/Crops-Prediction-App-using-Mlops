import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

df = pd.read_csv('artifacts\data.csv')

refer = pd.DataFrame()
encoder = LabelEncoder()

crop_names = df['label'].unique()
df['label'] = encoder.fit_transform(df['label'])
encode = pd.DataFrame(df['label'].unique())

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

    def prediction(results):
        if results == 0:
            return 'Apple'
        if results == 1:
            return 'Banana'
        if results == 2:
            return 'Blackgram'
        if results == 3:
            return 'Chickpea'
        if results == 4:
            return 'Coconut'
        if results == 5:
            return 'Coffee'
        if results == 6:
            return 'Cotton'
        if results == 7:
            return 'Grapes'
        if results == 8:
            return 'Jute'
        if results == 9:
            return 'Kidneybeans'
        if results == 10:
            return 'Lentil'
        if results == 11:
            return 'Maize'
        if results == 12:
            return 'Mango'
        if results == 13:
            return 'Mothbeans'
        if results == 14:
            return 'Mungbean'
        if results == 15:
            return 'Muskmelon'
        if results == 16:
            return 'Orange'
        if results == 17:
            return 'Papaya'
        if results == 18:
            return 'Pigeonpeas'
        if results == 19:
            return 'Pomegranate'
        if results == 20:
            return 'Rice'
        if results == 21:
            return 'Watermelon'

    st.title('The Predicted Crop to grow: '+ str(prediction(results[0])))