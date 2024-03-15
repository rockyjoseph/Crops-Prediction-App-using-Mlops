import numpy as np
import pandas as pd
import streamlit as st

import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# DATASET
df = pd.read_csv('artifacts/data.csv')

# USER MENU
st.sidebar.title('CROPS RECOMMENDER APP')
user_menu = st.sidebar.radio('Select an Option',
                             ('Dataset', 'Requirements', 'App'))


if user_menu == 'Dataset':
    # DATASET
    st.header('Dataset')
    st.dataframe(df)

    # df.shape
    st.header('Dataset Infomation')
    st.code('The dataset consits of 5109 rows and 11 columns')

    # DESCRIPTION OF THE DATA
    st.header('Description of the data')
    st.table(df.describe())

if user_menu == 'Requirements':
    # TITLE
    st.title('Crops Requirements')
    
    # SORTING TO GET REQUIRED CROPS TO GROW
    sorting = df.groupby(by='label').mean().reset_index()

    # TOP 10 MOST REQUIRED CROPS
    st.text('------------------------------------------------------------------')
    for feature in sorting.columns[1:]:
        st.header(f'Top 5 most {feature} requiring crops:')
        st.text('------------------------------------------------------------------')
        for crop ,values in sorting.sort_values(by=feature, ascending=False)[:5][['label', feature]].values:
            st.write(f'{crop} --> {values}')
        st.text('------------------------------------------------------------------')

    # TOP 10 LEAST REQUIRED CROPS
    for feature in sorting.columns[1:]:
        st.header(f'Top 5 least {feature} requiring crops:')
        st.text('------------------------------------------------------------------')
        for crop ,values in sorting.sort_values(by=feature)[:5][['label', feature]].values:
            st.markdown(f' {crop} --> {values}')
        st.text('------------------------------------------------------------------')

    # CROPS COMPARISON
    st.title('Crops Comparison')
    st.text('------------------------------------------------------------------')

    fig = px.scatter(x=df['phosphorus'], y=df['potassium'], color=df['label'], title="Phosphorus VS Potassium")
    st.plotly_chart(fig)
    
    st.text('------------------------------------------------------------------')

    fig = px.scatter(x=df['nitrogen'], y=df['potassium'], color=df['label'], title="Phosphorus VS Potassium")
    st.plotly_chart(fig)

    st.text('------------------------------------------------------------------')

    fig = px.scatter(x=df['nitrogen'], y=df['phosphorus'], color=df['label'], title="Phosphorus VS Potassium")
    st.plotly_chart(fig)

    st.text('------------------------------------------------------------------')

    fig = px.scatter_3d(x=df['nitrogen'], y=df['phosphorus'], z=df['potassium'], color=df['label'], title='3-Dimensional Comparison')
    st.plotly_chart(fig)

    st.text('------------------------------------------------------------------')

if user_menu == 'App':
    # TITLE
    st.title('CROPS RECOMMANDATION APP')

    # VARIABLES DECLARATION
    nitrogen = st.selectbox('Nirogen', df['nitrogen'].unique())
    phosphorus = st.selectbox('Phosphorus', df['phosphorus'].unique())
    potassium = st.selectbox('Potassium', df['potassium'].unique())
    temperature = st.selectbox('Temperature', df['temperature'].unique())
    humidity = st.selectbox('Humidity', df['humidity'].unique())
    ph = st.selectbox('Ph', df['ph'].unique())
    rainfall = st.selectbox('Rainfall', df['rainfall'].unique())

    # SENDING DATA TO PREDICT PIPELINE
    data = CustomData(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)

    # PREDICTING THE OUTPUT
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

        st.title('The Predicted Crop to grow is '+ str(prediction(results[0])))
