import numpy as np
import pandas as pd
import streamlit as st

import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# DATASET
df = pd.read_csv('artifacts\data.csv')

# USER MENU
st.sidebar.title('CROPS PREDICTION APP')
user_menu = st.sidebar.radio('Select an Option',
                             ('Overview', 'Dataset', 'EDA', 'Requirements', 'App'))

if user_menu == 'Overview':
    pass

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

if user_menu == 'EDA':
    # TITLE 
    st.title('Explorartory Data Analysis (EDA)')

    # VARIABLES DECLARATION
    nitrogen = df['nitrogen'].nunique()
    phosphorus = df['phosphorus'].nunique()
    potassium = df['potassium'].nunique()
    temperature = df['temperature'].nunique()
    humidity = df['humidity'].nunique()
    ph = df['ph'].nunique()
    rainfall = df['rainfall'].nunique()
    label = df['label'].nunique()

    st.header('Question 1')
    st.write('What are the unique values in each features in the data?')

    # TOP STATS
    st.title("Top Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Nitrogen")
        st.title(nitrogen)
    with col2:
        st.header("Phosphorus")
        st.title(phosphorus)
    with col3:
        st.header("Potassium")
        st.title(potassium)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.header("Temperature")
        st.title(temperature)
    with col2:
        st.header("Humidity")
        st.title(humidity)
    with col3:
        st.header("ph")
        st.title(ph)

    col1, col2 = st.columns(2)
    with col1:
        st.header("Rainfall")
        st.title(rainfall)
    with col2:
        st.header("Label")
        st.title(label)

    # COUNTPLOT VISUALISATION
    st.header('Question 2')
    st.write('What are the count values in each features in the data?')

    countplot_name = st.selectbox('Features for Value Counts & Outlier Detection', df.columns)

    fig = px.histogram(df, x=countplot_name, title=f'{countplot_name} counplot visualisation')
    st.plotly_chart(fig)

    fig = px.bar(df, y=countplot_name, color='label', title=f'{countplot_name} with stroke')
    st.plotly_chart(fig)

    # OUTLIER DETECTION
    st.header('Question 3')
    st.write('What if there are any outliers in the data?')

    fig = px.box(df, y=countplot_name, color='label', title='Outlier Detection (Boxplot)')
    st.plotly_chart(fig)
    countplot_name
    fig = px.violin(df, y=countplot_name, color='label', title='Outlier Detection (Violinplot)')
    st.plotly_chart(fig)

    # SORTING & CHECKING BY MEAN VALUES
    st.header('Question 3')
    st.write('Which crops requires more or less features? Why not check it by the mean?')

    sorting = df.groupby(by='label').mean().reset_index()
    st.table(sorting)

    # CONCLUSION FROM MEAN VALUES
    st.header('From above Observations the mean values concludes:')

    st.markdown('''
        - Cotton requires most Nitrogen.
        - Apple requires most Phosphorus.
        - Grapes requires most Potassium.
        - Papaya requires a hot climate.
        - Coconut requires a humid climate.
        - Chickpea requires high ph in soil.
        - Rice requires huge amount of rainfall or water.
    ''')

    st.header('Question 4')
    st.write("Can we use PCA for multivariate labels in the data?")

    # PCA FOR 2-DIMENSIONAL PLOT
    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(df.drop(['label'], axis=1))
    df_pca = pd.DataFrame(df_pca)
    fig = px.scatter(x=df_pca[0], y=df_pca[1], color=df['label'], title="2-Dimensional plot")
    st.plotly_chart(fig)

    # PCA FOR 3-DIMENSIONAL PLOT
    pca_3d = PCA(n_components=3)
    df_pca3d = pca_3d.fit_transform(df.drop(['label'], axis=1))
    df_pca3d = pd.DataFrame(df_pca3d)
    fig = px.scatter_3d(x = df_pca3d[0], y = df_pca3d[1], z = df_pca3d[2], color = df['label'],
                        title="3-Dimensional plot")
    st.plotly_chart(fig)

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
    st.title('CROPS PREDICTION APP')

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