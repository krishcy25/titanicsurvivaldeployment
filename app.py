#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pickle
from sklearn.externals import joblib
import streamlit as st
import pandas as pd
import numpy as np

model = joblib.load("xgb_titanic.pkl")


def predict(input_df):
    predictions_df = model.predict(data=input_df)
    #predictions = predictions_df['Label'][0]
    return predictions_df

def run():

    from PIL import Image
    image = Image.open('logo1.PNG')
    image_hospital = Image.open('house.PNG')
    
    #primary_color = "#FFFD80"
    
    
    #if primary_color != "#000000":
        #st.markdown(f"<style> body{{ background-color: {primary_color};}}</style>",unsafe_allow_html=True)
    
    #secondary_color = "#262730"
    
    
    st.image(image,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app is created to predict Titanic survival using XGBoost Model by "Krishna Yarlagadda"')
    st.sidebar.success('https://xgboost.ai/about')
    
    st.sidebar.image(image_hospital)

    st.title("Titanic Passenger Survival Prediction Application")

    if add_selectbox == 'Online':
        
        
        Pclass = st.selectbox('Pclass: Enter ticket class for passenger', [1,2,3])
        Sex = st.selectbox('Sex: Enter 0 for Male and 1 for Female', [0,1])
        Age = st.number_input('Age', min_value=1, max_value=200, value=1)
        SibSp= st.number_input('SibSp: Enter the number of siblings on board', min_value=1, max_value=100, value=1)
        Parch = st.number_input('Parch: Enter the number of parents/children on board', min_value=1, max_value=100, value=1)
        Fare = st.number_input('Fare :Enter the fare of the ticket', min_value=1, max_value=200000, value=100)
        Embarked = st.selectbox('Embarked: Enter the port of Embarkation 0=Cherbourg, 1=Queenstown 2=Southamption', [0,1,2])

        output=""

        input_dict = {'Pclass' :Pclass, 'Sex' : Sex, 'Age' : Age,
                      'SibSp' : SibSp
                    ,'Parch' : Parch,'Fare' : Fare,'Embarked' :Embarked}
        input_df = pd.DataFrame([input_dict])

        if st.button("Predict"):
            output = predict(input_df=input_df)
            output = 'Predicted_Survival(1-Yes,0-No denoted in [] )  ' + ' ' + str(output)

        st.success('The output is {}'.format(output))

    if add_selectbox == 'Batch':

        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_upload is not None:
            data = pd.read_csv(file_upload)
            predictions = predict(data=data)
            st.write(predictions)

if __name__ == '__main__':
    run()

