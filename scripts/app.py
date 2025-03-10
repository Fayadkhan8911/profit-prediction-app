import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model
import os

# Get the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the relative paths to the joblib files
model_path = os.path.join(script_dir, '../save/profit_predictor.joblib')
encoder_path = os.path.join(script_dir, '../save/encoder.joblib')

# Load the model and encoder
model = joblib.load(model_path)
encoder = joblib.load(encoder_path)



st.title('Startup Profit Predictor')
st.write('Enter the details of your startup and we will predict the profit for you')

# Get the user input
state=st.selectbox('State', ['Karnataka', 'Gujarat', 'Maharshtra'])
rnd=st.number_input('R&D Spend',max_value=200000.0, value=30000.0)
admin=st.number_input('Administration Spend',max_value=200000.0, value=10000.0)
marketing=st.number_input('Marketing Spend',max_value=500000.0, value=10000.0)

# process the user input
input_data = pd.DataFrame([[state, rnd, admin, marketing]], columns=['State', 'R&D Spend', 'Administration', 'Marketing Spend'])

# one hot encode the state
state_encoded = pd.DataFrame(encoder.transform(input_data[["State"]]),columns=encoder.get_feature_names_out(["State"]))
input_data=input_data.drop('State',axis=1)
input_data=pd.concat([input_data,state_encoded],axis=1)


feature_names=['R&D Spend', 'Administration Spend', 'Marketing Spend',
       'State_Karnataka', 'State_Maharshtra']
for col in feature_names:
    if col not in input_data.columns:
        input_data[col] = 0  # Add missing columns
input_data = input_data[feature_names]  # Reorder to match training

# Predict profit
if st.button("Predict Profit"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Profit: ${prediction:,.2f}")