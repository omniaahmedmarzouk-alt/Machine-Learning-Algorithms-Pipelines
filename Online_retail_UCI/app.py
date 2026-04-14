import streamlit as st
import joblib
import numpy as np
import os

# Get the directory where app.py is located
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'model.pkl')

# Load the model using the absolute path
model = joblib.load(model_path)

st.title('Customer Value Predictor')

recency = st.number_input('Recency (days)', min_value=0)
frequency = st.number_input('Frequency', min_value=0)
r_score = st.slider('R Score', 1, 5)
f_score = st.slider('F Score', 1, 5)
m_score = st.slider('M Score', 1, 5)
rfm_total = r_score + f_score + m_score

if st.button('Predict'):
    X = np.array([[recency, frequency, r_score, f_score, m_score, rfm_total]])
    prediction = model.predict(X)
    
    if prediction[0] == 1:
        st.success('High Value Customer!')
    else:
        st.warning('Regular Customer')
