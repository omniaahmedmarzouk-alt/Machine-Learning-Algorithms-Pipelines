import streamlit as st
import pandas as pd
import joblib
import os

# Configure page settings
st.set_page_config(page_title="University Admission Predictor", page_icon="🎓", layout="centered")

# Load the trained model robustly using 'os' to prevent path errors during deployment
@st.cache_resource
def load_model():
    # Construct absolute path to the model file dynamically
    model_path = os.path.join(os.path.dirname(__file__), 'admission_model.pkl')
    return joblib.load(model_path)

model = load_model()

# App Header
st.title("🎓 University Admission Predictor")
st.markdown("Enter your academic details below to estimate your admission chances based on our Machine Learning model.")

# Input form using columns for better UI layout
col1, col2 = st.columns(2)

with col1:
    gre_score = st.number_input("GRE Score", min_value=290, max_value=340, value=310)
    toefl_score = st.number_input("TOEFL Score", min_value=90, max_value=120, value=100)
    university_rating = st.selectbox("University Rating", [1, 2, 3, 4, 5])
    research = st.radio("Research Experience", ["Yes", "No"])

with col2:
    sop = st.slider("SOP (Statement of Purpose) Strength", 1.0, 5.0, 3.0, 0.5)
    lor = st.slider("LOR (Letter of Recommendation) Strength", 1.0, 5.0, 3.0, 0.5)
    cgpa = st.number_input("CGPA", min_value=6.0, max_value=10.0, value=8.0, step=0.1)

# Convert categorical input to binary format expected by the model
research_val = 1 if research == "Yes" else 0

# Prediction trigger
if st.button("Predict Admission Chance 🚀"):
    # Construct a DataFrame with exact feature names used during training
    input_data = pd.DataFrame({
        'GRE Score': [gre_score],
        'TOEFL Score': [toefl_score],
        'University Rating': [university_rating],
        'SOP': [sop],
        'LOR': [lor], 
        'CGPA': [cgpa],
        'Research': [research_val]
    })

    # Generate prediction
    prediction = model.predict(input_data)[0]

    # Display results
    st.markdown("---")
    if prediction: 
        st.success("🎉 Congratulations! You have a high probability of admission.")
        st.balloons()
    else:
        st.error("⚠️ It might be challenging. Consider improving your scores, especially your CGPA or GRE.")