import streamlit as st
import joblib
import os
import warnings

# Fix KMeans memory leak issue on Windows and suppress warnings
os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings("ignore")

# Load the trained model safely
model_path = os.path.join(os.path.dirname(__file__), 'kmeans_model.pkl')
model = joblib.load(model_path)

# App UI
st.title("🛒 Mall Customer Segmentation")
st.write("Enter the customer's data to predict their segment:")

# Input fields
income = st.number_input("Annual Income (k$)", min_value=0, max_value=200, value=50)
score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

# Prediction button
if st.button("Predict Segment"):
    prediction = model.predict([[income, score]])
    cluster = prediction[0]
    
    st.success(f"🎯 This customer belongs to Cluster: {cluster}")
    st.info("💡 Note: Map this cluster number to your specific personas (e.g., VIP, Saver).")