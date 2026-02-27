import streamlit as st
import pandas as pd
import joblib

# 1.load model & files
model = joblib.load('logistic_model.pkl')
scaler = joblib.load('scaler.pkl')
model_features = joblib.load('model_features.pkl')

# 2. UI
st.title("💰 Adult Income Prediction App")
st.write("Predict if a person earns >50K or <=50K a year based on demographic data.")

# Enter data by user
age = st.slider("Age", 17, 90, 30)
hours_per_week = st.slider("Hours per week", 1, 100, 40)
education_num = st.slider("Years of Education", 1, 16, 10)

marital_status = st.selectbox("Marital Status", [' Married-civ-spouse', ' Never-married', ' Divorced', ' Separated', ' Widowed'])
occupation = st.selectbox("Occupation", [' Tech-support', ' Craft-repair', ' Other-service', ' Sales', ' Exec-managerial', ' Prof-specialty'])
gender = st.selectbox("Gender", [" Male", " Female"])

# 3. predict button
if st.button("Predict Income"):
    # collect data in dataframe
    user_data = pd.DataFrame({
        'Age': [age],
        'Hours_per_week': [hours_per_week],
        'Education_num': [education_num],
        'Marital_status': [marital_status],
        'Occupation': [occupation],
        'Sex': [1 if gender == " Male" else 0],
    })
    
    #One-Hot Encoding 
    user_data = pd.get_dummies(user_data)
    
    # مطابقة أعمدة المستخدم مع أعمدة الموديل اللي اتدرب عليها (أهم خطوة)
    user_data = user_data.reindex(columns=model_features, fill_value=0)
    
    #Scaling
    user_data_scaled = scaler.transform(user_data)
    
    #predict
    prediction = model.predict(user_data_scaled)
    
    if prediction[0] == 1:
        st.success("🎉 This person is predicted to earn >50K!")
    else:
        st.warning("💵 This person is predicted to earn <=50K.")