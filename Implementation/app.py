import streamlit as st
import random
import math

def simple_trick(bias, weight, feature, label, learning_rate):
    predicted = bias + (weight * feature)
    if label > predicted:
        if feature > 0:
            weight += learning_rate
            bias += learning_rate
        else:
            weight -= learning_rate
            bias += learning_rate
    elif label < predicted:
        if feature > 0:
            weight -= learning_rate
            bias -= learning_rate
        else:
            weight += learning_rate
            bias -= learning_rate
    return weight, bias

def absolute_trick(bias, weight, feature, label, learning_rate):
    predicted = bias + (weight * feature)
    if label > predicted:
        weight += learning_rate * feature
        bias += learning_rate
    elif label < predicted:
        weight -= learning_rate * feature
        bias -= learning_rate
    return weight, bias

def square_trick(bias, weight, feature, label, learning_rate):
    predicted = bias + (weight * feature)
    error = label - predicted
    weight += learning_rate * feature * error
    bias += learning_rate * error
    return weight, bias

def sigmoid(x):
    # استخدام min/max لتجنب مشاكل الأرقام الكبيرة جداً
    x = max(-700, min(700, x))
    return 1 / (1 + math.exp(-x))

def logistic_trick(bias, weight, feature, label, learning_rate):
    predicted = sigmoid(bias + (weight * feature))
    error = label - predicted
    weight += learning_rate * feature * error
    bias += learning_rate * error
    return weight, bias

def log_loss(pred, label):
    return -(label * math.log(pred) + (1 - label) * math.log(1 - pred))

# ==========================================
# 2. UI
# ==========================================
st.set_page_config(page_title="Dynamic ML Dashboard", layout="wide")
st.title("⚙️ Dynamic Machine Learning Pipeline")
st.write("Train models from scratch on your own data and make instant predictions!")


if 'weights' not in st.session_state:
    st.session_state.weights = None
    st.session_state.bias = None
if 'current_model' not in st.session_state:
    st.session_state.current_model = None

# Side bar
st.sidebar.header("Model Configuration")
model_choice = st.sidebar.selectbox("Choose Algorithm:", ["Linear Regression", "Logistic Regression"])

#Linear Regression tricks
trick_choice = "Square Trick"
if model_choice == "Linear Regression":
    trick_choice = st.sidebar.radio(
        "Select Linear Regression Trick:", 
        ["Square Trick", "Absolute Trick", "Simple Trick"]
    )


if st.session_state.current_model != model_choice:
    st.session_state.weights = None
    st.session_state.bias = None
    st.session_state.current_model = model_choice

st.header(f"1. Train {model_choice} Model")

# data
if model_choice == "Linear Regression":
    default_x = "50, 60, 75, 80, 90, 100, 120, 140, 150, 180"
    default_y = "520, 580, 730, 810, 890, 1050, 1180, 1450, 1480, 1750"
    lr_default = 0.0001 if trick_choice == "Square Trick" else 0.01
else:
    default_x = "1.1, 1.4, 1.8, 2.0, 2.2, 2.8, 3.1, 3.5, 4.0, 4.5"
    default_y = "0, 0, 0, 0, 0, 1, 1, 1, 1, 1"
    lr_default = 0.01

# enter data
col1, col2 = st.columns(2)
with col1:
    x_input = st.text_input("Features (X) - comma separated:", value=default_x)
with col2:
    y_input = st.text_input("Labels (y) - comma separated:", value=default_y)

# setting 
epochs = st.sidebar.slider("Epochs (Iterations)", 100, 5000, 1000, 100)
learning_rate = st.sidebar.number_input("Learning Rate", value=lr_default, format="%.5f")

# button
if st.button("🚀 Train Model"):
    try:
        X = [float(i.strip()) for i in x_input.split(',')]
        y = [float(i.strip()) for i in y_input.split(',')]
        
        if len(X) != len(y):
            st.error("Error: Features (X) and Labels (y) must have the same number of items!")
        else:
            weight = random.random()
            bias = random.random()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for epoch in range(epochs):
                i = random.randint(0, len(X) - 1)
                
                if model_choice == "Linear Regression":
                    if trick_choice == "Simple Trick":
                        weight, bias = simple_trick(bias, weight, X[i], y[i], learning_rate)
                    elif trick_choice == "Absolute Trick":
                        weight, bias = absolute_trick(bias, weight, X[i], y[i], learning_rate)
                    else:
                        weight, bias = square_trick(bias, weight, X[i], y[i], learning_rate)
                else:
                    weight, bias = logistic_trick(bias, weight, X[i], y[i], learning_rate)
                    
                if epoch % max(1, (epochs // 10)) == 0 or epoch == epochs - 1:
                    progress_bar.progress((epoch + 1) / epochs)
                    if model_choice == "Logistic Regression":
                        pred = sigmoid(bias + (weight * X[i]))
                        loss = log_loss(pred, y[i])
                        status_text.text(f"Training... Epoch {epoch}/{epochs} | Log Loss: {loss:.4f}")
                    else:
                        status_text.text(f"Training with {trick_choice}... Epoch {epoch}/{epochs}")
            
            st.session_state.weights = weight
            st.session_state.bias = bias
            st.success(f"✅ Training Complete! Weight: {weight:.4f}, Bias: {bias:.4f}")
            
    except Exception as e:
        st.error(f"Invalid input data. Please check your formatting. Error: {e}")

st.markdown("---")
st.header("2. Make a Prediction")

if st.session_state.weights is not None:
    test_val = st.number_input("Enter a new Feature (X) to predict:", value=10.0)
    
    if st.button("🎯 Predict"):
        w = st.session_state.weights
        b = st.session_state.bias
        
        if model_choice == "Linear Regression":
            result = b + (w * test_val)
            st.info(f"Predicted Output (Y): {result:.2f}")
        else:
            prob = sigmoid(b + (w * test_val))
            st.info(f"Probability: {prob * 100:.2f}%")
            if prob > 0.5:
                st.error("⚠️ Classification: Class 1 (Positive)")
            else:
                st.success("✅ Classification: Class 0 (Negative)")
else:
    st.warning("Please train the model first to unlock predictions!")

st.sidebar.markdown("---")
st.sidebar.write("Developed by Omnia")