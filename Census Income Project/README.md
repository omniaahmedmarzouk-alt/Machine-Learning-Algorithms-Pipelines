# 💰 Adult Income Classification & Deployment

An end-to-end Machine Learning pipeline that predicts whether an individual earns more than $50K/year based on census data.

## ⚙️ Pipeline Stages:
1. **Exploratory Data Analysis (EDA):** Understanding data distribution and handling missing values.
2. **Data Preprocessing:** Applying `One-Hot Encoding` to categorical features to prevent numerical bias, and using `StandardScaler` for numerical stability.
3. **Model Training:** Training a Logistic Regression model suitable for binary classification.
4. **Deployment:** Building a dynamic web dashboard using **Streamlit**, allowing users to input custom data and get instant predictions.

## 🚀 How to run the Streamlit App:
```bash
streamlit run app.py