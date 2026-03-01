# 🎓 University Admission Prediction & Overfitting Analysis

This project predicts the probability of university admission based on student metrics (GRE, TOEFL, CGPA, etc.). 
The main objective is to practically demonstrate the **Overfitting problem** in Machine Learning and how to overcome it using standard engineering practices. 

## 🛠️ Techniques & Models Used
* **Data Preprocessing:** Pandas, Seaborn Heatmaps.
* **Problem 1 (Overfitting):** Unpruned Decision Tree & Polynomial Regression (Degree 3).
* **Solutions Applied:** Tree Pruning (`max_depth`) & Ensemble Learning (Random Forest).

## 📊 Key Results & Takeaways
1. **The Overfitting Trap:** The Unpruned Tree scored **100%** on training but dropped to **86%** on testing. Polynomial Regression exploded the features to 164 and failed miserably (**42%** test score).
2. **The Simple Solution (Pruning):** By simply setting `max_depth=6`, the Decision Tree generalized perfectly, achieving a **90%** testing accuracy.
3. **The Robust Solution (Ensemble):** For continuous regression, the **Random Forest** provided a stable and highly reliable $R^2$ score of **85.3%** without needing complex feature engineering.

**Deployment:** Building a dynamic web dashboard using **Streamlit**, allowing users to input custom data and get instant predictions.

## 🚀 How to run the Streamlit App:
```bash
streamlit run app.py