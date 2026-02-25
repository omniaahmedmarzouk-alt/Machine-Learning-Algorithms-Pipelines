# ⚙️ ML Algorithms From Scratch (Dynamic Dashboard)

This project demonstrates a deep understanding of the mathematics behind Machine Learning algorithms. Instead of relying on `sklearn`, I built the training loops, error calculations, and weight updates completely from scratch.

## 🧠 What's Inside?

### 1. Linear Regression
Implemented three different Gradient Descent tricks to update weights and bias:
* **The Square Trick:** Standard gradient descent using squared errors.
* **The Absolute Trick:** Using absolute errors.
* **The Simple Trick:** Fixed-step learning rate adjustments.

### 2. Logistic Regression
* Programmed the **Sigmoid Function** to map predictions between 0 and 1.
* Implemented the **Log Loss (Cross-Entropy)** function to monitor the model's learning progress.
* Built the weight update rules using the Logistic trick.

## 🚀 Interactive Streamlit Dashboard
I wrapped both algorithms in a dynamic web application using **Streamlit**. 
The app allows users to:
* Input custom data (`X` and `y`).
* Choose between Linear or Logistic Regression.
* Select the training trick, adjust `Epochs` and `Learning Rate`.
* Watch the training progress (and Log Loss reduction) in real-time.
* Make instant predictions on new data.

**To run the dashboard:**
```bash
streamlit run app.py