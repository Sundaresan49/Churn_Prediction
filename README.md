## 🧠 Churn Prediction App

**Churn Prediction** is a web app built with **Streamlit** that allows users to predict whether a customer is likely to churn (leave a service or subscription) based on inputs like age, gender, tenure, and monthly charges.

---

### 🚀 Demo

Users input basic customer information, and the app returns a prediction:
📌 `churn` or `not churn`, along with the model's confidence.

---

## 📦 Features

* Simple and interactive UI using **Streamlit**
* Machine learning model built with **Logistic Regression**
* Uses **StandardScaler** for input normalization
* Shows **probability of churn** if model supports it
* Designed to handle **imbalanced data** using `class_weight='balanced'`

---

## 🧮 How It Works

### 1. **Input Collection**

The app collects the following inputs:

* Age (numeric)
* Gender (Male or Female, encoded as binary)
* Tenure (how long the user has been with the service)
* Monthly Charges

### 2. **Preprocessing**

Inputs are converted into a numerical feature array:

```python
[age, gender (1 if female), tenure, monthlycharges]
```

This array is then scaled using a pre-fitted **StandardScaler** (`scaler.pkl`), which ensures that all values are on the same scale as during model training.

### 3. **Prediction**

The scaled inputs are passed into a **Logistic Regression model** (`model.pkl`) that predicts:

* `0` → Not Churn
* `1` → Churn

It also optionally shows the churn **probability** using `model.predict_proba()`.

---

## 🧠 Model Training Logic

The model is trained on a dataset (`churn.csv`) with the following features:

* `age`
* `gender` (encoded)
* `tenure`
* `MonthlyCharges`

### ➕ Key Logic

* **Logistic Regression** is used due to its interpretability and performance on binary classification tasks.
* To handle **class imbalance** (more non-churn than churn cases), the model uses:

  ```python
  LogisticRegression(class_weight='balanced')
  ```
* Inputs are scaled using:

  ```python
  StandardScaler().fit(X_train)
  ```

---

## 🔧 Technologies Used

| Tool           | Purpose                                      |
| -------------- | -------------------------------------------- |
| `Streamlit`    | Web application frontend                     |
| `scikit-learn` | Model training, logistic regression, scaling |
| `joblib`       | Save/load trained models and scalers         |
| `NumPy`        | Numerical operations                         |
| `pandas`       | Data handling (in training phase)            |

---

## ▶️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/churn-prediction-app.git
cd churn-prediction-app
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> Or install individually:

```bash
pip install streamlit scikit-learn joblib numpy
```

### 3. Run the App

```bash
streamlit run app.py
```

> This will open a browser window at `http://localhost:8501`.

---

## 📁 Files Explained

| File             | Purpose                                      |
| ---------------- | -------------------------------------------- |
| `app.py`         | Main Streamlit app logic                     |
| `model.pkl`      | Trained logistic regression model            |
| `scaler.pkl`     | StandardScaler used for input preprocessing  |
| `notebook.ipynb` | Jupyter notebook used for training the model |
| `churn.csv`      | Training dataset (not included here)         |

---

## 📊 Sample Prediction

| Input           | Value  |
| --------------- | ------ |
| Age             | 25     |
| Gender          | Female |
| Tenure          | 2      |
| Monthly Charges | 135.0  |

→ **Prediction:** `churn`
→ **Probability:** `0.74`

---


---

## 🙏 Acknowledgements

Inspired by real-world churn prediction use cases in telecom and SaaS industries.

---



