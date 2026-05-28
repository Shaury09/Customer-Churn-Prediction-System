# Customer Churn Prediction System

This is a complete Machine Learning project aimed at predicting whether a customer will churn based on demographic and account usage information.

## Project Overview
The "Customer Churn Prediction System" uses the Telco Customer Churn dataset to build an end-to-end Machine Learning pipeline. The project provides predictive analytics through an interactive Streamlit dashboard, helping stakeholders quickly identify high-risk customers alongside feature importances.

## Dataset Description
The [Telco Customer Churn dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) contains customer-level data such as:
- **Demographics:** Gender, SeniorCitizen, Partner, Dependents
- **Account details:** Tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Churn
- **Services:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies

## Folder Structure
```
customer_churn_project/
├── data/
│   └── telco_churn.csv            # Dataset (downloaded via script)
├── notebooks/
│   └── EDA.ipynb                  # Exploratory Data Analysis
├── src/
│   ├── data_preprocessing.py      # Cleans missing values/indexes
│   ├── feature_engineering.py     # Encoding, Scaling, and SMOTE
│   ├── train_model.py             # Trains classification models
│   ├── evaluate_model.py          # Finds and saves the best model
│   └── predict.py                 # Core function predicting specific customer risk
├── models/
│   ├── churn_model.pkl            # Final Best Estimator
│   ├── scaler.pkl                 # StandardScaler instance
│   └── encoder.pkl                # OneHotEncoder instance
├── app/
│   └── streamlit_app.py           # Interactive Dashboard UI
├── requirements.txt
└── README.md
```

## Installation Instructions
1. Navigate to the project directory:
   ```bash
   cd customer_churn_project
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download data and train models (optional if already performed):
   ```bash
   python fetch_data.py
   python src/train_model.py
   python src/evaluate_model.py
   ```

## Running the Streamlit App
To run the interactive Streamlit dashboard, use the command:
```bash
streamlit run app/streamlit_app.py
```
This will launch the application in your default web browser containing the prediction interface, EDA insights, and feature importance.

## Example Output
**Input context:**
- Tenure = 5
- Contract = Month-to-month
- MonthlyCharges = 90.0
- TechSupport = No
- PhoneService = Yes
- InternetService = DSL

**Output:**
- Churn Probability: ~60-85% (Depends on full feature set)
- Risk Level: High / Medium
