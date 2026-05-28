import streamlit as st
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# Add src to path so we can import our modules
base_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(base_dir, 'src'))

from predict import predict_churn

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Load data and model for insights
@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(base_dir, 'data', 'telco_churn.csv'))
    
@st.cache_resource
def load_model():
    with open(os.path.join(base_dir, 'models', 'churn_model.pkl'), 'rb') as f:
        return pickle.load(f)

df = load_data()
model = load_model()

st.title("Customer Churn Prediction System")

st.markdown("""
This application predicts the likelihood of a customer churning based on their demographic, service, and account information. 
It uses a Machine Learning model trained on the Telco Customer Churn dataset.
""")

tab1, tab2, tab3 = st.tabs(["Churn Prediction", "Dataset Insights", "Feature Importance"])

with tab1:
    st.header("Make a Prediction")
    st.markdown("Enter customer details below to predict their churn probability.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=5)
        
    with col2:
        phoneservice = st.selectbox("Phone Service", ["Yes", "No"])
        multiplelines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internetservice = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        onlinesecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        onlinebackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        
    with col3:
        deviceprotection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        techsupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streamingtv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streamingmovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly = st.number_input("Monthly Charges", min_value=0.0, value=90.0)
    total = st.number_input("Total Charges", min_value=0.0, value=450.0)
    
    if st.button("Predict Churn"):
        customer_data = {
            'gender': gender,
            'SeniorCitizen': senior,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phoneservice,
            'MultipleLines': multiplelines,
            'InternetService': internetservice,
            'OnlineSecurity': onlinesecurity,
            'OnlineBackup': onlinebackup,
            'DeviceProtection': deviceprotection,
            'TechSupport': techsupport,
            'StreamingTV': streamingtv,
            'StreamingMovies': streamingmovies,
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': payment,
            'MonthlyCharges': monthly,
            'TotalCharges': total
        }
        
        # We handle 'No xx service' mappings to ensure strictly correct data matching if it was transformed, although OneHotEncoder handles raw
        result = predict_churn(customer_data)
        
        prob = result['Churn Probability']
        risk = result['Risk Level']
        advice = result.get('Recommendations', [])
        
        st.subheader("Prediction Result")
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric("Churn Probability", f"{prob*100:.1f}%")
        with col_res2:
            color = "green" if risk == "Low" else "orange" if risk == "Medium" else "red"
            st.markdown(f"### Risk Level: <span style='color:{color}'>{risk}</span>", unsafe_allow_html=True)

        if advice:
            st.markdown("#### Recommended actions to reduce churn")
            for item in advice:
                st.markdown(f"- {item}")
            

with tab2:
    st.header("Dataset Insights (EDA)")
    
    if st.checkbox("Show raw data"):
        st.dataframe(df.head(10))
        
    st.subheader("Churn Distribution")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(data=df, x='Churn', ax=ax, palette='Set2', hue='Churn', legend=False)
    st.pyplot(fig)
    
    col_plot1, col_plot2 = st.columns(2)
    with col_plot1:
        st.subheader("Churn vs Contract Type")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(data=df, x='Contract', hue='Churn', ax=ax, palette='Set2')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
    with col_plot2:
        st.subheader("Tenure Distribution by Churn")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.kdeplot(data=df, x='tenure', hue='Churn', ax=ax, fill=True, palette='Set2')
        st.pyplot(fig)
        
with tab3:
    st.header("Feature Importance")
    st.markdown("Top factors influencing customer churn based on the trained model.")
    
    try:
        if hasattr(model, 'feature_importances_'):
            # Load test data to get column names
            with open(os.path.join(base_dir, 'models', 'test_data.pkl'), 'rb') as f:
                X_test, y_test = pickle.load(f)
                
            importances = pd.Series(model.feature_importances_, index=X_test.columns)
            importances = importances.sort_values(ascending=False).head(15)
            
            fig, ax = plt.subplots(figsize=(10,6))
            sns.barplot(x=importances.values, y=importances.index, ax=ax, palette='viridis', hue=importances.index, legend=False)
            ax.set_title("Top 15 Feature Importances")
            st.pyplot(fig)
        elif hasattr(model, 'coef_'):
            with open(os.path.join(base_dir, 'models', 'test_data.pkl'), 'rb') as f:
                X_test, y_test = pickle.load(f)
            importances = pd.Series(model.coef_[0], index=X_test.columns).abs()
            importances = importances.sort_values(ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(10,6))
            sns.barplot(x=importances.values, y=importances.index, ax=ax, palette='viridis', hue=importances.index, legend=False)
            ax.set_title("Top 15 absolute Feature Coefficients")
            st.pyplot(fig)
        else:
            st.info("The selected model does not afford simple feature importances.")
    except Exception as e:
        st.error(f"Could not load feature importance: {e}")
