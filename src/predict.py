import os
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression

def predict_churn(customer_data: dict, model_path=None, scaler_path=None, encoder_path=None):
    """
    Predicts churn probability and risk level for a single customer.
    Args:
        customer_data: dict containing customer features
    Returns:
        dict with probability and risk category
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(base_dir, 'models')
    
    if model_path is None:
        model_path = os.path.join(models_dir, 'churn_model.pkl')
    if scaler_path is None:
        scaler_path = os.path.join(models_dir, 'scaler.pkl')
    if encoder_path is None:
        encoder_path = os.path.join(models_dir, 'encoder.pkl')
        
    # Load model and preprocessors
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)

    # Compatibility fix: some environments may load a LogisticRegression
    # instance without the 'multi_class' attribute, which predict_proba expects.
    if isinstance(model, LogisticRegression) and not hasattr(model, "multi_class"):
        model.multi_class = "auto"
        
    df = pd.DataFrame([customer_data])
    
    # Needs matching numeric columns
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    # If TotalCharges is missing, default to MonthlyCharges * tenure or 0
    if pd.isna(df['TotalCharges'].iloc[0]) or df['TotalCharges'].iloc[0] == "":
        df['TotalCharges'] = df['MonthlyCharges'] * df['tenure']
        
    # Categorical columns
    cat_cols = [col for col in df.columns if col not in num_cols]
    
    # Scale and encode
    X_num = scaler.transform(df[num_cols])
    X_cat = encoder.transform(df[cat_cols])
    
    X_num_df = pd.DataFrame(X_num, columns=num_cols)
    cat_feature_names = encoder.get_feature_names_out(cat_cols)
    X_cat_df = pd.DataFrame(X_cat, columns=cat_feature_names)
    
    X_processed = pd.concat([X_num_df, X_cat_df], axis=1)
    
    # Predict
    prob = model.predict_proba(X_processed)[0][1]
    
    # Determine risk category
    if prob < 0.3:
        risk = "Low"
    elif prob < 0.7:
        risk = "Medium"
    else:
        risk = "High"

    # Generate simple, rule-based retention advice
    advice = []

    # Contract / loyalty related
    contract = df['Contract'].iloc[0]
    tenure_val = df['tenure'].iloc[0]
    if contract == "Month-to-month":
        advice.append(
            "Offer an incentive (discount or added benefits) to move the customer from a month‑to‑month contract "
            "to a one‑year or two‑year plan to increase commitment."
        )
    elif contract in ["One year", "Two year"] and tenure_val > 6:
        advice.append(
            "Proactively reach out before contract renewal with loyalty rewards or a personalized offer so the "
            "customer has a reason to stay."
        )

    # Price sensitivity
    monthly_charges = df['MonthlyCharges'].iloc[0]
    if monthly_charges >= 90:
        advice.append(
            "Review the customer’s plan to see if there is a lower‑cost bundle or a way to remove unused services, "
            "and clearly communicate the savings."
        )

    # Payment and billing experience
    payment_method = df['PaymentMethod'].iloc[0]
    paperless_billing = df['PaperlessBilling'].iloc[0]
    if payment_method == "Electronic check":
        advice.append(
            "Encourage switching from electronic check to automatic bank transfer or credit card with a small "
            "discount to reduce payment friction."
        )
    if paperless_billing == "Yes":
        advice.append(
            "Send clear, concise billing summaries that highlight value received and any discounts so invoices "
            "don’t become a churn trigger."
        )

    # Service quality / support features
    internet_service = df['InternetService'].iloc[0]
    if internet_service != "No":
        for col, feature_name in [
            ('OnlineSecurity', "online security"),
            ('OnlineBackup', "online backup"),
            ('DeviceProtection', "device protection"),
            ('TechSupport', "priority tech support"),
        ]:
            if col in df.columns and df[col].iloc[0] == "No":
                advice.append(
                    f"Offer a trial or discounted upgrade that includes {feature_name} to increase perceived value "
                    "and reduce reasons to leave."
                )

    # General retention guidance based on risk level
    if risk == "High":
        advice.insert(
            0,
            "Assign this customer to a retention specialist for a proactive outreach call and a personalized offer "
            "within the next 7 days."
        )
    elif risk == "Medium":
        advice.insert(
            0,
            "Send a personalized email highlighting how the current services match the customer’s usage and offer a "
            "small incentive for staying."
        )
    else:  # Low risk
        advice.append(
            "Maintain regular positive touchpoints (e.g., thank‑you messages, occasional perks) to keep satisfaction "
            "high and prevent future churn."
        )
        
    return {
        'Churn Probability': round(prob, 2),
        'Risk Level': risk,
        'Recommendations': advice
    }

if __name__ == "__main__":
    test_customer = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 5,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'DSL',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 90.0,
        'TotalCharges': 450.0
    }
    
    result = predict_churn(test_customer)
    print(f"Prediction for test customer: {result}")
