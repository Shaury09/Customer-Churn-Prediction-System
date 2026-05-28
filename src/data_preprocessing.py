import pandas as pd
import numpy as np

def load_data(filepath):
    """Loads the customer churn dataset."""
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset loaded with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def preprocess_data(df):
    """
    Cleans the raw dataset:
    - Drops CustomerID column
    - Handles missing values in TotalCharges
    """
    df_clean = df.copy()
    
    # 1. Drop irrelevant columns
    if 'customerID' in df_clean.columns:
        df_clean = df_clean.drop('customerID', axis=1)
        print("Dropped 'customerID' column.")

    # 2. Handle TotalCharges missing values
    # TotalCharges is sometimes empty string " " instead of NaN
    if 'TotalCharges' in df_clean.columns:
        # Replace empty strings with NaN
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        
        # Fill missing values with median or drop. 
        # Since it's usually new customers (Tenure=0), we can fill with 0
        missing_count = df_clean['TotalCharges'].isnull().sum()
        if missing_count > 0:
            df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0)
            print(f"Filled {missing_count} missing values in 'TotalCharges' with 0.")
            
    return df_clean

if __name__ == "__main__":
    # Test preprocessing
    import os
    filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'telco_churn.csv')
    df = load_data(filepath)
    if df is not None:
        df_clean = preprocess_data(df)
        print(df_clean.info())
