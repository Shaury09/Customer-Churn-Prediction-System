import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from data_preprocessing import load_data, preprocess_data
from feature_engineering import engineer_features

def train_models():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, 'data', 'telco_churn.csv')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    encoder_path = os.path.join(models_dir, 'encoder.pkl')
    
    # Load and clean data
    df = load_data(data_path)
    if df is None:
        return
    df_clean = preprocess_data(df)
    
    # Split into train and test sets FIRST to prevent data leakage
    df_train, df_test = train_test_split(df_clean, test_size=0.2, random_state=42, stratify=df_clean['Churn'])
    
    print(f"Train samples: {len(df_train)}, Test samples: {len(df_test)}")
    
    # Feature Engineering
    # Engineer train data (will fit scaler/encoder and apply SMOTE)
    X_train, y_train = engineer_features(df_train, is_training=True, scaler_path=scaler_path, encoder_path=encoder_path)
    
    # Engineer test data (will load scaler/encoder and NOT apply SMOTE)
    X_test, y_test = engineer_features(df_test, is_training=False, scaler_path=scaler_path, encoder_path=encoder_path)
    
    # Initialize Models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    trained_models = {}
    
    # Train Models
    print("\nTraining models...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
    # Save trained models and test set for evaluation
    models_path = os.path.join(models_dir, 'all_models.pkl')
    with open(models_path, 'wb') as f:
        pickle.dump(trained_models, f)
        
    test_data_path = os.path.join(models_dir, 'test_data.pkl')
    with open(test_data_path, 'wb') as f:
        pickle.dump((X_test, y_test), f)
        
    print(f"Models and test data saved to {models_dir}")

if __name__ == "__main__":
    train_models()
