import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE
import pickle
import os

def engineer_features(df, is_training=True, scaler_path=None, encoder_path=None):
    """
    Encodes categorical features, scales numerical features.
    If is_training is True, applies SMOTE to balance the dataset.
    Returns: X, y, (and potentially balanced X, y if training)
    """
    df_feat = df.copy()

    # Define feature types
    target_col = 'Churn'
    
    if target_col in df_feat.columns:
        # Convert target to binary
        df_feat[target_col] = df_feat[target_col].map({'Yes': 1, 'No': 0})
        y = df_feat[target_col]
        X = df_feat.drop(target_col, axis=1)
    else:
        y = None
        X = df_feat
        
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_cols = [col for col in X.columns if col not in num_cols]

    # Pre-process inputs (e.g., lowercase columns if mismatch, but data usually matches)
    for col in num_cols:
        if col not in X.columns:
            # Maybe title case issue
            col_title = col.title() if col != 'TotalCharges' else 'TotalCharges'
            if col_title in X.columns:
                num_cols[num_cols.index(col)] = col_title
            
    # Load or initialize preprocessors
    if is_training:
        scaler = StandardScaler()
        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        
        # Fit and transform
        X_num = scaler.fit_transform(X[num_cols])
        X_cat = encoder.fit_transform(X[cat_cols])
        
        # Save preprocessors
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        with open(encoder_path, 'wb') as f:
            pickle.dump(encoder, f)
            
    else:
        # Load preprocessors for inference
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)
            
        # Transform (handle unknown categories nicely if any arise)
        X_num = scaler.transform(X[num_cols])
        # Need to ensure we don't fail on unseen
        X_cat = encoder.transform(X[cat_cols])

    # Convert back to DataFrame for feature names
    X_num_df = pd.DataFrame(X_num, columns=num_cols)
    cat_feature_names = encoder.get_feature_names_out(cat_cols)
    X_cat_df = pd.DataFrame(X_cat, columns=cat_feature_names)
    
    X_processed = pd.concat([X_num_df, X_cat_df], axis=1)
    
    if is_training and y is not None:
        print(f"Original class distribution: {y.value_counts().to_dict()}")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_processed, y)
        print(f"Resampled class distribution: {pd.Series(y_resampled).value_counts().to_dict()}")
        return X_resampled, y_resampled
    
    return X_processed, y

if __name__ == "__main__":
    from data_preprocessing import load_data, preprocess_data
    filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'telco_churn.csv')
    df = load_data(filepath)
    df_clean = preprocess_data(df)
    
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    encoder_path = os.path.join(models_dir, 'encoder.pkl')
    
    X_train, y_train = engineer_features(df_clean, is_training=True, scaler_path=scaler_path, encoder_path=encoder_path)
    print(f"X_train shape after engineering: {X_train.shape}")
