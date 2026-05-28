import os
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_models():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(base_dir, 'models')
    
    models_path = os.path.join(models_dir, 'all_models.pkl')
    test_data_path = os.path.join(models_dir, 'test_data.pkl')
    
    with open(models_path, 'rb') as f:
        models = pickle.load(f)
        
    with open(test_data_path, 'rb') as f:
        X_test, y_test = pickle.load(f)
        
    results = []
    
    print("Evaluating models...")
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        })
        
    results_df = pd.DataFrame(results).set_index('Model')
    print("\nModel Comparison Table:")
    print(results_df.round(4))
    
    # Select best model based on F1-Score
    best_model_name = results_df['F1-Score'].idxmax()
    best_model = models[best_model_name]
    print(f"\nBest Model selected: {best_model_name}")
    
    # Save the best model
    best_model_path = os.path.join(models_dir, 'churn_model.pkl')
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"Best model saved to {best_model_path}")

    # Display basic feature importance if applicable
    if hasattr(best_model, 'feature_importances_'):
        importances = pd.Series(best_model.feature_importances_, index=X_test.columns)
        importances = importances.sort_values(ascending=False).head(10)
        print("\nTop 10 Feature Importances:")
        print(importances)
    elif hasattr(best_model, 'coef_'):
        importances = pd.Series(best_model.coef_[0], index=X_test.columns)
        importances = importances.abs().sort_values(ascending=False).head(10)
        print("\nTop 10 Feature Importances (Absolute Coefficients):")
        print(importances)

if __name__ == "__main__":
    evaluate_models()
