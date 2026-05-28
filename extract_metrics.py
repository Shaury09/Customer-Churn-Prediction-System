import json
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

base_dir = r"c:\Users\Admin\OneDrive\Desktop\customer_churn_project"
models_path = os.path.join(base_dir, 'models', 'all_models.pkl')
test_data_path = os.path.join(base_dir, 'models', 'test_data.pkl')

with open(models_path, 'rb') as f:
    models = pickle.load(f)

with open(test_data_path, 'rb') as f:
    X_test, y_test = pickle.load(f)

results = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    results[name] = {
        'Accuracy': float(accuracy_score(y_test, y_pred)),
        'Precision': float(precision_score(y_test, y_pred)),
        'Recall': float(recall_score(y_test, y_pred)),
        'F1-Score': float(f1_score(y_test, y_pred)),
        'ROC-AUC': float(roc_auc_score(y_test, y_prob))
    }

with open(os.path.join(base_dir, 'metrics.json'), 'w') as f:
    json.dump(results, f, indent=4)

print("Metrics saved to metrics.json")
