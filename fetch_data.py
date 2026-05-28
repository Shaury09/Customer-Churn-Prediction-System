import pandas as pd
import os

url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
output_path = os.path.join(os.path.dirname(__file__), 'data', 'telco_churn.csv')

print(f"Downloading Telco Customer Churn dataset from {url}...")
df = pd.read_csv(url)
print(f"Dataset downloaded. Shape: {df.shape}")

# Saving to CSV
df.to_csv(output_path, index=False)
print(f"Dataset successfully saved to {output_path}")
