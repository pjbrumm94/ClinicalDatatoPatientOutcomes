import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load Data
df = pd.read_csv('data/raw_data.csv')

# Handle Missing Values
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Feature Scaling
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df_imputed.columns)

# Save Cleaned Data for Pipeline Integration
df_scaled.to_csv('data/cleaned_data.csv', index=False)

# Save Preprocessing Objects
joblib.dump(imputer, 'models/imputer.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("Data preprocessing complete. Cleaned data saved to 'data/cleaned_data.csv'.")
