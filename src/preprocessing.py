import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def preprocess_data(data, is_training=False):
    """Preprocess input data for training or inference."""
    
    # Encode categorical values
    for col in ['International plan', 'Voice mail plan']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
    
    # Scale features
    scaler_path = "models/scaler.pkl"
    
    if is_training:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        joblib.dump(scaler, scaler_path)
    else:
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            data_scaled = scaler.transform(data)
        else:
            raise FileNotFoundError("Scaler file not found. Train the model first.")
    
    return data_scaled

def save_scaler(X_train):
    """Fit and save a scaler for future use."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, "models/scaler.pkl")
