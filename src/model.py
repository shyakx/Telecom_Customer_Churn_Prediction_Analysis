import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def load_model():
    """Load the trained model and scaler"""
    model_path = os.path.join('models', 'classifier.pkl')
    scaler_path = os.path.join('models', 'scaler.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler

def predict_churn(model, scaler, features):
    """Make a churn prediction"""
    # Preprocess the input features
    processed_features = preprocess_input(features, scaler)
    
    # Make prediction
    probability = model.predict_proba(processed_features)[0][1]
    prediction = 'Churn' if probability > 0.5 else 'No Churn'
    
    return prediction, round(probability * 100, 2)

def preprocess_input(features, scaler):
    """Preprocess input features to match training data format"""
    # Convert categoricals
    features['International plan'] = features['International plan'].map({'Yes': 1, 'No': 0})
    features['Voice mail plan'] = features['Voice mail plan'].map({'Yes': 1, 'No': 0})
    
    # Select and order features
    feature_cols = [
        'Account length', 'International plan', 'Voice mail plan',
        'Number vmail messages', 'Total day minutes', 'Total day calls',
        'Total day charge', 'Total eve minutes', 'Total eve calls',
        'Total eve charge', 'Total night minutes', 'Total night calls',
        'Total night charge', 'Total intl minutes', 'Total intl calls',
        'Total intl charge', 'Customer service calls'
    ]
    
    features = features[feature_cols]
    
    # Scale features
    scaled_features = scaler.transform(features)
    
    return scaled_features