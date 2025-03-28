import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pickle
import os
from data_preprocessing import preprocess_data

def retrain_model():
    """Retrain model with original + new data"""
    # Create directories if they don't exist
    os.makedirs('data/uploaded', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load original data with proper path handling
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    original_data = pd.read_csv(os.path.join(data_dir, 'raw/telecom_churn.csv'))
    
    # Load any uploaded data
    uploaded_dir = os.path.join(data_dir, 'uploaded')
    uploaded_files = [f for f in os.listdir(uploaded_dir) if f.endswith('.csv')]
    
    if uploaded_files:
        new_data = pd.concat([
            pd.read_csv(os.path.join(uploaded_dir, file)) 
            for file in uploaded_files
        ])
        combined_data = pd.concat([original_data, new_data])
    else:
        combined_data = original_data
    
    # Preprocess data
    X, y = preprocess_data(combined_data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Save model and scaler
    models_dir = os.path.join(os.path.dirname(__file__), '../models')
    with open(os.path.join(models_dir, 'classifier.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    with open(os.path.join(models_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    return {
        'accuracy': accuracy,
        'classification_report': report
    }