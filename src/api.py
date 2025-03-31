import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load pre-trained scaler and label encoders
scaler_path = "models/scaler.pkl"
encoder_path = "models/label_encoders.pkl"
model_path = "models/best_model.pkl"

def load_model():
    """Load the latest trained model dynamically."""
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def preprocess_data(data: pd.DataFrame):
    """Preprocess input data for prediction or retraining."""
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    with open(encoder_path, 'rb') as file:
        encoders = pickle.load(file)
    
    # Encode categorical features
    categorical_cols = ['State', 'International plan', 'Voice mail plan']
    for col in categorical_cols:
        if col in data.columns and col in encoders:
            data[col] = encoders[col].transform(data[col])
    
    # Scale numerical features
    num_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    data[num_cols] = scaler.transform(data[num_cols])
    
    return data

def predict(data: pd.DataFrame):
    """Make predictions using the trained model."""
    model = load_model()
    processed_data = preprocess_data(data)
    return model.predict(processed_data)

def retrain_model(new_data: pd.DataFrame):
    """Retrain the model with new data and update saved model."""
    X = new_data.drop(columns=['Churn'])
    y = new_data['Churn']
    
    # Preprocess the new data
    X = preprocess_data(X)
    
    # Train a new model
    new_model = RandomForestClassifier(n_estimators=100, random_state=42)
    new_model.fit(X, y)
    
    # Save the updated model
    with open(model_path, 'wb') as file:
        pickle.dump(new_model, file)
    
    return "Model retrained and updated successfully."
