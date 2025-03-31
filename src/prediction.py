import pandas as pd
from .model import load_model

def predict_churn(input_data, model_path='../models/best_model.pkl'):
    """Make churn predictions on new data"""
    # Load model
    model = load_model(model_path)
    
    # Convert input to DataFrame if it isn't already
    if not isinstance(input_data, pd.DataFrame):
        input_data = pd.DataFrame(input_data)
    
    # Make predictions
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)[:, 1]
    
    return {
        'predictions': predictions.tolist(),
        'probabilities': probabilities.tolist()
    }

def batch_predict(input_file, output_file=None, model_path='../models/best_model.pkl'):
    """Make batch predictions from a CSV file"""
    # Load data
    data = pd.read_csv(input_file)
    
    # Get predictions
    results = predict_churn(data, model_path)
    
    # Add predictions to data
    data['Churn_Prediction'] = results['predictions']
    data['Churn_Probability'] = results['probabilities']
    
    # Save or return results
    if output_file:
        data.to_csv(output_file, index=False)
        return f"Predictions saved to {output_file}"
    return data