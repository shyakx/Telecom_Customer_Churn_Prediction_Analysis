import numpy as np
import pandas as pd
from model import load_model
from preprocessing import preprocess_data

def predict(data):
    """Make predictions using the trained model."""
    
    try:
        model = load_model()
        processed_data = preprocess_data(data)
        
        prediction_prob = model.predict(processed_data)
        prediction = (prediction_prob > 0.5).astype(int)
        
        return {"churn_probability": prediction_prob.tolist(), "prediction": prediction.tolist()}
    
    except Exception as e:
        return {"error": str(e)}
