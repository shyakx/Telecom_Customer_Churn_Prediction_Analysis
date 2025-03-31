import pandas as pd
from .model import load_model, train_model, evaluate_model, save_model
from .preprocessing import load_preprocessor
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_retraining_need(new_data_path, model_path, threshold=0.05):
    """Check if model needs retraining based on data drift"""
    # Load current model and preprocessor
    model = load_model(model_path)
    
    # Load new data
    new_data = pd.read_csv(new_data_path)
    X_new = new_data.drop('Churn', axis=1, errors='ignore')
    y_new = new_data['Churn'] if 'Churn' in new_data else None
    
    # Get predictions on new data
    if y_new is not None:
        metrics = evaluate_model(model, X_new, y_new)
        current_performance = metrics['roc_auc']
        
        # Compare with original performance (would need to store this)
        original_performance = 0.852  # From initial training
        
        if (original_performance - current_performance) > threshold:
            logger.info("Significant performance drop detected. Retraining recommended.")
            return True
    
    # Additional checks could be added here (data distribution changes, etc.)
    return False

def retrain_model(new_data_path, model_path, preprocessor_path=None):
    """Retrain the model with new data"""
    # Load existing data
    existing_data = pd.read_csv('../data/telecom_churn.csv')
    
    # Load new data
    new_data = pd.read_csv(new_data_path)
    
    # Combine datasets
    combined_data = pd.concat([existing_data, new_data], ignore_index=True)
    
    # Preprocess and train
    from .preprocessing import preprocess_data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(combined_data)
    
    # Train new model
    model = train_model(X_train, y_train, preprocessor)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)
    logger.info("Retrained Model Metrics:")
    for name, value in metrics.items():
        logger.info(f"{name.capitalize()}: {value:.4f}")
    
    # Save updated model
    save_model(model, model_path)
    logger.info(f"Model retrained and saved to {model_path}")
    
    return model, metrics