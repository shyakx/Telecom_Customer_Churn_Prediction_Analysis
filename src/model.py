from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, 
                            recall_score, f1_score, roc_auc_score)
from .preprocessing import preprocess_data, load_data
import joblib

def train_model(X_train, y_train, preprocessor, model_params=None):
    """Train a Random Forest model with preprocessing"""
    if model_params is None:
        model_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'class_weight': 'balanced'
        }
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(**model_params))
    ])
    
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    return metrics

def save_model(model, filepath):
    """Save the trained model"""
    joblib.dump(model, filepath)

def load_model(filepath):
    """Load a saved model"""
    return joblib.load(filepath)

def run_training_pipeline(data_path='../data/telecom_churn.csv', 
                         model_path='../models/best_model.pkl'):
    """Run the complete training pipeline"""
    # Load and preprocess data
    df = load_data(data_path)
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)
    
    # Train model
    model = train_model(X_train, y_train, preprocessor)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    print("Model Evaluation Metrics:")
    for name, value in metrics.items():
        print(f"{name.capitalize()}: {value:.4f}")
    
    # Save model
    save_model(model, model_path)
    print(f"Model saved to {model_path}")
    
    return model, metrics

if __name__ == "__main__":
    run_training_pipeline()