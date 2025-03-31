import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from src.model import train_model, evaluate_model
from src.preprocessing import preprocess_data

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'State': ['CA', 'NY', 'TX', 'CA', 'FL', 'WA'],
        'Account_Length': [128, 107, 137, 84, 75, 112],
        'International_Plan': ['no', 'yes', 'no', 'yes', 'no', 'yes'],
        'Voice_Mail_Plan': ['yes', 'no', 'yes', 'no', 'yes', 'no'],
        'Total_Day_Minutes': [265.1, 161.6, 243.4, 299.4, 215.3, 178.9],
        'Customer_Service_Calls': [1, 2, 0, 3, 1, 4],
        'Churn': [0, 1, 0, 1, 0, 1]
    })

def test_train_model(sample_data):
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(sample_data)
    model = train_model(X_train, y_train, preprocessor)
    
    assert isinstance(model, Pipeline)
    assert hasattr(model, 'predict')
    assert hasattr(model, 'predict_proba')

def test_evaluate_model(sample_data):
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(sample_data)
    model = train_model(X_train, y_train, preprocessor)
    
    metrics = evaluate_model(model, X_test, y_test)
    
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 'roc_auc' in metrics
    
    # All metrics should be between 0 and 1
    for value in metrics.values():
        assert 0 <= value <= 1