import pytest
import pandas as pd
import numpy as np
from src.prediction import predict_churn
from src.model import train_model
from src.preprocessing import preprocess_data

@pytest.fixture
def sample_model(sample_data):
    X_train, _, y_train, _, preprocessor = preprocess_data(sample_data)
    return train_model(X_train, y_train, preprocessor)

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

def test_predict_churn(sample_model, sample_data):
    # Test with DataFrame input
    input_df = sample_data.drop('Churn', axis=1).head(2)
    result = predict_churn(input_df, sample_model)
    
    assert 'predictions' in result
    assert 'probabilities' in result
    assert len(result['predictions']) == 2
    assert len(result['probabilities']) == 2
    
    # Test with dict input
    input_dict = input_df.iloc[0].to_dict()
    result = predict_churn(input_dict, sample_model)
    assert len(result['predictions']) == 1
    assert len(result['probabilities']) == 1