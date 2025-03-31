import pytest
import pandas as pd
import numpy as np
from src.preprocessing import load_data, preprocess_data

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'State': ['CA', 'NY', 'TX', 'CA'],
        'Account_Length': [128, 107, 137, 84],
        'International_Plan': ['no', 'yes', 'no', 'yes'],
        'Voice_Mail_Plan': ['yes', 'no', 'yes', 'no'],
        'Total_Day_Minutes': [265.1, 161.6, 243.4, 299.4],
        'Customer_Service_Calls': [1, 2, 0, 3],
        'Churn': [0, 1, 0, 1]
    })

def test_load_data(tmp_path):
    # Create a temporary CSV file
    data = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    file_path = tmp_path / "test.csv"
    data.to_csv(file_path, index=False)
    
    # Test loading
    loaded = load_data(file_path)
    pd.testing.assert_frame_equal(loaded, data)

def test_preprocess_data(sample_data):
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(sample_data)
    
    # Check shapes
    assert X_train.shape[0] == 3  # 75% of 4 samples
    assert X_test.shape[0] == 1   # 25% of 4 samples
    assert len(y_train) == 3
    assert len(y_test) == 1
    
    # Check preprocessor is fitted
    assert hasattr(preprocessor, 'transformers_')