import os
from train import retrain_model
from data_preprocessing import preprocess_data
import pandas as pd

def initial_training():
    # Create all required directories
    base_dir = os.path.dirname(os.path.dirname(__file__))
    os.makedirs(os.path.join(base_dir, 'data/raw'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'data/uploaded'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'models'), exist_ok=True)
    
    # Load the original data
    data_path = os.path.join(base_dir, 'data/raw/telecom_churn.csv')
    data = pd.read_csv(data_path)
    
    # Preprocess and train
    print("Starting initial model training...")
    result = retrain_model()
    
    print("\nTraining completed!")
    print(f"Model accuracy: {result['accuracy']:.2f}")
    print("Model saved to models/classifier.pkl")
    print("Scaler saved to models/scaler.pkl")

if __name__ == '__main__':
    initial_training()