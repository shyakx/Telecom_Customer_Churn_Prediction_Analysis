# Customer Churn Prediction

## Project Overview
This project aims to predict customer churn using machine learning models. It involves data preprocessing, model training (Logistic Regression and Random Forest), evaluation, and deployment. The pipeline is scalable and includes a web application for predictions, retraining, and monitoring.

## Features
- **Data Preprocessing**: Handles missing values, encodes categorical features, and scales numerical features.
- **Model Training & Evaluation**: Compares Logistic Regression and Random Forest, selects the best model based on F1-score.
- **Feature Importance Analysis**: Identifies key factors affecting churn.
- **Web Application**: Allows users to upload data and make predictions.
- **Retraining Mechanism**: Supports model retraining with new data.
- **Cloud Deployment**: Containerized with Docker for scalability.

## Project Structure
```
Customer_Churn_ML/
│
├── README.md               # Project overview and instructions
├── notebook/
│   ├── customer_churn_analysis.ipynb  # Jupyter notebook with EDA & model training
├── src/                     # Source code for the ML pipeline
│   ├── preprocessing.py      # Data preprocessing functions
│   ├── model.py              # Model training and evaluation functions
│   ├── prediction.py         # Model inference functions
│   ├── retraining.py         # Logic to trigger retraining
├── data/                    # Dataset storage
│   ├── train/               # Training data
│   ├── test/                # Testing data
│   ├── telecom_churn.csv    # Raw dataset
├── models/                  # Saved models
│   ├── best_model.pkl       # Serialized best model
├── app/                     # Web application
│   ├── Dockerfile           # Docker containerization setup
│   ├── app.py               # Flask/FastAPI app for prediction
│   ├── templates/           # HTML templates if needed
│   ├── static/              # CSS, JS files
├── tests/                   # Unit tests for ML pipeline
│   ├── test_preprocessing.py
│   ├── test_model.py
│   ├── test_prediction.py
├── deployment/              # Deployment scripts
│   ├── docker-compose.yml   # Multi-container setup
│   ├── cloud_setup.md       # Instructions for cloud deployment
└── requirements.txt         # Dependencies list
```

## Installation & Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/Customer_Churn_ML.git
   cd Customer_Churn_ML
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run Jupyter Notebook for analysis:
   ```sh
   jupyter notebook notebook/customer_churn_analysis.ipynb
   ```
4. Start the web application:
   ```sh
   python app/app.py
   ```
5. To retrain the model, upload new data via the web app and trigger retraining.

## Deployment
- Use Docker to containerize the application:
  ```sh
  docker build -t churn-prediction-app .
  docker run -p 5000:5000 churn-prediction-app
  ```
- Deploy on a cloud platform (AWS/GCP/Azure) as described in `deployment/cloud_setup.md`.

## Usage
- Upload customer data in CSV format.
- Get churn predictions instantly.
- View feature importance insights.
- Retrain the model with new data.

## Contributing
Pull requests are welcome. Ensure your code is well-documented and tested.

## License
This project is licensed under the MIT License.
