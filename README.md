# Customer Churn Prediction

## Overview
This project aims to predict customer churn using machine learning models. The goal is to help businesses retain customers by identifying potential churners early. The project follows a structured pipeline, from data preprocessing to model training, evaluation, deployment, and retraining.

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

## Dataset
The dataset consists of customer details, usage patterns, and churn status. Key features include:
- **State:** The location of the customer.
- **International Plan:** Whether the customer has an international calling plan.
- **Voice Mail Plan:** Subscription to voicemail services.
- **Total Day Minutes:** The total number of minutes spent on calls during the day.
- **Customer Service Calls:** The number of calls made to customer service.
- **Churn (Target):** Whether the customer has left the service (1) or not (0).

## Model Training and Evaluation
Two models were trained: **Logistic Regression** and **Random Forest**, with **Random Forest** being selected as the final model due to superior performance.

### Logistic Regression
- **Accuracy:** 85.6%
- **Precision:** 58.6%
- **Recall:** 16.8%
- **F1 Score:** 26.2%
- **ROC AUC:** 57.4%

### Random Forest (Best Model)
- **Accuracy:** 94.9%
- **Precision:** 93.5%
- **Recall:** 71.3%
- **F1 Score:** 80.9%
- **ROC AUC:** 85.2%

The **Random Forest** model significantly outperforms Logistic Regression and has been selected as the final model for deployment.

## Features
- **Model Prediction:** Allows users to input data and get a churn prediction.
- **Visualizations:** Provides insights into important features affecting churn.
- **Data Upload:** Supports bulk data uploads for retraining.
- **Retraining Trigger:** Users can trigger model retraining with new data.
- **Performance Testing:** Simulates high-traffic scenarios using Locust.

## Installation and Setup
### Prerequisites
Ensure you have Python 3.8+ installed along with pip and virtual environment tools.

1. Clone the repository:
   ```sh
   git clone https://github.com/shyakx/Telecom_Customer_Churn_Prediction_Analysis.git
   cd Telecom_Customer_Churn_Prediction_Analysis
   ```
2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run the model training pipeline:
   ```sh
   python src/model.py
   ```
5. Start the web app (Flask/FastAPI):
   ```sh
   python app/app.py
   ```

## Deployment
The model is containerized using Docker and deployed on a cloud platform. To deploy:
```sh
docker-compose up --build
```

## Testing and Performance Evaluation
To simulate multiple concurrent requests and measure response times:
1. Install Locust:
   ```sh
   pip install locust
   ```
2. Run the Locust test:
   ```sh
   locust -f tests/load_test.py
   ```
3. Access the Locust web interface at `http://localhost:8089` to configure and start the test.

## Retraining Process
Users can upload new data to retrain the model automatically. The retraining process:
1. Reads new data uploaded by users.
2. Preprocesses and integrates the new data into the training set.
3. Triggers the training script to update the model.
4. Saves the new best-performing model and updates the deployment.

## Next Steps
- Fine-tune the Random Forest model further.
- Implement real-time monitoring and logging.
- Expand model explainability features.
- Improve the web UI/UX for better usability.

## Contributing
Feel free to contribute by opening issues or submitting pull requests.

## License
This project is open-source under the MIT License.

## Author
Steven SHYAKA
