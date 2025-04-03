# 📡 Telecom Churn Prediction - AI Model & Deployment 🚀

## 📌 Project Overview

This project aims to predict customer churn in the telecom industry using a deep learning model. The trained model is deployed as an API, allowing businesses to integrate it into their systems for real-time predictions.

## 🎯 Features

✅ Machine learning-based churn prediction 📊\
✅ REST API for easy integration 🛠️\
✅ Scalable deployment using Docker & FastAPI ⚡\
✅ Interactive Swagger UI for API testing 🔥\
✅ Frontend for visualization 🌐

## Prediction Process

The prediction process enables users to obtain churn predictions based on input data. The steps involved are:

- User Input: The user submits customer data via the API or frontend.

- Preprocessing: The input data is standardized using the saved scaler.pkl file to match the trained model’s format.

- Model Inference: The request is passed to the deployed model, which makes a prediction.

- Thresholding: The model outputs a probability score, which is converted into a binary classification (churn or not) using a threshold (e.g., 0.5).

- Response: The API returns the prediction result, indicating whether the customer is likely to churn, along with confidence scores.

## Retraining Process

Users can upload new data to retrain the model automatically. The retraining process:

- Reads new data uploaded by users.

- Preprocesses and integrates the new data into the training set.

- Triggers the training script to update the model.

- Saves the new best-performing model and updates the deployment.

## 🛠️ Technologies Used

- **TensorFlow/Keras** - Model Training 🧠
- **Scikit-Learn** - Data Preprocessing 📈
- **FastAPI** - API Backend ⚡
- **Swagger UI** - API Documentation 📝
- **React.js** - Frontend 🌐

## 🌍 Live Links

🔗 Prediction Endpoint: [Prediction Endpoint](https://telecom-api.onrender.com/docs#/default/predict_churn_predict_churn__post)

🔗 Retraining Endpoint: [Retraining Endpoint](https://telecom-api.onrender.com/docs#/default/retrain_model_retrain_model__post)

🔗 Full API Repository: [GitHub - Backend](https://github.com/shyakx/Telecom_API.git)

🔗 Full Frontend Repository: [GitHub - Frontend](https://github.com/shyakx/Predict-Prevent-Customer-Churn-with-AI-Frontend.git)

🔗 Full Website Link: [Website](https://predict-prevent-customer-churn-with-ai-steven-shyakas-projects.vercel.app/?#solution)

🔗 Video Presentation: [YouTube Video]([YOUR_YOUTUBE_URL_HERE](https://youtu.be/MKbAXVvX37w))

## 🚀 How to Use

### 1️⃣ API Usage

Send a POST request to the API with customer data:

```bash
'{
    "account_length": 120,
    "international_plan": 1,
    "voice_mail_plan": 0,
    "total_day_minutes": 300,
    "total_eve_minutes": 250,
    "total_night_minutes": 200,
    "total_intl_minutes": 15,
    "customer_service_calls": 3
}'
```

Response Example:

```json
{
    "churn_probability": 0.85,
    "prediction": "Churn Likely"
}
```

### 2️⃣ Run Locally (Development Mode)



## 📊 Model Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 93%   |
| Precision | 85%   |
| Recall    | 63%   |
| F1-Score  | 73%   |
| ROC-AUC   | 80%   |

## 🤖 Future Improvements

- Improve model performance with hyperparameter tuning 🔧
- Add real-time retraining with feedback loop 🔄
- Enhance frontend UI with better UX 🎨
- Deploy on a scalable cloud platform ☁️

## Contributing
Feel free to contribute by opening issues or submitting pull requests.

## License
This project is open-source under the MIT License.

## Author
Steven SHYAKA
