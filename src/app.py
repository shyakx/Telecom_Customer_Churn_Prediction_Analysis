from flask import Flask, request, jsonify, render_template
import pandas as pd
from model import load_model, predict_churn
from visualization import generate_visualizations
from upload_data import handle_upload
from train import retrain_model
import os

app = Flask(__name__)

# Load models and scaler at startup
model, scaler = load_model()

@app.route('/')
def home():
    return render_template('/frontend/templates/index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Convert to DataFrame with same structure as training data
        features = pd.DataFrame([data])
        
        # Make prediction
        prediction, probability = predict_churn(model, scaler, features)
        
        return render_template('results.html', 
                            prediction=prediction,
                            probability=probability,
                            customer_data=data)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/visualize')
def visualize():
    try:
        # Generate visualization paths
        plot_paths = generate_visualizations()
        return render_template('visualizations.html', plots=plot_paths)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        result = handle_upload(file)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        accuracy_report = retrain_model()
        return jsonify({
            'message': 'Model retrained successfully',
            'accuracy': accuracy_report
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)