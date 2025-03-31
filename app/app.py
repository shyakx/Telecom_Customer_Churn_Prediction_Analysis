from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import pandas as pd
from src.prediction import predict_churn, batch_predict
from src.retraining import retrain_model, check_retraining_need
import logging

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        if request.is_json:
            data = request.get_json()
            result = predict_churn(data)
            return jsonify(result)
        elif request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process CSV file
            output_file = os.path.join(app.config['UPLOAD_FOLDER'], f"predicted_{filename}")
            result = batch_predict(filepath, output_file)
            
            return jsonify({
                'message': 'Batch prediction complete',
                'output_file': output_file
            })
        else:
            return jsonify({'error': 'Unsupported content type'}), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Check if retraining is needed
        if check_retraining_need(filepath, '../models/best_model.pkl'):
            model, metrics = retrain_model(filepath, '../models/best_model.pkl')
            return jsonify({
                'message': 'Model retrained successfully',
                'metrics': metrics
            })
        else:
            return jsonify({
                'message': 'Retraining not needed based on current performance'
            })
    except Exception as e:
        logger.error(f"Retraining error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)