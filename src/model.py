import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os

def build_model(input_shape):
    """Create and compile a neural network model."""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(X_train, y_train):
    """Train the model and save the best version."""
    model = build_model(X_train.shape[1])
    
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
    
    # Save the trained model
    model.save("models/best_model.h5")
    return model

def load_model():
    """Load the saved model or raise an error if missing."""
    model_path = "models/best_model.h5"
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        raise FileNotFoundError("Model file not found. Train the model first.")
