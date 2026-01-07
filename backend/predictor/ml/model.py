import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
import joblib
from django.conf import settings

class SolarStillANN:
    """ANN model for solar still performance prediction"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def preprocess_data(self, X, y=None, fit=False):
        """Preprocess the data by scaling"""
        if fit:
            X_scaled = self.scaler_X.fit_transform(X)
            if y is not None:
                y_scaled = self.scaler_y.fit_transform(y)
                return X_scaled, y_scaled
            return X_scaled
        else:
            X_scaled = self.scaler_X.transform(X)
            if y is not None:
                y_scaled = self.scaler_y.transform(y)
                return X_scaled, y_scaled
            return X_scaled
    
    def build(self, input_shape):
        """Build the ANN model architecture"""
        input_dim = input_shape[0]
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=input_dim))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(2, activation='linear'))  # Output: efficiency and freshwater quantity
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                      loss='mean_squared_error', 
                      metrics=['mae'])
        
        self.model = model
        return model
    
    # Alias for backward compatibility
    def build_model(self, input_dim):
        """Alias for build method for backward compatibility"""
        return self.build(input_shape=(input_dim,))
    
    def train(self, X, y, epochs=100, batch_size=32, validation_split=0.2, validation_data=None):
        """Train the model with the provided data"""
        # Preprocess the data
        X_scaled, y_scaled = self.preprocess_data(X, y, fit=True)
        
        # Build the model if not already built
        if self.model is None:
            self.model = self.build(input_shape=(X.shape[1],))
        
        # Prepare validation data if provided
        validation_kwargs = {}
        if validation_data:
            X_val, y_val = validation_data
            X_val_scaled = self.scaler_X.transform(X_val)
            y_val_scaled = self.scaler_y.transform(y_val)
            validation_kwargs['validation_data'] = (X_val_scaled, y_val_scaled)
        else:
            validation_kwargs['validation_split'] = validation_split
        
        # Train the model
        history = self.model.fit(
            X_scaled, y_scaled,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            **validation_kwargs
        )
        
        return history
    
    def evaluate(self, X, y):
        """Evaluate the model performance"""
        if self.model is None:
            raise ValueError("Model not trained or loaded yet")
        
        # Preprocess the data
        X_scaled = self.preprocess_data(X)
        
        # Predict
        y_pred_scaled = self.model.predict(X_scaled)
        
        # Inverse transform the predictions
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        
        # Calculate accuracy as percentage of predictions within 10% of actual
        errors = np.abs(y_pred - y) / np.maximum(np.abs(y), 1e-10)
        accuracy = 100 * np.mean(errors < 0.1)
        
        return mse, accuracy
    
    def predict(self, X):
        """Make predictions for new data"""
        if self.model is None:
            raise ValueError("Model not trained or loaded yet")
        
        # Handle if X is already scaled (from our management command)
        if isinstance(X, np.ndarray) and X.shape[1] == self.model.input_shape[1]:
            # Just predict without scaling again
            y_pred_scaled = self.model.predict(X)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
            return y_pred
            
        # Otherwise, preprocess the data
        X_scaled = self.preprocess_data(X)
        
        # Predict
        y_pred_scaled = self.model.predict(X_scaled)
        
        # Inverse transform the predictions
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        
        return y_pred
    
    def save(self, model_name):
        """Save the model and scalers"""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Create models directory if it doesn't exist
        model_dir = os.path.join(settings.MEDIA_ROOT, 'trained_models')
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the model
        model_path = os.path.join(model_dir, f"{model_name}.h5")
        self.model.save(model_path)
        
        # Save the scalers
        joblib.dump(self.scaler_X, os.path.join(model_dir, 'scaler_X.pkl'))
        joblib.dump(self.scaler_y, os.path.join(model_dir, 'scaler_y.pkl'))
        
        self.model_path = model_path
        return model_path
    
    # Alias for backward compatibility
    def save_model(self, model_path):
        """Alias for save method for backward compatibility"""
        model_name = os.path.basename(model_path).replace('.h5', '')
        return self.save(model_name)
    
    def load(self, model_path):
        """Load the model and scalers"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load the model
        self.model = load_model(model_path)
        
        # Load the scalers
        scaler_dir = os.path.dirname(model_path)
        scaler_X_path = os.path.join(scaler_dir, 'scaler_X.pkl')
        scaler_y_path = os.path.join(scaler_dir, 'scaler_y.pkl')
        
        if os.path.exists(scaler_X_path):
            self.scaler_X = joblib.load(scaler_X_path)
        
        if os.path.exists(scaler_y_path):
            self.scaler_y = joblib.load(scaler_y_path)
        
        self.model_path = model_path
        
    # Alias for backward compatibility
    def load_model(self, model_path):
        """Alias for load method for backward compatibility"""
        return self.load(model_path) 