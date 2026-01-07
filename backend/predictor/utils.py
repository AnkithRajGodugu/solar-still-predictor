import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import io
import base64
from django.conf import settings
from .models import SolarStillMeasurement, PredictionResult, TrainedModel
from .ml.data_handler import DataHandler
from .ml.model import SolarStillANN
from datetime import datetime

def train_ann_model(queryset=None, epochs=100, batch_size=32):
    """Train a new ANN model with the dataset"""
    # Create a data handler
    if queryset is None or queryset.count() < 10:
        # Generate sample data if no data is available
        data = DataHandler.generate_sample_data(n_samples=100)
        X = data[['ambient_temperature', 'water_temperature', 'glass_temperature', 
                'solar_radiation', 'wind_speed', 'humidity']]
        y = data[['efficiency', 'freshwater_quantity']]
    else:
        # Prepare data from queryset
        X, y = DataHandler.prepare_data_from_queryset(queryset)
    
    # Split the data
    X_train, X_test, y_train, y_test = DataHandler.split_data(X, y)
    
    # Create and train the model
    ann = SolarStillANN()
    history = ann.train(
        X_train, 
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2
    )
    
    # Evaluate the model
    mse, accuracy = ann.evaluate(X_test, y_test)
    
    # Save the model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f'solar_still_model_{timestamp}'
    model_path = ann.save(model_name)
    
    return {
        'model_path': model_path,
        'mse': float(mse),
        'accuracy': float(accuracy),
        'history': history.history if hasattr(history, 'history') else history
    }

def predict_efficiency(model_path, measurement):
    """Predict efficiency and freshwater quantity for a measurement"""
    try:
        # Check if model_path is a TrainedModel instance
        if isinstance(model_path, TrainedModel):
            model_path = model_path.model_file.path
            
        # Ensure the model file exists
        if not os.path.exists(model_path):
            raise ValueError(f"Model file not found at: {model_path}")
            
        # Load the model
        ann = SolarStillANN(model_path)
        
        # Prepare the data
        X = DataHandler.prepare_single_prediction(measurement)
        
        # Make prediction
        predictions = ann.predict(X)
        
        # Extract predictions (shape: [1, 2] - first is efficiency, second is freshwater)
        efficiency = float(predictions[0, 0])
        freshwater = float(predictions[0, 1])
        
        return efficiency, freshwater
    except Exception as e:
        raise ValueError(f"Prediction error: {e}")

def get_latest_model():
    """Get the latest trained model"""
    try:
        # First try to get the active model
        active_model = TrainedModel.objects.filter(is_active=True).first()
        if active_model:
            return active_model
        
        # If no active model, get the latest created model
        return TrainedModel.objects.order_by('-created_at').first()
    except TrainedModel.DoesNotExist:
        return None

def generate_performance_chart(queryset, predictions=None):
    """Generate a performance chart comparing actual vs predicted values"""
    # Prepare data
    dates = []
    actual_efficiency = []
    actual_freshwater = []
    predicted_efficiency = []
    predicted_freshwater = []
    
    # Get predictions for all measurements
    if predictions is None:
        predictions = {}
        for measurement in queryset:
            pred_result = PredictionResult.objects.filter(measurement=measurement).first()
            if pred_result:
                predictions[measurement.id] = {
                    'efficiency': pred_result.predicted_efficiency,
                    'freshwater': pred_result.predicted_freshwater
                }
    
    # Sort queryset by date and time
    sorted_queryset = sorted(queryset, key=lambda x: (x.date, x.time))
    
    for item in sorted_queryset:
        # Format date and time for better readability
        date_str = f"{item.date.strftime('%Y-%m-%d')} {item.time.strftime('%H:%M')}"
        dates.append(date_str)
        
        # Handle actual values
        actual_efficiency.append(float(item.efficiency) if item.efficiency is not None else None)
        actual_freshwater.append(float(item.freshwater_quantity) if item.freshwater_quantity is not None else None)
        
        # Get predictions if available
        if item.id in predictions:
            pred = predictions[item.id]
            predicted_efficiency.append(float(pred['efficiency']))
            predicted_freshwater.append(float(pred['freshwater']))
        else:
            predicted_efficiency.append(None)
            predicted_freshwater.append(None)
    
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot efficiency
    ax1.plot(dates, actual_efficiency, 'b-', label='Actual Efficiency', marker='o')
    if any(pred is not None for pred in predicted_efficiency):
        ax1.plot(dates, predicted_efficiency, 'r--', label='Predicted Efficiency', marker='s')
    ax1.set_xlabel('Date and Time')
    ax1.set_ylabel('Efficiency (%)')
    ax1.set_title('Solar Still Efficiency Over Time')
    ax1.legend()
    ax1.grid(True)
    
    # Rotate x-axis labels for better readability
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot freshwater quantity
    ax2.plot(dates, actual_freshwater, 'g-', label='Actual Freshwater', marker='o')
    if any(pred is not None for pred in predicted_freshwater):
        ax2.plot(dates, predicted_freshwater, 'r--', label='Predicted Freshwater', marker='s')
    ax2.set_xlabel('Date and Time')
    ax2.set_ylabel('Freshwater (ml)')
    ax2.set_title('Freshwater Production Over Time')
    ax2.legend()
    ax2.grid(True)
    
    # Rotate x-axis labels for better readability
    ax2.tick_params(axis='x', rotation=45)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Convert plot to base64 string for displaying in HTML
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    image_base64 = base64.b64encode(image_png).decode('utf-8')
    return f"data:image/png;base64,{image_base64}" 