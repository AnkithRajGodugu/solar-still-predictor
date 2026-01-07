import pandas as pd
import numpy as np
from django.conf import settings
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataHandler:
    """Handler for data operations related to solar still measurements"""
    
    @staticmethod
    def load_csv_data(file_path):
        """Load data from a CSV file"""
        try:
            data = pd.read_csv(file_path)
            return data
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")
    
    @staticmethod
    def save_csv_data(data, file_path):
        """Save data to a CSV file"""
        try:
            data.to_csv(file_path, index=False)
            return True
        except Exception as e:
            raise ValueError(f"Error saving CSV file: {e}")
    
    @staticmethod
    def prepare_data_from_queryset(queryset):
        """Prepare features and targets from a Django queryset"""
        data = []
        for item in queryset:
            data.append({
                'ambient_temperature': item.ambient_temperature,
                'water_temperature': item.water_temperature,
                'glass_temperature': item.glass_temperature,
                'solar_radiation': item.solar_radiation,
                'wind_speed': item.wind_speed if item.wind_speed is not None else 0,
                'humidity': item.humidity if item.humidity is not None else 0,
                'efficiency': item.efficiency,
                'freshwater_quantity': item.freshwater_quantity
            })
        
        df = pd.DataFrame(data)
        
        # Handle missing values
        df = df.fillna(0)
        
        # Separate features and targets
        X = df[['ambient_temperature', 'water_temperature', 'glass_temperature', 
                'solar_radiation', 'wind_speed', 'humidity']]
        y = df[['efficiency', 'freshwater_quantity']]
        
        return X, y
    
    @staticmethod
    def prepare_data(df):
        """Prepare features and targets from a pandas DataFrame"""
        # Handle missing values
        df = df.fillna(0)
        
        # Separate features and targets
        X = df[['ambient_temperature', 'water_temperature', 'glass_temperature', 
                'solar_radiation', 'wind_speed', 'humidity']]
        y = df[['efficiency', 'freshwater_quantity']]
        
        return X.values, y.values
    
    @staticmethod
    def prepare_single_prediction(measurement):
        """Prepare a single measurement for prediction"""
        data = {
            'ambient_temperature': measurement.ambient_temperature,
            'water_temperature': measurement.water_temperature,
            'glass_temperature': measurement.glass_temperature,
            'solar_radiation': measurement.solar_radiation,
            'wind_speed': measurement.wind_speed if measurement.wind_speed is not None else 0,
            'humidity': measurement.humidity if measurement.humidity is not None else 0
        }
        
        return pd.DataFrame([data])
    
    @staticmethod
    def split_data(X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def normalize_data(X_train, X_test):
        """Normalize the input data"""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    @staticmethod
    def generate_sample_data(n_samples=100):
        """Generate synthetic data for testing the model"""
        np.random.seed(42)
        
        # Generate features
        ambient_temp = np.random.uniform(20, 40, n_samples)  # 20-40°C
        water_temp = np.random.uniform(25, 65, n_samples)    # 25-65°C
        glass_temp = np.random.uniform(25, 60, n_samples)    # 25-60°C
        solar_rad = np.random.uniform(400, 1000, n_samples)  # 400-1000 W/m²
        wind_speed = np.random.uniform(0, 5, n_samples)      # 0-5 m/s
        humidity = np.random.uniform(30, 80, n_samples)      # 30-80%
        
        # Create a dataframe
        data = pd.DataFrame({
            'ambient_temperature': ambient_temp,
            'water_temperature': water_temp,
            'glass_temperature': glass_temp,
            'solar_radiation': solar_rad,
            'wind_speed': wind_speed,
            'humidity': humidity
        })
        
        # Simple model for efficiency: mainly affected by temperature differences and solar radiation
        efficiency = (
            0.2 * (water_temp - ambient_temp) / 40 +
            0.3 * (water_temp - glass_temp) / 30 +
            0.4 * solar_rad / 1000 -
            0.1 * wind_speed / 5
        ) * 100  # scaling to percentage
        
        # Bound efficiency between 0-100%
        efficiency = np.clip(efficiency, 0, 100)
        
        # Simple model for freshwater production: mainly affected by efficiency and solar radiation
        freshwater = efficiency * solar_rad / 100 * 0.05  # ml
        
        # Add some noise to make it more realistic
        efficiency += np.random.normal(0, 5, n_samples)
        freshwater += np.random.normal(0, 10, n_samples)
        
        # Ensure non-negative values
        efficiency = np.maximum(efficiency, 0)
        freshwater = np.maximum(freshwater, 0)
        
        # Add to dataframe
        data['efficiency'] = efficiency
        data['freshwater_quantity'] = freshwater
        
        return data 