"""
Management command to run a demo of the solar still predictor model with mock data.
"""
import os
import numpy as np
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone
from backend.predictor.ml.data_handler import DataHandler
from backend.predictor.ml.model import SolarStillANN
from backend.predictor.models import TrainedModel, SolarStillMeasurement, PredictionResult
import shutil

class Command(BaseCommand):
    help = 'Run the solar still prediction model with mock data'

    def add_arguments(self, parser):
        parser.add_argument(
            '--samples',
            type=int,
            default=100,
            help='Number of mock data samples to generate'
        )
        parser.add_argument(
            '--epochs',
            type=int,
            default=50,
            help='Number of training epochs'
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=32,
            help='Batch size for training'
        )

    def handle(self, *args, **options):
        n_samples = options['samples']
        epochs = options['epochs']
        batch_size = options['batch_size']
        
        self.stdout.write(f"Generating {n_samples} mock data samples...")
        
        # Generate mock data
        data = DataHandler.generate_sample_data(n_samples=n_samples)
        
        # Save mock data to CSV in media directory
        media_dir = os.path.join(settings.BASE_DIR, 'media')
        os.makedirs(media_dir, exist_ok=True)
        csv_path = os.path.join(media_dir, 'mock_solar_still_data.csv')
        data.to_csv(csv_path, index=False)
        self.stdout.write(self.style.SUCCESS(f"Mock data saved to {csv_path}"))
        
        # Prepare data for model
        X, y = DataHandler.prepare_data(data)
        
        # Split data
        X_train, X_test, y_train, y_test = DataHandler.split_data(X, y, test_size=0.2)
        
        # Normalize data
        X_train_scaled, X_test_scaled = DataHandler.normalize_data(X_train, X_test)
        
        # Build and train model
        self.stdout.write("Training model...")
        model = SolarStillANN()
        model.build(input_shape=(X_train_scaled.shape[1],))
        
        history = model.train(
            X_train_scaled, 
            y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_data=(X_test_scaled, y_test)
        )
        
        # Evaluate model
        mse, accuracy = model.evaluate(X_test_scaled, y_test)
        self.stdout.write(self.style.SUCCESS(f"Model training completed with MSE: {mse:.4f}, Accuracy: {accuracy:.2f}%"))
        
        # Save the model
        model_name = f"MockData_ANN_{timezone.now().strftime('%Y%m%d_%H%M%S')}"
        model_path = model.save(model_name)
        
        # Get the model filename
        model_filename = os.path.basename(model_path)
        
        # Ensure media directory exists
        media_dir = os.path.join(settings.MEDIA_ROOT, 'trained_models')
        os.makedirs(media_dir, exist_ok=True)
        
        # Copy the model file to media directory
        media_model_path = os.path.join(media_dir, model_filename)
        shutil.copy2(model_path, media_model_path)
        
        # Create TrainedModel record with relative path for model_file
        trained_model = TrainedModel.objects.create(
            name=model_name,
            model_file=f'trained_models/{model_filename}',
            mse=mse,
            accuracy=accuracy,
            is_active=True,
            description="Model trained on mock data via management command"
        )
        self.stdout.write(self.style.SUCCESS(f"Model saved as '{model_name}'"))
        
        # Make sample predictions with the first 5 test data points
        self.stdout.write("\nSample predictions:")
        sample_inputs = X_test_scaled[:5]
        predictions = model.predict(sample_inputs)
        
        # Compare with actual values
        self.stdout.write("-" * 80)
        self.stdout.write(f"{'Predicted Efficiency':>20} {'Actual Efficiency':>20} {'Predicted Freshwater':>20} {'Actual Freshwater':>20}")
        self.stdout.write("-" * 80)
        
        # Use min to avoid index errors
        num_samples = min(len(predictions), 5)
        for i in range(num_samples):
            pred_eff, pred_fresh = predictions[i]
            actual_eff, actual_fresh = y_test[i]
            
            self.stdout.write(f"{pred_eff:>20.2f} {actual_eff:>20.2f} {pred_fresh:>20.2f} {actual_fresh:>20.2f}")
        
        # Optionally save test data as SolarStillMeasurement records
        self.stdout.write("\nSaving 10 sample measurements to database...")
        num_measurements = min(len(X_test), 10)
        for i in range(num_measurements):
            # Get original (non-normalized) data
            ambient_temp = X_test[i][0]
            water_temp = X_test[i][1]
            glass_temp = X_test[i][2]
            solar_rad = X_test[i][3]
            wind_speed = X_test[i][4]
            humidity = X_test[i][5]
            efficiency = y_test[i][0]
            freshwater = y_test[i][1]
            
            # Create measurement record
            measurement = SolarStillMeasurement.objects.create(
                date=timezone.now().date(),
                time=timezone.now().time(),
                ambient_temperature=ambient_temp,
                water_temperature=water_temp,
                glass_temperature=glass_temp,
                solar_radiation=solar_rad,
                wind_speed=wind_speed,
                humidity=humidity,
                efficiency=efficiency,
                freshwater_quantity=freshwater
            )
            
            # Make prediction for this measurement
            input_data = np.array([[
                ambient_temp, water_temp, glass_temp, 
                solar_rad, wind_speed, humidity
            ]])
            input_scaled = X_test_scaled[i].reshape(1, -1)
            pred_eff, pred_fresh = model.predict(input_scaled)[0]
            
            # Save prediction result
            PredictionResult.objects.create(
                measurement=measurement,
                model=trained_model,
                predicted_efficiency=pred_eff,
                predicted_freshwater=pred_fresh
            )
        
        self.stdout.write(self.style.SUCCESS(f"Saved 10 measurements and predictions to database"))
        self.stdout.write(self.style.SUCCESS(f"\nDemo completed successfully. You can now explore the data in the admin interface or dashboard.")) 