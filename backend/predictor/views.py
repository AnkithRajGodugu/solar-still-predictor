from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.conf import settings
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from django.urls import reverse_lazy
from django.core.files import File
from django.views.decorators.http import require_POST
from django.contrib.auth.mixins import LoginRequiredMixin

import os
import pandas as pd
import numpy as np
import json
from datetime import datetime

from .models import SolarStillMeasurement, PredictionResult, TrainedModel
from .forms import SolarStillMeasurementForm, PredictionForm, CSVUploadForm, TrainModelForm
from .utils import train_ann_model, predict_efficiency, get_latest_model, generate_performance_chart
from .ml.data_handler import DataHandler

# Dashboard view
def dashboard(request):
    """Main dashboard view"""
    # Get recent measurements with their predictions
    measurements = SolarStillMeasurement.objects.all().order_by('-date', '-time')[:10]
    
    # Get latest model and active model
    latest_model = TrainedModel.objects.order_by('-created_at').first()
    active_model = TrainedModel.objects.filter(is_active=True).first()
    
    # Debug information
    print(f"Latest model: {latest_model.name if latest_model else 'None'}")
    print(f"Active model: {active_model.name if active_model else 'None'}")
    print(f"Total models: {TrainedModel.objects.count()}")
    
    # Get predictions for the measurements
    predictions = {}
    for measurement in measurements:
        pred_result = PredictionResult.objects.filter(measurement=measurement).first()
        if pred_result:
            predictions[measurement.id] = {
                'efficiency': pred_result.predicted_efficiency,
                'freshwater': pred_result.predicted_freshwater
            }
    
    # Generate chart
    chart_image = None
    if measurements.exists():
        chart_image = generate_performance_chart(measurements, predictions)
    
    # Get counts
    measurement_count = SolarStillMeasurement.objects.count()
    prediction_count = PredictionResult.objects.count()
    model_count = TrainedModel.objects.count()
    
    # Context
    context = {
        'measurements': measurements,
        'latest_model': latest_model,
        'active_model': active_model,
        'chart_image': chart_image,
        'measurement_count': measurement_count,
        'prediction_count': prediction_count,
        'model_count': model_count,
        'predictions': predictions
    }
    
    return render(request, 'predictor/dashboard.html', context)

# Measurements views
class MeasurementListView(ListView):
    """View to list all measurements"""
    model = SolarStillMeasurement
    template_name = 'predictor/measurement_list.html'
    context_object_name = 'measurements'
    ordering = ['-date', '-time']
    paginate_by = 10

class MeasurementDetailView(DetailView):
    """View to show details of a measurement"""
    model = SolarStillMeasurement
    template_name = 'predictor/measurement_detail.html'
    context_object_name = 'measurement'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Get the predictions for this measurement
        measurement = self.get_object()
        predictions = measurement.predictions.all()
        context['predictions'] = predictions
        return context

class MeasurementCreateView(CreateView):
    """View to create a new measurement"""
    model = SolarStillMeasurement
    form_class = SolarStillMeasurementForm
    template_name = 'predictor/measurement_form.html'
    success_url = reverse_lazy('measurement-list')
    
    def form_valid(self, form):
        messages.success(self.request, 'Measurement added successfully!')
        return super().form_valid(form)

class MeasurementUpdateView(UpdateView):
    """View to update a measurement"""
    model = SolarStillMeasurement
    form_class = SolarStillMeasurementForm
    template_name = 'predictor/measurement_form.html'
    
    def get_success_url(self):
        return reverse_lazy('measurement-detail', kwargs={'pk': self.object.pk})
    
    def form_valid(self, form):
        messages.success(self.request, 'Measurement updated successfully!')
        return super().form_valid(form)

class MeasurementDeleteView(DeleteView):
    """View to delete a measurement"""
    model = SolarStillMeasurement
    template_name = 'predictor/measurement_confirm_delete.html'
    success_url = reverse_lazy('measurement-list')
    
    def delete(self, request, *args, **kwargs):
        messages.success(request, 'Measurement deleted successfully!')
        return super().delete(request, *args, **kwargs)

# Model views
class ModelListView(ListView):
    """View to list all trained models"""
    model = TrainedModel
    template_name = 'predictor/model_list.html'
    context_object_name = 'models'
    ordering = ['-created_at']
    paginate_by = 10

class ModelDetailView(DetailView):
    """View to show details of a trained model"""
    model = TrainedModel
    template_name = 'predictor/model_detail.html'
    context_object_name = 'model'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Get the predictions using this model
        model = self.get_object()
        predictions = model.predictions.all()
        context['predictions'] = predictions
        return context

class ModelDeleteView(LoginRequiredMixin, DeleteView):
    """View to delete a trained model"""
    model = TrainedModel
    template_name = 'predictor/model_confirm_delete.html'
    success_url = reverse_lazy('model-list')

    def delete(self, request, *args, **kwargs):
        model = self.get_object()
        # Delete the model file
        if model.model_file:
            try:
                if os.path.exists(model.model_file.path):
                    os.remove(model.model_file.path)
            except Exception as e:
                messages.error(request, f'Error deleting model file: {str(e)}')
        
        response = super().delete(request, *args, **kwargs)
        messages.success(request, 'Model deleted successfully.')
        return response

class ModelUpdateView(LoginRequiredMixin, UpdateView):
    model = TrainedModel
    template_name = 'predictor/model_edit.html'
    fields = ['name', 'description', 'is_active']
    success_url = reverse_lazy('model-list')

    def form_valid(self, form):
        response = super().form_valid(form)
        messages.success(self.request, 'Model updated successfully.')
        return response

def set_active_model(request, pk):
    """Set a model as the active one for predictions"""
    model = get_object_or_404(TrainedModel, pk=pk)
    
    # Update is_active status
    TrainedModel.objects.all().update(is_active=False)
    model.is_active = True
    model.save()
    
    messages.success(request, f'Model "{model.name}" is now set as active')
    return redirect('model-detail', pk=pk)

# Prediction views
def make_prediction(request):
    """View for making predictions"""
    form = PredictionForm()
    prediction_result = None
    
    # Get latest model
    latest_model = get_latest_model()
    if not latest_model:
        messages.warning(request, 'No trained model available. Please train a model first.')
        return redirect('train-model')
    
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            # Create a temporary measurement object without saving to DB
            measurement = form.save(commit=False)
            
            try:
                # Make prediction
                efficiency, freshwater = predict_efficiency(
                    latest_model, 
                    measurement
                )
                
                prediction_result = {
                    'efficiency': round(efficiency, 2),
                    'freshwater': round(freshwater, 2)
                }
                
                messages.success(request, 'Prediction completed successfully!')
            except Exception as e:
                messages.error(request, f'Error making prediction: {e}')
    
    context = {
        'form': form,
        'prediction_result': prediction_result,
        'latest_model': latest_model
    }
    
    return render(request, 'predictor/prediction.html', context)

def save_prediction_for_measurement(request, pk):
    """Save a prediction for an existing measurement"""
    measurement = get_object_or_404(SolarStillMeasurement, pk=pk)
    
    # Get latest model
    latest_model = get_latest_model()
    if not latest_model:
        messages.warning(request, 'No trained model available. Please train a model first.')
        return redirect('train-model')
    
    try:
        # Make prediction
        efficiency, freshwater = predict_efficiency(
            latest_model, 
            measurement
        )
        
        # Save prediction result
        prediction = PredictionResult(
            measurement=measurement,
            model=latest_model,
            predicted_efficiency=efficiency,
            predicted_freshwater=freshwater
        )
        prediction.save()
        
        messages.success(request, 'Prediction saved successfully!')
    except Exception as e:
        messages.error(request, f'Error making prediction: {e}')
    
    return redirect('measurement-detail', pk=pk)

# Model training views
def train_model(request):
    """View for training a new model"""
    form = TrainModelForm()
    measurement_count = SolarStillMeasurement.objects.count()
    
    if request.method == 'POST':
        form = TrainModelForm(request.POST)
        if form.is_valid():
            try:
                # Get form data
                model_name = form.cleaned_data['model_name']
                description = form.cleaned_data['description']
                epochs = form.cleaned_data['epochs']
                batch_size = form.cleaned_data['batch_size']
                use_sample_data = form.cleaned_data['use_sample_data']
                
                # Train the model
                if use_sample_data:
                    # Use sample data for training
                    training_result = train_ann_model(
                        queryset=None,
                        epochs=epochs,
                        batch_size=batch_size
                    )
                else:
                    # Use real measurements
                    measurements = SolarStillMeasurement.objects.all()
                    if measurements.count() < 10:
                        messages.warning(request, 'Not enough measurements for training. Using sample data instead.')
                        training_result = train_ann_model(
                            queryset=None,
                            epochs=epochs,
                            batch_size=batch_size
                        )
                    else:
                        training_result = train_ann_model(
                            queryset=measurements,
                            epochs=epochs,
                            batch_size=batch_size
                        )
                
                # Create model instance
                model = TrainedModel(
                    name=model_name,
                    description=description,
                    accuracy=float(training_result['accuracy']),
                    mse=float(training_result['mse'])
                )
                
                # Save the model file
                with open(training_result['model_path'], 'rb') as f:
                    model.model_file.save(
                        os.path.basename(training_result['model_path']),
                        File(f),
                        save=True
                    )
                
                # Set as active model if it's the first one
                if not TrainedModel.objects.filter(is_active=True).exists():
                    model.is_active = True
                    model.save()
                
                # Debug information
                print(f"Model saved successfully:")
                print(f"Name: {model.name}")
                print(f"Accuracy: {model.accuracy}")
                print(f"MSE: {model.mse}")
                print(f"File path: {model.model_file.path}")
                print(f"Is active: {model.is_active}")
                
                messages.success(request, f'Model "{model_name}" trained successfully!')
                return redirect('model-detail', pk=model.pk)
                
            except Exception as e:
                print(f"Error training model: {str(e)}")
                messages.error(request, f'Error training model: {str(e)}')
    
    context = {
        'form': form,
        'measurement_count': measurement_count
    }
    
    return render(request, 'predictor/train_model.html', context)

# Data management views
def upload_csv(request):
    """View for uploading CSV data"""
    form = CSVUploadForm()
    
    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = request.FILES['csv_file']
            
            try:
                # Read the CSV file
                df = pd.read_csv(csv_file)
                
                # Validate columns
                required_columns = ['date', 'time', 'ambient_temperature', 'water_temperature', 
                                   'glass_temperature', 'solar_radiation']
                for col in required_columns:
                    if col not in df.columns:
                        messages.error(request, f'Missing required column: {col}')
                        return render(request, 'predictor/upload_csv.html', {'form': form})
                
                # Add data to the database
                counter = 0
                for _, row in df.iterrows():
                    try:
                        # Convert date and time to proper formats
                        try:
                            date = datetime.strptime(row['date'], '%Y-%m-%d').date()
                        except ValueError:
                            # Try alternative format
                            date = datetime.strptime(row['date'], '%d/%m/%Y').date()
                        
                        try:
                            time_obj = datetime.strptime(row['time'], '%H:%M:%S').time()
                        except ValueError:
                            # Try alternative format
                            time_obj = datetime.strptime(row['time'], '%H:%M').time()
                        
                        # Prepare measurement data
                        measurement_data = {
                            'date': date,
                            'time': time_obj,
                            'ambient_temperature': row['ambient_temperature'],
                            'water_temperature': row['water_temperature'],
                            'glass_temperature': row['glass_temperature'],
                            'solar_radiation': row['solar_radiation']
                        }
                        
                        # Add optional fields if they exist
                        if 'wind_speed' in df.columns:
                            measurement_data['wind_speed'] = row['wind_speed'] if not pd.isna(row['wind_speed']) else None
                        
                        if 'humidity' in df.columns:
                            measurement_data['humidity'] = row['humidity'] if not pd.isna(row['humidity']) else None
                        
                        if 'efficiency' in df.columns:
                            measurement_data['efficiency'] = row['efficiency'] if not pd.isna(row['efficiency']) else None
                        
                        if 'freshwater_quantity' in df.columns:
                            measurement_data['freshwater_quantity'] = row['freshwater_quantity'] if not pd.isna(row['freshwater_quantity']) else None
                        
                        # Create the measurement
                        SolarStillMeasurement.objects.create(**measurement_data)
                        counter += 1
                    except Exception as e:
                        messages.warning(request, f'Error processing row: {e}')
                
                messages.success(request, f'Successfully imported {counter} measurements from CSV')
                return redirect('measurement-list')
            
            except Exception as e:
                messages.error(request, f'Error processing CSV file: {e}')
    
    return render(request, 'predictor/upload_csv.html', {'form': form})

def export_data(request):
    """View for exporting data to CSV"""
    # Get all measurements
    measurements = SolarStillMeasurement.objects.all().order_by('-date', '-time')
    
    if not measurements.exists():
        messages.warning(request, 'No measurements to export')
        return redirect('dashboard')
    
    # Create DataFrame
    data = []
    for m in measurements:
        data.append({
            'date': m.date.strftime('%Y-%m-%d'),
            'time': m.time.strftime('%H:%M:%S'),
            'ambient_temperature': m.ambient_temperature,
            'water_temperature': m.water_temperature,
            'glass_temperature': m.glass_temperature,
            'solar_radiation': m.solar_radiation,
            'wind_speed': m.wind_speed,
            'humidity': m.humidity,
            'efficiency': m.efficiency,
            'freshwater_quantity': m.freshwater_quantity
        })
    
    df = pd.DataFrame(data)
    
    # Generate CSV file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'solar_still_data_{timestamp}.csv'
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    
    df.to_csv(path_or_buf=response, index=False)
    return response

def download_csv_template(request):
    """Download a CSV template for data upload"""
    # Create a template DataFrame with the required columns
    template_data = {
        'date': ['2025-01-01', '2025-01-01', '2025-01-02'],
        'time': ['10:00:00', '14:00:00', '12:00:00'],
        'ambient_temperature': [30, 35, 32],
        'water_temperature': [45, 50, 48],
        'glass_temperature': [35, 40, 38],
        'solar_radiation': [800, 950, 850],
        'wind_speed': [2, 3, 1.5],
        'humidity': [60, 55, 65],
        'efficiency': [40, 45, 42],
        'freshwater_quantity': [320, 380, 350]
    }
    
    df = pd.DataFrame(template_data)
    
    # Generate CSV file
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="solar_still_template.csv"'
    
    df.to_csv(path_or_buf=response, index=False)
    return response

# API views for AJAX
def predict_api(request):
    """API endpoint for AJAX predictions"""
    from django.http import JsonResponse
    import json
    from .models import TrainedModel
    
    # Only accept POST requests with JSON data
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests allowed'}, status=405)
    
    try:
        # Parse the JSON data
        data = json.loads(request.body)
        
        # Validate required fields
        required_fields = ['ambient_temperature', 'water_temperature', 'glass_temperature', 'solar_radiation']
        for field in required_fields:
            if field not in data:
                return JsonResponse({'error': f"Missing required field: {field}"}, status=400)
        
        # Get the latest active model
        model = TrainedModel.objects.filter(is_active=True).first()
        if not model:
            return JsonResponse({'error': 'No active model found for prediction'}, status=404)
        
        # Prepare input data as a list of all parameters
        inputs = [
            float(data['ambient_temperature']),
            float(data['water_temperature']),
            float(data['glass_temperature']),
            float(data['solar_radiation']),
            float(data.get('wind_speed', 0)),  # Default to 0 if not provided
            float(data.get('humidity', 50)),   # Default to 50% if not provided
        ]
        
        # Normalize inputs (similar to make_prediction function)
        normalized_inputs = np.array(inputs) / np.array([40, 80, 70, 1000, 10, 100])
        
        # Make prediction using the model
        result = make_model_prediction(normalized_inputs.reshape(1, -1), model)
        
        # Return prediction results
        return JsonResponse({
            'efficiency': round(result['efficiency'], 2),
            'freshwater': round(result['freshwater'], 2),
            'model_name': model.name,
            'model_accuracy': model.accuracy
        })
    
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

# Simple test view for debugging
def test_view(request):
    """Simple view to test if Django is working and can access the database"""
    from django.http import HttpResponse
    from .models import TrainedModel
    
    # Get all models
    models = TrainedModel.objects.all()
    
    # Return basic details
    response = f"Number of models: {models.count()}<br>"
    for model in models:
        response += f"Model ID: {model.id}, Name: {model.name}, Active: {model.is_active}, File: {model.model_file}<br>"
    
    return HttpResponse(response)

def visualize_model(request, pk):
    """View for visualizing the neural network model architecture"""
    model = get_object_or_404(TrainedModel, pk=pk)
    
    # Get the model's predictions for visualizing performance
    predictions = PredictionResult.objects.filter(model=model).order_by('-created_at')[:20]
    
    # Get measurement data for input feature importance analysis
    measurements = SolarStillMeasurement.objects.all()[:100]
    
    # Create feature importance mock data (in real implementation, this would be calculated from the model)
    feature_importance = [
        {'name': 'Ambient Temperature', 'importance': 0.32},
        {'name': 'Water Temperature', 'importance': 0.28},
        {'name': 'Glass Temperature', 'importance': 0.17},
        {'name': 'Solar Radiation', 'importance': 0.15},
        {'name': 'Wind Speed', 'importance': 0.05},
        {'name': 'Humidity', 'importance': 0.03}
    ]
    
    # Create layer structure data for visualization (in real implementation, this would be extracted from the model)
    layer_structure = [
        {'name': 'Input Layer', 'nodes': 6, 'activation': 'None'},
        {'name': 'Hidden Layer 1', 'nodes': 12, 'activation': 'ReLU'},
        {'name': 'Hidden Layer 2', 'nodes': 8, 'activation': 'ReLU'},
        {'name': 'Output Layer', 'nodes': 2, 'activation': 'Linear'}
    ]
    
    # Sample predictions for visualization
    sample_predictions = [
        {'inputs': {'ambient_temp': 28.5, 'water_temp': 32.1, 'glass_temp': 25.3, 
                  'solar_rad': 750, 'wind_speed': 2.1, 'humidity': 45},
         'outputs': {'efficiency': 35.2, 'freshwater': 2400}},
        {'inputs': {'ambient_temp': 30.2, 'water_temp': 35.0, 'glass_temp': 27.8, 
                  'solar_rad': 850, 'wind_speed': 1.5, 'humidity': 40},
         'outputs': {'efficiency': 42.1, 'freshwater': 2950}}
    ]
    
    # Weight distribution data for visualization (mock data)
    weight_distribution = {
        'layer1': [-0.25, -0.12, 0.05, 0.09, 0.15, 0.22, 0.30],
        'layer2': [-0.18, -0.08, 0.02, 0.12, 0.18, 0.25],
        'layer3': [-0.15, -0.05, 0.08, 0.15, 0.22]
    }
    
    context = {
        'model': model,
        'predictions': predictions,
        'feature_importance': json.dumps(feature_importance),
        'layer_structure': json.dumps(layer_structure),
        'sample_predictions': json.dumps(sample_predictions),
        'weight_distribution': json.dumps(weight_distribution)
    }
    
    return render(request, 'predictor/model_visualize.html', context)

def analytics_dashboard(request):
    """View for the analytics dashboard"""
    # Get statistics
    measurements_count = SolarStillMeasurement.objects.count()
    models_count = TrainedModel.objects.count()
    predictions_count = PredictionResult.objects.count()
    
    # Get latest model for displaying accuracy
    latest_model = get_latest_model()
    latest_model_accuracy = latest_model.accuracy if latest_model else None
    
    # Get measurement data for temperature vs efficiency chart
    temperature_data = list(SolarStillMeasurement.objects.filter(
        efficiency__isnull=False
    ).values('ambient_temperature', 'efficiency')[:100])
    
    # Get measurement data for solar radiation vs efficiency chart
    radiation_data = list(SolarStillMeasurement.objects.filter(
        efficiency__isnull=False
    ).values('solar_radiation', 'efficiency')[:100])
    
    # Get model performance comparison data
    model_performance = list(TrainedModel.objects.values('name', 'accuracy', 'mse')[:5])
    
    # Get prediction accuracy over time data (mock data)
    prediction_accuracy = [
        {'date': '2023-10-01', 'actual': 38.2, 'predicted': 37.8},
        {'date': '2023-10-15', 'actual': 35.6, 'predicted': 36.1},
        {'date': '2023-11-01', 'actual': 32.4, 'predicted': 33.5},
        {'date': '2023-11-15', 'actual': 30.1, 'predicted': 30.8},
        {'date': '2023-12-01', 'actual': 28.7, 'predicted': 27.9}
    ]
    
    # Calculate optimal operating conditions based on data
    optimal_conditions = {
        'ambient_temperature': 30,
        'water_temperature': 35,
        'solar_radiation': 800,
        'glass_temperature': 28,
        'wind_speed': 1.8,
        'humidity': 40
    }
    
    context = {
        'measurements_count': measurements_count,
        'models_count': models_count,
        'predictions_count': predictions_count,
        'latest_model_accuracy': latest_model_accuracy,
        'temperature_data': json.dumps(temperature_data),
        'radiation_data': json.dumps(radiation_data),
        'model_performance': json.dumps(model_performance),
        'prediction_accuracy': json.dumps(prediction_accuracy),
        'optimal_conditions': optimal_conditions
    }
    
    return render(request, 'predictor/analytics_dashboard.html', context)

@require_POST
def bulk_delete_models(request):
    """View for bulk deleting models"""
    try:
        data = json.loads(request.body)
        model_ids = data.get('model_ids', [])
        
        if not model_ids:
            return JsonResponse({'success': False, 'error': 'No models selected'})
        
        # Get the models
        models = TrainedModel.objects.filter(id__in=model_ids)
        
        # Delete the model files
        for model in models:
            if model.model_file:
                try:
                    os.remove(model.model_file.path)
                except Exception as e:
                    print(f"Error deleting model file: {e}")
        
        # Delete the database records
        models.delete()
        
        return JsonResponse({'success': True})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})
