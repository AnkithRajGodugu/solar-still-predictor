from django import forms
from .models import SolarStillMeasurement, TrainedModel

class SolarStillMeasurementForm(forms.ModelForm):
    """Form for entering solar still measurements"""
    
    class Meta:
        model = SolarStillMeasurement
        fields = [
            'date', 'time', 'ambient_temperature', 'water_temperature', 
            'glass_temperature', 'solar_radiation', 'wind_speed', 'humidity',
            'efficiency', 'freshwater_quantity'
        ]
        widgets = {
            'date': forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}),
            'time': forms.TimeInput(attrs={'type': 'time', 'class': 'form-control'}),
            'ambient_temperature': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'water_temperature': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'glass_temperature': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'solar_radiation': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'wind_speed': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'humidity': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'efficiency': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'freshwater_quantity': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'})
        }
        labels = {
            'ambient_temperature': 'Ambient Temperature (°C)',
            'water_temperature': 'Water Temperature (°C)',
            'glass_temperature': 'Glass Temperature (°C)',
            'solar_radiation': 'Solar Radiation (W/m²)',
            'wind_speed': 'Wind Speed (m/s)',
            'humidity': 'Relative Humidity (%)',
            'efficiency': 'Efficiency (%)',
            'freshwater_quantity': 'Fresh Water Quantity (ml)'
        }

class PredictionForm(forms.ModelForm):
    """Form for predicting solar still performance without saving the measurements"""
    
    class Meta:
        model = SolarStillMeasurement
        fields = [
            'ambient_temperature', 'water_temperature', 'glass_temperature', 
            'solar_radiation', 'wind_speed', 'humidity'
        ]
        widgets = {
            'ambient_temperature': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'water_temperature': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'glass_temperature': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'solar_radiation': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'wind_speed': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'}),
            'humidity': forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'})
        }
        labels = {
            'ambient_temperature': 'Ambient Temperature (°C)',
            'water_temperature': 'Water Temperature (°C)',
            'glass_temperature': 'Glass Temperature (°C)',
            'solar_radiation': 'Solar Radiation (W/m²)',
            'wind_speed': 'Wind Speed (m/s)',
            'humidity': 'Relative Humidity (%)'
        }

class CSVUploadForm(forms.Form):
    """Form for uploading CSV data"""
    csv_file = forms.FileField(
        label='Upload CSV File',
        help_text='The CSV file should contain the following columns: ambient_temperature, water_temperature, glass_temperature, solar_radiation, wind_speed, humidity, efficiency, freshwater_quantity.',
        widget=forms.FileInput(attrs={'class': 'form-control'})
    )

class TrainModelForm(forms.Form):
    """Form for training a new model"""
    model_name = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    description = forms.CharField(
        required=False,
        widget=forms.Textarea(attrs={'class': 'form-control', 'rows': 3})
    )
    epochs = forms.IntegerField(
        initial=100,
        min_value=10,
        max_value=1000,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    batch_size = forms.IntegerField(
        initial=32,
        min_value=8,
        max_value=128,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    use_sample_data = forms.BooleanField(
        required=False,
        initial=False,
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        help_text='Use sample data instead of real measurements'
    ) 