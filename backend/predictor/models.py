from django.db import models
from django.utils import timezone
import numpy as np
import os
from django.core.files.storage import default_storage

# Create your models here.
class SolarStillMeasurement(models.Model):
    """Model for storing solar still measurements"""
    # Metadata
    date = models.DateField(default=timezone.now)
    time = models.TimeField(default=timezone.now)
    
    # Input parameters - adjust based on your specific requirements
    ambient_temperature = models.FloatField(help_text="Ambient temperature (°C)")
    water_temperature = models.FloatField(help_text="Water temperature (°C)")
    glass_temperature = models.FloatField(help_text="Glass temperature (°C)")
    solar_radiation = models.FloatField(help_text="Solar radiation (W/m²)")
    wind_speed = models.FloatField(help_text="Wind speed (m/s)", null=True, blank=True)
    humidity = models.FloatField(help_text="Relative humidity (%)", null=True, blank=True)
    
    # Output parameters - what we're trying to predict
    efficiency = models.FloatField(help_text="Efficiency (%)", null=True, blank=True)
    freshwater_quantity = models.FloatField(help_text="Fresh water produced (ml)", null=True, blank=True)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Measurement on {self.date} at {self.time}"
    
    class Meta:
        ordering = ['-date', '-time']

class TrainedModel(models.Model):
    """Model for storing trained ANN models"""
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True, null=True)
    model_file = models.FileField(upload_to='trained_models/', storage=default_storage)
    accuracy = models.FloatField(help_text="Model accuracy (%)", null=True, blank=True)
    mse = models.FloatField(help_text="Mean Squared Error", null=True, blank=True)
    is_active = models.BooleanField(default=False, help_text="Whether this model is currently active for predictions")
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name
    
    def delete(self, *args, **kwargs):
        # Delete the file when the model is deleted
        if self.model_file:
            if os.path.isfile(self.model_file.path):
                os.remove(self.model_file.path)
        super().delete(*args, **kwargs)
    
    def save(self, *args, **kwargs):
        # If this model is being set as active, deactivate all other models
        if self.is_active:
            TrainedModel.objects.exclude(pk=self.pk).update(is_active=False)
        super().save(*args, **kwargs)
    
    class Meta:
        ordering = ['-created_at']

class PredictionResult(models.Model):
    """Model for storing prediction results"""
    measurement = models.ForeignKey(SolarStillMeasurement, on_delete=models.CASCADE, related_name='predictions')
    model = models.ForeignKey(TrainedModel, on_delete=models.SET_NULL, null=True, related_name='predictions')
    predicted_efficiency = models.FloatField(help_text="Predicted efficiency (%)")
    predicted_freshwater = models.FloatField(help_text="Predicted fresh water quantity (ml)")
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Prediction for {self.measurement}"
