from django.contrib import admin
from .models import SolarStillMeasurement, PredictionResult, TrainedModel

@admin.register(SolarStillMeasurement)
class SolarStillMeasurementAdmin(admin.ModelAdmin):
    list_display = ('date', 'time', 'ambient_temperature', 'water_temperature', 
                   'glass_temperature', 'solar_radiation', 'efficiency', 'freshwater_quantity')
    list_filter = ('date',)
    search_fields = ('date', 'time')
    date_hierarchy = 'date'

@admin.register(PredictionResult)
class PredictionResultAdmin(admin.ModelAdmin):
    list_display = ('measurement', 'predicted_efficiency', 'predicted_freshwater', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('measurement__date',)

@admin.register(TrainedModel)
class TrainedModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'accuracy', 'mse', 'created_at')
    list_filter = ('created_at',)
    search_fields = ('name', 'description')
