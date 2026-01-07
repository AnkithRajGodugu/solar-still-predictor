from django.urls import path
from . import views

urlpatterns = [
    # Dashboard
    path('', views.dashboard, name='dashboard'),
    
    # Measurements
    path('measurements/', views.MeasurementListView.as_view(), name='measurement-list'),
    path('measurements/add/', views.MeasurementCreateView.as_view(), name='measurement-create'),
    path('measurements/<int:pk>/', views.MeasurementDetailView.as_view(), name='measurement-detail'),
    path('measurements/<int:pk>/edit/', views.MeasurementUpdateView.as_view(), name='measurement-update'),
    path('measurements/<int:pk>/delete/', views.MeasurementDeleteView.as_view(), name='measurement-delete'),
    
    # Model management
    path('models/', views.ModelListView.as_view(), name='model-list'),
    path('models/<int:pk>/', views.ModelDetailView.as_view(), name='model-detail'),
    path('models/<int:pk>/edit/', views.ModelUpdateView.as_view(), name='model-update'),
    path('models/<int:pk>/delete/', views.ModelDeleteView.as_view(), name='model-delete'),
    path('models/<int:pk>/set-active/', views.set_active_model, name='set-active-model'),
    path('models/<int:pk>/visualize/', views.visualize_model, name='model-visualize'),
    path('models/bulk-delete/', views.bulk_delete_models, name='bulk-delete-models'),
    path('train-model/', views.train_model, name='train-model'),
    
    # Prediction
    path('predict/', views.make_prediction, name='predict'),
    path('measurements/<int:pk>/predict/', views.save_prediction_for_measurement, name='save-prediction'),
    path('api/predict/', views.predict_api, name='api-predict'),
    
    # Data management
    path('upload-csv/', views.upload_csv, name='upload-csv'),
    path('export-data/', views.export_data, name='export-csv'),
    path('download-template/', views.download_csv_template, name='download-template'),
    
    # Analytics
    path('analytics/dashboard/', views.analytics_dashboard, name='analytics-dashboard'),
    
    # Test view
    path('test/', views.test_view, name='test-view'),
] 