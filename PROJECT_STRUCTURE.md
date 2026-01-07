# Solar Still Predictor - Project Structure

This document explains the project structure, which separates frontend and backend concerns for better code organization and maintainability.

## Overview

The project follows a structured approach that separates:

1. **Backend**: Contains all server-side logic, database models, and business logic
2. **Frontend**: Contains all UI templates, styles, and client-side scripts

This separation makes the codebase more maintainable and easier to understand.

## Directory Structure

```
solar_still_predictor/
│
├── backend/                      # All server-side code
│   ├── predictor/                # Main application backend
│   │   ├── ml/                   # Machine learning components
│   │   │   ├── __init__.py
│   │   │   ├── model.py          # ANN model implementation
│   │   │   └── data_handler.py   # Data processing utilities
│   │   ├── migrations/           # Database migrations
│   │   ├── __init__.py
│   │   ├── admin.py              # Django admin configuration
│   │   ├── apps.py               # App configuration
│   │   ├── forms.py              # Form definitions
│   │   ├── models.py             # Database models
│   │   ├── tests.py              # Unit tests
│   │   ├── urls.py               # URL routing for the app
│   │   ├── utils.py              # Utility functions
│   │   └── views.py              # View controllers
│   │
│   └── solar_still/              # Project settings
│       ├── __init__.py
│       ├── asgi.py               # ASGI configuration
│       ├── settings.py           # Project settings
│       ├── urls.py               # Main URL routing
│       └── wsgi.py               # WSGI configuration
│
├── frontend/                     # All client-side code
│   ├── static/                   # Static assets
│   │   ├── css/                  # Stylesheets
│   │   │   └── main.css          # Main stylesheet
│   │   ├── js/                   # JavaScript files
│   │   │   └── main.js           # Main JavaScript file
│   │   └── img/                  # Images
│   └── templates/                # HTML templates
│       └── predictor/            # Templates for the predictor app
│           ├── base.html         # Base template
│           ├── dashboard.html    # Dashboard template
│           ├── measurement_detail.html
│           ├── measurement_form.html
│           ├── measurement_list.html
│           ├── model_detail.html
│           ├── model_list.html
│           ├── model_visualize.html    # Neural network visualization
│           ├── analytics_dashboard.html # Analytics dashboard
│           ├── prediction.html
│           ├── train_model.html
│           ├── upload_csv.html
│           ├── export_data.html
│           └── confirm_delete.html
│
├── media/                        # User-uploaded files
├── trained_models/               # Saved ANN models
├── staticfiles/                  # Collected static files
├── runserver/                    # Server logs and runtime files
├── manage.py                     # Django management script
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

## Database Schema

### Models

1. **SolarStillMeasurement**
   - Stores actual measurements from the solar still
   - Fields:
     - Input parameters (ambient_temperature, water_temperature, etc.)
     - Output parameters (efficiency, freshwater_quantity)
     - Metadata (date, time, created_at, updated_at)

2. **TrainedModel**
   - Stores trained neural network models
   - Fields:
     - Model information (name, description)
     - Performance metrics (accuracy, mse)
     - Model file storage
     - Active status flag

3. **PredictionResult**
   - Stores predictions made by models
   - Fields:
     - Links to measurement and model
     - Predicted values
     - Creation timestamp

### Relationships
- One-to-Many: SolarStillMeasurement → PredictionResult
- One-to-Many: TrainedModel → PredictionResult
- Many-to-One: PredictionResult → SolarStillMeasurement
- Many-to-One: PredictionResult → TrainedModel

## Key Components

### Backend Components

1. **Models (backend/predictor/models.py)**
   - Data structure definitions for the application
   - Includes SolarStillMeasurement, TrainedModel, and PredictionResult models

2. **Views (backend/predictor/views.py)**
   - View controllers handling HTTP requests and responses
   - Implements business logic for the application

3. **Machine Learning (backend/predictor/ml/)**
   - Contains the ANN model implementation
   - Handles data preprocessing and model training

4. **Forms (backend/predictor/forms.py)**
   - Form definitions for data input and validation

5. **Utilities (backend/predictor/utils.py)**
   - Helper functions for common tasks

### Frontend Components

1. **Templates (frontend/templates/predictor/)**
   - HTML templates for rendering the UI
   - Uses Django's template language for dynamic content

2. **CSS (frontend/static/css/main.css)**
   - Styling for the application
   - Responsive design for various device sizes

3. **JavaScript (frontend/static/js/main.js)**
   - Client-side behavior and interactivity
   - AJAX functionality for real-time predictions

## Application Flow

1. **Data Input**:
   - User submits measurements via the form or uploads a CSV
   - Backend validates and saves the data to the database

2. **Model Training**:
   - User initiates model training
   - Backend processes data and trains an ANN
   - Saves trained model for future predictions

3. **Prediction**:
   - User requests prediction with input parameters
   - Backend loads the model and makes predictions
   - Results are displayed to the user

4. **Data Export**:
   - User exports data for external analysis
   - Backend processes and formats the data
   - Downloads the file in the selected format

## How to Run

Run the Django development server:
```
python manage.py runserver
```

Make sure your PYTHONPATH includes the project root directory:
```
export PYTHONPATH=$PYTHONPATH:/path/to/project
```

## Development Workflow

When working on this project:

1. **Backend Changes**:
   - Modify files in the `backend` directory
   - Run tests to ensure functionality
   - Apply database migrations if needed

2. **Frontend Changes**:
   - Modify templates, CSS, or JavaScript in the `frontend` directory
   - Changes will be reflected when the page is refreshed

3. **Adding New Features**:
   - Add backend logic in the appropriate backend files
   - Add frontend components in the corresponding frontend files
   - Update URL routing in `backend/predictor/urls.py` 

## Future Architecture Changes

### Backend Enhancements
- **API Layer**: New `api/` directory for RESTful endpoints
- **Task Queue**: Integration with Celery for background processing
- **Caching Layer**: Redis integration for performance optimization
- **Authentication**: JWT-based authentication system
- **WebSocket Support**: Real-time updates and notifications

### Frontend Enhancements
- **Component Library**: Reusable UI components
- **State Management**: Integration with Redux or Vuex
- **Progressive Web App**: PWA support for offline functionality
- **Mobile-First Design**: Enhanced responsive layouts
- **Theme System**: Customizable UI themes

### Data Layer Improvements
- **Data Pipeline**: ETL processes for data processing
- **Data Warehouse**: Integration with data warehouse solutions
- **Time Series DB**: Specialized storage for time-series data
- **Search Engine**: Elasticsearch integration for advanced search
- **File Storage**: Distributed file storage system

### Monitoring and Logging
- **Application Monitoring**: Performance and error tracking
- **User Analytics**: Usage patterns and behavior analysis
- **System Health**: Resource usage and system metrics
- **Audit Logging**: Track all system changes and actions
- **Alerting System**: Automated alert generation

### Security Enhancements
- **Role-Based Access**: Granular permission system
- **API Security**: Rate limiting and API key management
- **Data Encryption**: End-to-end encryption for sensitive data
- **Security Headers**: Enhanced security headers
- **Audit Trail**: Comprehensive security audit logging 