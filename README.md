# Solar Still Predictor

A Django-based web application for predicting solar still performance using Artificial Neural Networks (ANN).

## Overview

This application helps researchers and engineers predict the efficiency and freshwater production of solar stills based on various environmental parameters. By leveraging machine learning techniques, specifically ANNs, the system can learn from historical data to make accurate predictions about solar still performance under different conditions.

## Features

- **Interactive Dashboard**: View key metrics, recent measurements, and performance summaries at a glance
- **Measurement Management**: Add, edit, view, and delete solar still measurements
- **Predictive Modeling**: Train custom ANN models using your own data or sample datasets
- **Performance Prediction**: Predict efficiency and freshwater production based on environmental parameters
- **Data Import/Export**: Upload data via CSV or export in various formats (CSV, XLSX, JSON)
- **Visualization**: 
  - Interactive neural network visualization
  - Performance metrics charts
  - Analytics dashboard with D3.js visualizations
  - Real-time data updates

## Tech Stack

- **Backend**: Django
- **Machine Learning**: TensorFlow, Keras
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Frontend**: Bootstrap, jQuery, D3.js
- **Database**: SQLite (default)
- **Visualization**: D3.js for interactive charts and neural network visualization


## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.

## Installation

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/coriuday/solar-still-predictor.git
   cd solar-still-predictor
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run database migrations:
   ```
   python manage.py migrate
   ```

5. (Recommended for production) Set environment variables for security:
    - On Linux/macOS:
       ```sh
       export DJANGO_SECRET_KEY='your-production-secret-key'
       export DJANGO_DEBUG=False
       ```
    - On Windows (cmd):
       ```cmd
       set DJANGO_SECRET_KEY=your-production-secret-key
       set DJANGO_DEBUG=False
       ```
    - On Windows (PowerShell):
       ```powershell
       $env:DJANGO_SECRET_KEY="your-production-secret-key"
       $env:DJANGO_DEBUG="False"
       ```
    If these are not set, the app will use the default development key and DEBUG=True.

5. Create a superuser (optional, for admin access):
   ```
   python manage.py createsuperuser
   ```

## Usage

1. Start the development server:
   ```
   python manage.py runserver
   ```

2. Open your browser and navigate to `http://127.0.0.1:8000/`

### Workflow

1. **Add Measurements**: Enter solar still measurement data including environmental parameters and actual performance metrics (if available)
2. **Train a Model**: Create a custom ANN model using your collected data
3. **Make Predictions**: Predict solar still performance by entering environmental parameters
4. **Analyze Results**: View predictions, compare with actual results, and export data for further analysis

## Project Structure

- `predictor/`: Main application directory
  - `ml/`: Machine learning components
    - `model.py`: ANN implementation
    - `data_handler.py`: Data processing utilities
  - `models.py`: Database models
  - `views.py`: View controllers
  - `forms.py`: Form definitions
  - `utils.py`: Utility functions
  - `templates/`: HTML templates
  - `admin.py`: Admin interface configuration

## Data Format

The system works with the following parameters:

### Input Parameters
- Ambient Temperature (°C)
- Water Temperature (°C)
- Glass Temperature (°C)
- Solar Radiation (W/m²)
- Wind Speed (m/s) - optional
- Humidity (%) - optional

### Output Parameters
- Efficiency (%)
- Fresh Water Quantity (ml)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project was developed as part of research on improving solar still efficiency prediction.
- The ANN architecture is based on research in solar energy applications and machine learning techniques.

## Planned Features and Improvements

### Advanced Analytics
- **Time Series Analysis**: Track performance trends over time
- **Seasonal Analysis**: Compare performance across different seasons
- **Correlation Analysis**: Identify relationships between input parameters
- **Anomaly Detection**: Flag unusual measurements or predictions

### Enhanced Visualization
- **3D Model Visualization**: Interactive 3D representation of the neural network
- **Heat Maps**: Visualize parameter relationships and correlations
- **Geographic Visualization**: Map-based view of solar still locations
- **Custom Chart Builder**: Allow users to create custom visualizations

### Machine Learning Improvements
- **AutoML Integration**: Automatic model selection and hyperparameter tuning
- **Ensemble Methods**: Combine predictions from multiple models
- **Transfer Learning**: Adapt pre-trained models for specific solar still configurations
- **Online Learning**: Continuous model updates with new data

### User Experience
- **Mobile App**: Native mobile application for field measurements
- **API Access**: RESTful API for external system integration
- **Batch Processing**: Handle large datasets efficiently
- **Real-time Alerts**: Notifications for performance issues or maintenance needs

### Data Management
- **Data Validation**: Advanced validation rules for measurements
- **Version Control**: Track changes to models and datasets
- **Backup System**: Automated backup of models and data
- **Data Cleaning**: Automated data cleaning and preprocessing

### Collaboration Features
- **Multi-user Support**: Role-based access control
- **Project Sharing**: Share models and datasets with other users
- **Comments and Notes**: Add annotations to measurements and predictions
- **Export Templates**: Customizable export formats

### Integration Capabilities
- **IoT Integration**: Connect with solar still sensors
- **Weather API**: Integration with weather forecasting services
- **Cloud Storage**: Backup and sync with cloud services
- **External Tools**: Export to common analysis tools (MATLAB, R, etc.)