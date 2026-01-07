/**
 * Solar Still Predictor - Main JavaScript File
 */

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    const tooltipList = [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));
    
    // Initialize date pickers
    const datePickers = document.querySelectorAll('.datepicker');
    datePickers.forEach(picker => {
        if (picker) {
            flatpickr(picker, {
                dateFormat: "Y-m-d",
                allowInput: true
            });
        }
    });
    
    // Form validation
    const forms = document.querySelectorAll('.needs-validation');
    
    Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            
            form.classList.add('was-validated');
        }, false);
    });
    
    // Real-time prediction functionality (for prediction.html)
    setupRealTimePrediction();
    
    // Chart initialization (for dashboard.html)
    initializeCharts();
});

/**
 * Setup real-time prediction using input sliders
 */
function setupRealTimePrediction() {
    const predictionForm = document.getElementById('real-time-prediction-form');
    if (!predictionForm) return;
    
    const sliders = predictionForm.querySelectorAll('input[type="range"]');
    const predictButton = document.getElementById('predict-button');
    
    // Update displayed values when sliders change
    sliders.forEach(slider => {
        const output = document.getElementById(slider.id + '-value');
        if (output) {
            // Set initial value
            output.textContent = slider.value;
            
            // Update on change
            slider.addEventListener('input', function() {
                output.textContent = this.value;
            });
        }
        
        // Make prediction when slider changes (with debounce)
        slider.addEventListener('change', debounce(makePrediction, 500));
    });
    
    // Make prediction when button is clicked
    if (predictButton) {
        predictButton.addEventListener('click', makePrediction);
    }
}

/**
 * Make an AJAX prediction request
 */
function makePrediction() {
    const form = document.getElementById('real-time-prediction-form');
    if (!form) return;
    
    const resultContainer = document.getElementById('prediction-results');
    const loadingIndicator = document.getElementById('loading-indicator');
    
    if (loadingIndicator) loadingIndicator.classList.remove('d-none');
    if (resultContainer) resultContainer.classList.add('d-none');
    
    const formData = new FormData(form);
    
    fetch('/predict/', {
        method: 'POST',
        body: formData,
        headers: {
            'X-Requested-With': 'XMLHttpRequest'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (loadingIndicator) loadingIndicator.classList.add('d-none');
        if (resultContainer) {
            resultContainer.classList.remove('d-none');
            
            // Update prediction results
            const efficiencyElement = document.getElementById('predicted-efficiency');
            const freshwaterElement = document.getElementById('predicted-freshwater');
            
            if (efficiencyElement) efficiencyElement.textContent = data.efficiency.toFixed(2) + '%';
            if (freshwaterElement) freshwaterElement.textContent = data.freshwater.toFixed(2) + ' ml';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        if (loadingIndicator) loadingIndicator.classList.add('d-none');
        
        // Show error message
        const errorElement = document.getElementById('prediction-error');
        if (errorElement) {
            errorElement.classList.remove('d-none');
            errorElement.textContent = 'Error making prediction. Please try again.';
        }
    });
}

/**
 * Initialize charts for dashboard
 */
function initializeCharts() {
    const performanceChart = document.getElementById('performance-chart');
    if (!performanceChart) return;
    
    const ctx = performanceChart.getContext('2d');
    
    // Sample data - would be replaced with real data from the backend
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            datasets: [
                {
                    label: 'Efficiency (%)',
                    data: [42, 45, 48, 46, 52, 54],
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    borderWidth: 2,
                    tension: 0.3
                },
                {
                    label: 'Freshwater (ml)',
                    data: [320, 340, 360, 350, 400, 420],
                    borderColor: '#2ecc71',
                    backgroundColor: 'rgba(46, 204, 113, 0.1)',
                    borderWidth: 2,
                    tension: 0.3,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Efficiency (%)'
                    }
                },
                y1: {
                    beginAtZero: true,
                    position: 'right',
                    grid: {
                        drawOnChartArea: false,
                    },
                    title: {
                        display: true,
                        text: 'Freshwater (ml)'
                    }
                }
            }
        }
    });
}

/**
 * Debounce function to limit function calls
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
} 