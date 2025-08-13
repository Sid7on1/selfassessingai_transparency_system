import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError
import logging
import logging.config
from typing import Dict, List, Tuple
from pathlib import Path

# Set up logging
logging.config.dictConfig({
    'version': 1,
    'formatters': {
        'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        }
    },
    'handlers': {
        'wsgi': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://flask.logging.wsgi_errors_stream',
            'formatter': 'default'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'trust_calibration_analyzer.log',
            'maxBytes': 1000000,
            'backupCount': 1,
            'formatter': 'default'
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi', 'file']
    }
})

# Constants and configuration
DATA_DIR = Path('data')
RESULTS_DIR = Path('results')
CONFIG_FILE = Path('config.json')

# Load configuration
import json
with open(CONFIG_FILE, 'r') as f:
    config = json.load(f)

# Load data
def load_data(file_name: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(DATA_DIR / file_name)
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {file_name}")
        return None

# Preprocess data
def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    # Scale features
    scaler = StandardScaler()
    data[['feature1', 'feature2', 'feature3']] = scaler.fit_transform(data[['feature1', 'feature2', 'feature3']])
    
    # One-hot encode categorical variables
    data = pd.get_dummies(data, columns=['category'])
    
    return data

# Calculate compliance rates
def calculate_compliance_rates(data: pd.DataFrame) -> Dict[str, float]:
    # Calculate accuracy, precision, recall, and F1 score
    accuracy = accuracy_score(data['label'], data['predicted_label'])
    precision = precision_score(data['label'], data['predicted_label'])
    recall = recall_score(data['label'], data['predicted_label'])
    f1 = f1_score(data['label'], data['predicted_label'])
    
    # Calculate compliance rates
    compliance_rates = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return compliance_rates

# Analyze trust calibration
def analyze_trust_calibration(data: pd.DataFrame) -> Dict[str, float]:
    # Calculate trust calibration metrics
    trust_calibration_metrics = {
        'mean_absolute_error': np.mean(np.abs(data['trust'] - data['accuracy'])),
        'mean_squared_error': np.mean((data['trust'] - data['accuracy']) ** 2),
        'r_squared': stats.pearsonr(data['trust'], data['accuracy'])[0] ** 2
    }
    
    return trust_calibration_metrics

# Generate performance reports
def generate_performance_reports(data: pd.DataFrame, compliance_rates: Dict[str, float], trust_calibration_metrics: Dict[str, float]) -> None:
    # Create a report
    report = f"Compliance Rates:\n"
    report += f"Accuracy: {compliance_rates['accuracy']:.4f}\n"
    report += f"Precision: {compliance_rates['precision']:.4f}\n"
    report += f"Recall: {compliance_rates['recall']:.4f}\n"
    report += f"F1 Score: {compliance_rates['f1']:.4f}\n"
    report += f"\nTrust Calibration Metrics:\n"
    report += f"Mean Absolute Error: {trust_calibration_metrics['mean_absolute_error']:.4f}\n"
    report += f"Mean Squared Error: {trust_calibration_metrics['mean_squared_error']:.4f}\n"
    report += f"R Squared: {trust_calibration_metrics['r_squared']:.4f}\n"
    
    # Save the report to a file
    with open(RESULTS_DIR / 'performance_report.txt', 'w') as f:
        f.write(report)

# Create visualizations
def create_visualizations(data: pd.DataFrame, compliance_rates: Dict[str, float], trust_calibration_metrics: Dict[str, float]) -> None:
    # Create a plot of trust vs accuracy
    plt.scatter(data['trust'], data['accuracy'])
    plt.xlabel('Trust')
    plt.ylabel('Accuracy')
    plt.title('Trust vs Accuracy')
    plt.savefig(RESULTS_DIR / 'trust_vs_accuracy.png')
    plt.close()
    
    # Create a plot of compliance rates
    plt.bar(compliance_rates.keys(), compliance_rates.values())
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Compliance Rates')
    plt.savefig(RESULTS_DIR / 'compliance_rates.png')
    plt.close()
    
    # Create a plot of trust calibration metrics
    plt.bar(trust_calibration_metrics.keys(), trust_calibration_metrics.values())
    plt.xlabel('Metric')
    plt.ylabel('Value')
    plt.title('Trust Calibration Metrics')
    plt.savefig(RESULTS_DIR / 'trust_calibration_metrics.png')
    plt.close()

# Main function
def main() -> None:
    # Load data
    data = load_data('user_study_data.csv')
    
    # Preprocess data
    data = preprocess_data(data)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data.drop('label', axis=1), data['label'], test_size=0.2, random_state=42)
    
    # Train a model
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate compliance rates
    compliance_rates = calculate_compliance_rates(pd.DataFrame({'label': y_test, 'predicted_label': y_pred}))
    
    # Analyze trust calibration
    trust_calibration_metrics = analyze_trust_calibration(pd.DataFrame({'trust': data['trust'], 'accuracy': model.score(X_test, y_test)}))
    
    # Generate performance reports
    generate_performance_reports(data, compliance_rates, trust_calibration_metrics)
    
    # Create visualizations
    create_visualizations(data, compliance_rates, trust_calibration_metrics)

if __name__ == '__main__':
    main()