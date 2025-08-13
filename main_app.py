import logging
import json
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# Configuration
class Config:
    """Application configuration"""
    DEBUG = True
    TESTING = False
    SQLALCHEMY_DATABASE_URI = 'sqlite:///database.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

# Initialize Flask application
app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)

# Define models
class UserInteraction(db.Model):
    """User interaction model"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    interaction_type = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False)

    def __init__(self, user_id: int, interaction_type: str, timestamp: str):
        self.user_id = user_id
        self.interaction_type = interaction_type
        self.timestamp = timestamp

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Define exception classes
class InvalidRequestError(Exception):
    """Invalid request error"""
    pass

class ModelInitializationError(Exception):
    """Model initialization error"""
    pass

# Define utility functions
def load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV file"""
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def split_data(data: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into training and testing sets"""
    try:
        X = data.drop('target', axis=1)
        y = data['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        raise

# Define main class
class MainApp:
    """Main application class"""
    def __init__(self):
        self.random_forest_model = None
        self.decision_tree_model = None

    def initialize_models(self) -> None:
        """Initialize models"""
        try:
            data = load_data('data.csv')
            X_train, X_test, y_train, y_test = split_data(data)
            self.random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.random_forest_model.fit(X_train, y_train)
            self.decision_tree_model = DecisionTreeClassifier(random_state=42)
            self.decision_tree_model.fit(X_train, y_train)
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise ModelInitializationError("Error initializing models")

    def handle_prediction_request(self, request_data: Dict) -> Dict:
        """Handle prediction request"""
        try:
            user_id = request_data['user_id']
            input_data = request_data['input_data']
            input_df = pd.DataFrame([input_data])
            prediction = self.random_forest_model.predict(input_df)
            explanation = self.generate_explanations(input_df, prediction)
            return {'prediction': prediction[0], 'explanation': explanation}
        except Exception as e:
            logger.error(f"Error handling prediction request: {e}")
            raise InvalidRequestError("Error handling prediction request")

    def generate_explanations(self, input_data: pd.DataFrame, prediction: np.ndarray) -> str:
        """Generate explanations"""
        try:
            explanation = self.decision_tree_model.predict(input_data)
            return f"Explanation: {explanation[0]}"
        except Exception as e:
            logger.error(f"Error generating explanations: {e}")
            raise

    def log_user_interaction(self, user_id: int, interaction_type: str, timestamp: str) -> None:
        """Log user interaction"""
        try:
            interaction = UserInteraction(user_id, interaction_type, timestamp)
            db.session.add(interaction)
            db.session.commit()
        except Exception as e:
            logger.error(f"Error logging user interaction: {e}")

    def serve_frontend(self) -> str:
        """Serve frontend"""
        try:
            return "Frontend served successfully"
        except Exception as e:
            logger.error(f"Error serving frontend: {e}")
            raise

# Create main app instance
main_app = MainApp()

# Define routes
@app.route('/predict', methods=['POST'])
def predict():
    """Predict route"""
    try:
        request_data = request.get_json()
        response = main_app.handle_prediction_request(request_data)
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error handling prediction request: {e}")
        return jsonify({'error': 'Error handling prediction request'}), 500

@app.route('/log_interaction', methods=['POST'])
def log_interaction():
    """Log interaction route"""
    try:
        request_data = request.get_json()
        main_app.log_user_interaction(request_data['user_id'], request_data['interaction_type'], request_data['timestamp'])
        return jsonify({'message': 'Interaction logged successfully'})
    except Exception as e:
        logger.error(f"Error logging interaction: {e}")
        return jsonify({'error': 'Error logging interaction'}), 500

@app.route('/frontend', methods=['GET'])
def frontend():
    """Frontend route"""
    try:
        response = main_app.serve_frontend()
        return response
    except Exception as e:
        logger.error(f"Error serving frontend: {e}")
        return jsonify({'error': 'Error serving frontend'}), 500

# Initialize models
main_app.initialize_models()

# Run application
if __name__ == '__main__':
    app.run(debug=True)