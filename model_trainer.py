import logging
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Tuple, Dict
import numpy as np
from sklearn.exceptions import NotFittedError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelTrainer:
    """
    Training pipeline for both the primary Random Forest and the self-assessing decision tree.
    """

    def __init__(self, config: Dict):
        """
        Initialize the ModelTrainer with a configuration dictionary.

        Args:
        - config (Dict): A dictionary containing configuration settings.
        """
        self.config = config
        self.random_forest_model = None
        self.decision_tree_model = None

    def load_acs_data(self, file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load ACS data from a CSV file.

        Args:
        - file_path (str): The path to the CSV file.

        Returns:
        - Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the feature data and target data.
        """
        try:
            data = pd.read_csv(file_path)
            features = data.drop(self.config['target_column'], axis=1)
            target = data[self.config['target_column']]
            return features, target
        except Exception as e:
            logging.error(f"Failed to load data: {str(e)}")
            raise

    def train_random_forest(self, features: pd.DataFrame, target: pd.DataFrame) -> RandomForestClassifier:
        """
        Train a Random Forest classifier.

        Args:
        - features (pd.DataFrame): The feature data.
        - target (pd.DataFrame): The target data.

        Returns:
        - RandomForestClassifier: The trained Random Forest model.
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=self.config['test_size'], random_state=self.config['random_state'])
            model = RandomForestClassifier(n_estimators=self.config['n_estimators'], random_state=self.config['random_state'])
            model.fit(X_train, y_train)
            self.random_forest_model = model
            return model
        except Exception as e:
            logging.error(f"Failed to train Random Forest model: {str(e)}")
            raise

    def train_flaw_decision_tree(self, features: pd.DataFrame, target: pd.DataFrame) -> DecisionTreeClassifier:
        """
        Train a decision tree classifier to predict flaws.

        Args:
        - features (pd.DataFrame): The feature data.
        - target (pd.DataFrame): The target data.

        Returns:
        - DecisionTreeClassifier: The trained decision tree model.
        """
        try:
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=self.config['test_size'], random_state=self.config['random_state'])
            model = DecisionTreeClassifier(random_state=self.config['random_state'])
            model.fit(X_train, y_train)
            self.decision_tree_model = model
            return model
        except Exception as e:
            logging.error(f"Failed to train decision tree model: {str(e)}")
            raise

    def evaluate_models(self, features: pd.DataFrame, target: pd.DataFrame) -> Dict:
        """
        Evaluate the performance of the trained models.

        Args:
        - features (pd.DataFrame): The feature data.
        - target (pd.DataFrame): The target data.

        Returns:
        - Dict: A dictionary containing the evaluation metrics.
        """
        try:
            if self.random_forest_model is None or self.decision_tree_model is None:
                raise NotFittedError("Models are not trained yet.")
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=self.config['test_size'], random_state=self.config['random_state'])
            random_forest_predictions = self.random_forest_model.predict(X_test)
            decision_tree_predictions = self.decision_tree_model.predict(X_test)
            random_forest_accuracy = accuracy_score(y_test, random_forest_predictions)
            decision_tree_accuracy = accuracy_score(y_test, decision_tree_predictions)
            random_forest_report = classification_report(y_test, random_forest_predictions)
            decision_tree_report = classification_report(y_test, decision_tree_predictions)
            random_forest_confusion_matrix = confusion_matrix(y_test, random_forest_predictions)
            decision_tree_confusion_matrix = confusion_matrix(y_test, decision_tree_predictions)
            evaluation_metrics = {
                'random_forest_accuracy': random_forest_accuracy,
                'decision_tree_accuracy': decision_tree_accuracy,
                'random_forest_report': random_forest_report,
                'decision_tree_report': decision_tree_report,
                'random_forest_confusion_matrix': random_forest_confusion_matrix,
                'decision_tree_confusion_matrix': decision_tree_confusion_matrix
            }
            return evaluation_metrics
        except Exception as e:
            logging.error(f"Failed to evaluate models: {str(e)}")
            raise

    def save_models(self, file_path: str) -> None:
        """
        Save the trained models to a file.

        Args:
        - file_path (str): The path to save the models.
        """
        try:
            if self.random_forest_model is None or self.decision_tree_model is None:
                raise NotFittedError("Models are not trained yet.")
            joblib.dump(self.random_forest_model, f"{file_path}_random_forest.joblib")
            joblib.dump(self.decision_tree_model, f"{file_path}_decision_tree.joblib")
        except Exception as e:
            logging.error(f"Failed to save models: {str(e)}")
            raise

def main():
    config = {
        'target_column': 'target',
        'test_size': 0.2,
        'random_state': 42,
        'n_estimators': 100
    }
    trainer = ModelTrainer(config)
    features, target = trainer.load_acs_data('acs_data.csv')
    random_forest_model = trainer.train_random_forest(features, target)
    decision_tree_model = trainer.train_flaw_decision_tree(features, target)
    evaluation_metrics = trainer.evaluate_models(features, target)
    trainer.save_models('trained_models')

if __name__ == "__main__":
    main()