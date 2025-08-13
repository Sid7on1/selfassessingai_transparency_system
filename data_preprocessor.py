import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.exceptions import NotFittedError
from typing import Tuple, Dict, Any
import warnings
import os
import json
from enum import Enum
from threading import Lock

# Define constants
DATA_DIR = 'data'
RAW_DATA_FILE = 'acs_pums.csv'
PROCESSED_DATA_FILE = 'processed_data.csv'
CONFIG_FILE = 'config.json'

# Define logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('data_preprocessor.log')
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Define exception classes
class DataPreprocessorError(Exception):
    pass

class InvalidConfigError(DataPreprocessorError):
    pass

class DataLoadingError(DataPreprocessorError):
    pass

class DataProcessingError(DataPreprocessorError):
    pass

# Define configuration class
class Config:
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.load_config()

    def load_config(self):
        try:
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            raise InvalidConfigError(f'Config file {self.config_file} not found')
        except json.JSONDecodeError:
            raise InvalidConfigError(f'Invalid config file {self.config_file}')

    def get_config(self, key: str):
        return self.config.get(key)

# Define data preprocessor class
class DataPreprocessor:
    def __init__(self, config: Config):
        self.config = config
        self.raw_data = None
        self.processed_data = None
        self.lock = Lock()

    def load_raw_data(self) -> pd.DataFrame:
        try:
            self.raw_data = pd.read_csv(os.path.join(DATA_DIR, RAW_DATA_FILE))
            logger.info(f'Loaded raw data from {RAW_DATA_FILE}')
            return self.raw_data
        except FileNotFoundError:
            raise DataLoadingError(f'Raw data file {RAW_DATA_FILE} not found')
        except pd.errors.EmptyDataError:
            raise DataLoadingError(f'Raw data file {RAW_DATA_FILE} is empty')
        except pd.errors.ParserError:
            raise DataLoadingError(f'Error parsing raw data file {RAW_DATA_FILE}')

    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            # Replace missing values with mean or median
            for col in data.columns:
                if data[col].dtype == 'object':
                    data[col] = data[col].fillna(data[col].mode()[0])
                else:
                    data[col] = data[col].fillna(data[col].mean())
            logger.info('Handled missing values')
            return data
        except Exception as e:
            raise DataProcessingError(f'Error handling missing values: {str(e)}')

    def create_income_threshold(self, data: pd.DataFrame) -> float:
        try:
            # Calculate income threshold using 75th percentile
            income_threshold = np.percentile(data['income'], 75)
            logger.info(f'Created income threshold: {income_threshold}')
            return income_threshold
        except Exception as e:
            raise DataProcessingError(f'Error creating income threshold: {str(e)}')

    def balance_dataset(self, data: pd.DataFrame, income_threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            # Split data into two classes based on income threshold
            class_labels = np.where(data['income'] > income_threshold, 1, 0)
            data['class_label'] = class_labels
            # Balance dataset using class weights
            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(class_labels), y=class_labels)
            logger.info('Balanced dataset')
            return data, class_weights
        except Exception as e:
            raise DataProcessingError(f'Error balancing dataset: {str(e)}')

    def save_processed_data(self, data: pd.DataFrame, file_name: str):
        try:
            data.to_csv(os.path.join(DATA_DIR, file_name), index=False)
            logger.info(f'Saved processed data to {file_name}')
        except Exception as e:
            raise DataProcessingError(f'Error saving processed data: {str(e)}')

    def process_data(self):
        with self.lock:
            try:
                raw_data = self.load_raw_data()
                processed_data = self.handle_missing_values(raw_data)
                income_threshold = self.create_income_threshold(processed_data)
                balanced_data, class_weights = self.balance_dataset(processed_data, income_threshold)
                self.save_processed_data(balanced_data, PROCESSED_DATA_FILE)
                logger.info('Data processing complete')
            except Exception as e:
                logger.error(f'Data processing error: {str(e)}')

# Define main function
def main():
    config = Config(CONFIG_FILE)
    data_preprocessor = DataPreprocessor(config)
    data_preprocessor.process_data()

if __name__ == '__main__':
    main()