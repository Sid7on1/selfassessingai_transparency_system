import os
import json
import logging
from typing import Dict, Any
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfigKeys(Enum):
    """Enum for configuration keys"""
    MODEL_PARAMETERS = 'model_parameters'
    STUDY_SETTINGS = 'study_settings'
    API_CREDENTIALS = 'api_credentials'

class ConfigException(Exception):
    """Custom exception for configuration errors"""
    pass

class Config:
    """Central configuration class"""
    def __init__(self, config_file: str = 'config.json'):
        """
        Initialize the configuration class.

        Args:
        - config_file (str): Path to the configuration file (default: 'config.json')
        """
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """
        Load the configuration from the file.

        Returns:
        - config (Dict[str, Any]): Loaded configuration
        """
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file '{self.config_file}' not found.")
            raise ConfigException(f"Configuration file '{self.config_file}' not found.")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise ConfigException(f"Error parsing configuration file: {e}")

    def load_environment_variables(self) -> None:
        """
        Load environment variables into the configuration.
        """
        for key, value in os.environ.items():
            if key.startswith('CONFIG_'):
                config_key = key.replace('CONFIG_', '')
                self.config[config_key] = value

    def validate_configurations(self) -> None:
        """
        Validate the configuration.

        Raises:
        - ConfigException: If the configuration is invalid
        """
        required_keys = [ConfigKeys.MODEL_PARAMETERS.value, ConfigKeys.STUDY_SETTINGS.value, ConfigKeys.API_CREDENTIALS.value]
        for key in required_keys:
            if key not in self.config:
                logger.error(f"Missing configuration key: {key}")
                raise ConfigException(f"Missing configuration key: {key}")

    def provide_model_settings(self) -> Dict[str, Any]:
        """
        Provide the model settings.

        Returns:
        - model_settings (Dict[str, Any]): Model settings
        """
        return self.config[ConfigKeys.MODEL_PARAMETERS.value]

    def export_study_parameters(self) -> Dict[str, Any]:
        """
        Export the study parameters.

        Returns:
        - study_parameters (Dict[str, Any]): Study parameters
        """
        return self.config[ConfigKeys.STUDY_SETTINGS.value]

    def get_api_credentials(self) -> Dict[str, Any]:
        """
        Get the API credentials.

        Returns:
        - api_credentials (Dict[str, Any]): API credentials
        """
        return self.config[ConfigKeys.API_CREDENTIALS.value]

class ModelSettings:
    """Model settings class"""
    def __init__(self, model_parameters: Dict[str, Any]):
        """
        Initialize the model settings class.

        Args:
        - model_parameters (Dict[str, Any]): Model parameters
        """
        self.model_parameters = model_parameters

    def get_velocity_threshold(self) -> float:
        """
        Get the velocity threshold.

        Returns:
        - velocity_threshold (float): Velocity threshold
        """
        return self.model_parameters['velocity_threshold']

    def get_flow_theory_parameters(self) -> Dict[str, Any]:
        """
        Get the flow theory parameters.

        Returns:
        - flow_theory_parameters (Dict[str, Any]): Flow theory parameters
        """
        return self.model_parameters['flow_theory_parameters']

class StudySettings:
    """Study settings class"""
    def __init__(self, study_parameters: Dict[str, Any]):
        """
        Initialize the study settings class.

        Args:
        - study_parameters (Dict[str, Any]): Study parameters
        """
        self.study_parameters = study_parameters

    def get_study_name(self) -> str:
        """
        Get the study name.

        Returns:
        - study_name (str): Study name
        """
        return self.study_parameters['study_name']

    def get_study_description(self) -> str:
        """
        Get the study description.

        Returns:
        - study_description (str): Study description
        """
        return self.study_parameters['study_description']

class APICredentials:
    """API credentials class"""
    def __init__(self, api_credentials: Dict[str, Any]):
        """
        Initialize the API credentials class.

        Args:
        - api_credentials (Dict[str, Any]): API credentials
        """
        self.api_credentials = api_credentials

    def get_api_key(self) -> str:
        """
        Get the API key.

        Returns:
        - api_key (str): API key
        """
        return self.api_credentials['api_key']

    def get_api_secret(self) -> str:
        """
        Get the API secret.

        Returns:
        - api_secret (str): API secret
        """
        return self.api_credentials['api_secret']

def main():
    config = Config()
    config.load_environment_variables()
    config.validate_configurations()
    model_settings = ModelSettings(config.provide_model_settings())
    study_settings = StudySettings(config.export_study_parameters())
    api_credentials = APICredentials(config.get_api_credentials())
    logger.info(f"Model settings: {model_settings.model_parameters}")
    logger.info(f"Study settings: {study_settings.study_parameters}")
    logger.info(f"API credentials: {api_credentials.api_credentials}")

if __name__ == '__main__':
    main()