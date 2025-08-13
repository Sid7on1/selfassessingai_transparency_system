import logging
import os
from typing import List, Dict, Tuple, Union

import pandas as pd
import requests
from requests.exceptions import RequestException
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UserStudyPlatform:
    """
    Integration layer for Prolific recruitment and Qualtrics survey data collection.

    ...

    Attributes
    ----------
    api_keys : Dict[str, str]
        Dictionary containing API keys for Prolific and Qualtrics.
    headers : Dict[str, str]
        Dictionary containing request headers for API calls.
    prolific_url : str
        Base URL for the Prolific API.
    qualtrics_url : str
        Base URL for the Qualtrics API.

    Methods
    -------
    create_study_sessions(num_sessions, participants_per_session)
        Create study sessions on Prolific and return session IDs.
    assign_conditions(session_ids, conditions)
        Assign conditions to each study session.
    collect_responses(session_ids)
        Collect survey responses from Qualtrics for the given session IDs.
    export_study_data(session_ids, output_file)
        Export study data to a CSV file for further analysis.
    """

    def __init__(self, api_keys: Dict[str, str], prolific_url: str, qualtrics_url: str):
        """
        Initialize the UserStudyPlatform class with API credentials and URLs.

        Parameters
        ----------
        api_keys : Dict[str, str]
            Dictionary containing API keys for Prolific and Qualtrics.
        prolific_url : str
            Base URL for the Prolific API.
        qualtrics_url : str
            Base URL for the Qualtrics API.
        """
        self.api_keys = api_keys
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_keys["prolific"]}'
        }
        self.prolific_url = prolific_url
        self.qualtrics_url = qualtrics_url

    def _make_api_request(self, url: str, method: str = 'GET', data: Dict = None) -> Dict:
        """
        Make an API request and handle potential errors.

        Parameters
        ----------
        url : str
            The API endpoint URL.
        method : str, optional
            The HTTP method for the request (default is 'GET').
        data : Dict, optional
            Data to be sent in the request body (default is None).

        Returns
        -------
        Dict
            The JSON response from the API.

        Raises
        ------
        RequestException
            If an error occurs during the API request.
        """
        try:
            response = requests.request(method, url, headers=self.headers, json=data)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    def create_study_sessions(self, num_sessions: int, participants_per_session: int) -> List[str]:
        """
        Create study sessions on Prolific and return session IDs.

        Parameters
        ----------
        num_sessions : int
            Number of study sessions to create.
        participants_per_session : int
            Number of participants required for each session.

        Returns
        -------
        List[str]
            List of session IDs.

        Raises
        ------
        ValueError
            If the number of sessions or participants per session is invalid.
        APIError
            If an error occurs during the API request.
        """
        if num_sessions < 1 or participants_per_session < 1:
            raise ValueError("Number of sessions and participants per session must be at least 1.")

        logger.info("Creating study sessions on Prolific...")
        session_data = {
            "prescreening": {
                "country": "US"
            },
            "participation": {
                "duration_in_seconds": 3600,
                "completion_rate": {"minimum": "95"},
                "approvals": {"minimum": 100}
            },
            "num_participants": participants_per_session,
            "study_image_url": "https://example.com/study_image.png",
            "study_description": "Description of the user study",
            "reward": {"amount": 5, "currency": "USD"}
        }

        session_ids = []
        for i in range(num_sessions):
            session_data["name"] = f"Session {i + 1}"
            response = self._make_api_request(self.prolific_url + "/studies", "POST", session_data)
            session_ids.append(response["id"])

        logger.info(f"Created {num_sessions} study sessions on Prolific.")
        return session_ids

    def assign_conditions(self, session_ids: List[str], conditions: List[Dict]) -> None:
        """
        Assign conditions to each study session.

        Parameters
        ----------
        session_ids : List[str]
            List of session IDs.
        conditions : List[Dict]
            List of conditions, each specified as a dictionary.

        Raises
        ------
        ValueError
            If the length of session IDs and conditions does not match.
        APIError
            If an error occurs during the API request.
        """
        if len(session_ids) != len(conditions):
            raise ValueError("Number of session IDs and conditions must match.")

        logger.info("Assigning conditions to study sessions...")
        for session_id, condition in zip(session_ids, conditions):
            response = self._make_api_request(f"{self.prolific_url}/studies/{session_id}/data-uploads", "POST", condition)

        logger.info("Successfully assigned conditions to study sessions.")

    def collect_responses(self, session_ids: List[str]) -> pd.DataFrame:
        """
        Collect survey responses from Qualtrics for the given session IDs.

        Parameters
        ----------
        session_ids : List[str]
            List of session IDs.

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame containing survey responses.

        Raises
        ------
        APIError
            If an error occurs during the API request.
        """
        logger.info("Collecting survey responses from Qualtrics...")
        responses_data = []
        for session_id in session_ids:
            response = self._make_api_request(f"{self.qualtrics_url}/surveys/{session_id}/responses")
            responses_data.extend(response["results"])

        responses_df = pd.json_normalize(responses_data)
        logger.info("Successfully collected survey responses.")
        return responses_df

    def export_study_data(self, session_ids: List[str], output_file: str) -> None:
        """
        Export study data to a CSV file for further analysis.

        Parameters
        ----------
        session_ids : List[str]
            List of session IDs.
        output_file : str
            Path to the output CSV file.

        Raises
        ------
        ValueError
            If the output file path is invalid.
        """
        if not output_file.endswith(".csv"):
            raise ValueError("Output file must be a CSV file.")

        responses_df = self.collect_responses(session_ids)
        logger.info(f"Exporting study data to {output_file}...")
        responses_df.to_csv(output_file, index=False)
        logger.info("Successfully exported study data.")

# Example usage
if __name__ == "__main__":
    api_keys = {
        "prolific": os.getenv("PROLIFIC_API_KEY"),
        "qualtrics": os.getenv("QUALTRICS_API_KEY")
    }
    prolific_url = "https://api.prolific.co/v1"
    qualtrics_url = "https://api.qualtrics.com"

    platform = UserStudyPlatform(api_keys, prolific_url, qualtrics_url)

    # Create study sessions
    num_sessions = 3
    participants_per_session = 20
    session_ids = platform.create_study_sessions(num_sessions, participants_per_session)
    print("Created session IDs:", session_ids)

    # Assign conditions (example conditions)
    conditions = [
        {"condition": "control"},
        {"condition": "treatment1"},
        {"condition": "treatment2"}
    ]
    platform.assign_conditions(session_ids, conditions)
    print("Assigned conditions to sessions.")

    # Collect and export responses
    output_file = "study_data.csv"
    platform.export_study_data(session_ids, output_file)
    print(f"Exported study data to {output_file}.")