"""
Main Streamlit APP
"""

from pathlib import Path
from src import components


# data folder is in root folder
DATA_FOLDER = Path(__file__).parent.parent / "data"
CLASSIFIER_PATH = Path(__file__).parent / 'classifier.sav'


if __name__ == "__main__":

    # Data collection
    components.data_collection(DATA_FOLDER)

    # Classifier Training
    components.model_training(DATA_FOLDER, CLASSIFIER_PATH)

    # Inference
    components.inference(DATA_FOLDER, CLASSIFIER_PATH)
