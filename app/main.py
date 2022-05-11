"""
Main Streamlit APP
"""

from pathlib import Path
from src import components
import streamlit as st

# data folder is in root folder
DATA_FOLDER = Path(__file__).parent.parent / "data"
ASSETS_FOLDER = Path(__file__).parent.parent / "assets"
CLASSIFIER_PATH = Path(__file__).parent / 'classifier.sav'


if __name__ == "__main__":

    # Project Description
    components.project_description()

    # Data collection
    components.data_collection(DATA_FOLDER)
    components.explainers.explain_face_plane()
    components.explainers.explain_face_mesh_code()
    components.explainers.explain_project_landmarks_code()


    # Classifier Training
    components.model_training(DATA_FOLDER, CLASSIFIER_PATH)
    components.explainers.explain_load_train_data_code()
    components.explainers.explain_train_kmeans_code()

    # Inference
    components.inference(DATA_FOLDER, CLASSIFIER_PATH)
    components.explainers.explain_load_save_model_code()
    components.explainers.explain_inference_code()
