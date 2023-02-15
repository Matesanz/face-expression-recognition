"""
Main Streamlit APP
"""

import streamlit as st
from app.settings import settings
from app import explainers

st.set_page_config(
    page_title="Face Emotion Recognition!",
    page_icon="😃",
)

# Project Description
ASSETS_FOLDER = settings.ASSETS_PATH

st.title("😃 Face Emotion Recognition!")
col1, col2, col3 = st.columns(3)
with col1:
    st.image(str(ASSETS_FOLDER / ("emotions/happy.png")))
with col2:
    st.image(str(ASSETS_FOLDER / ("emotions/surprise.png")))
with col3:
    st.image(str(ASSETS_FOLDER / ("emotions/angry.png")))

st.markdown(
    """
    **In this project, we will build a classifier that can detect emotions from faces.
    First we will collect data, then we will train a classifier, and finally we will
    use our classifier to perform inference on a webcam stream.**

    ⚡ Everything will be running **REAL TIME** using only **CPU** resources.
    """
)

# Data collection
st.subheader("1. Data Collection")
explainers.explain_data_collection_stage()

st.subheader("2. Data Processing")
explainers.explain_data_processing_stage()

# Classifier Training
st.subheader("3. Tranining the Classifier")
explainers.explain_model_training_stage()

# Inference
st.subheader("4. Perform Inference using your Webcam")
explainers.explain_model_prediction_stage()
