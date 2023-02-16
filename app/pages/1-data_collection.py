"""
This Streamlit page is used to get inut from webcam, video or images
and categorize the face by feeling
"""

from pathlib import Path
import shutil

import cv2
import streamlit as st
import zipfile

from app.settings import settings
from app import explainers

DATA_FOLDER = settings.IMAGES_PATH
IMAGES_FOLDER = settings.IMAGES_PATH

Path(DATA_FOLDER).mkdir(exist_ok=True, parents=True)
st.title("1. Data Collection")
explainers.explain_data_collection_stage()

clear_data = st.button("Clear Data")
if clear_data:
    # remove all files in data folder
    shutil.rmtree(DATA_FOLDER)
    Path(DATA_FOLDER).mkdir(parents=True, exist_ok=True)

    shutil.rmtree(IMAGES_FOLDER)
    Path(IMAGES_FOLDER).mkdir(parents=True, exist_ok=True)

# upload folder with images
data_zip = st.file_uploader(
    "Upload zip with folders representing feelings to be detected.",
    type="zip",
)

# unzip subfolders with images
if data_zip is not None:
    with zipfile.ZipFile(data_zip) as zip_ref:
        zip_ref.extractall(DATA_FOLDER)

# show loaded feelings: every folder in data folder is a feeling
feelings = [p for p in Path(DATA_FOLDER).iterdir() if p.is_dir()]
if feelings:
    st.success(f"Found {len(feelings)} feelings: {', '.join([f.name for f in feelings])}")
else:
    st.error("No feelings found. Upload a zipped folder with images.")

# show images in each feeling
for feeling_folder in Path(DATA_FOLDER).iterdir():

    images = []
    for image_path in feeling_folder.iterdir():

        # ignore folders
        if not image_path.is_file():
            continue

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)

    st.subheader(f"{feeling_folder.name}")
    st.image(images, width=200)
