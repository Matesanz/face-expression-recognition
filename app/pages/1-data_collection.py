"""
This Streamlit page is used to get inut from webcam, video or images
and categorize the face by feeling
"""

from pathlib import Path

import cv2
import streamlit as st
import zipfile

from app.settings import settings
from app import explainers

data_folder = settings.IMAGES_PATH
st.title("1. Data Collection")
explainers.explain_data_collection_stage()

# upload folder with images
data_zip = st.file_uploader(
    "Upload zip with folders representing feelings to be detected.",
    type="zip",
)
Path(data_folder).mkdir(exist_ok=True, parents=True)

# unzip subfolders with images
if data_zip is not None:
    with zipfile.ZipFile(data_zip) as zip_ref:
        zip_ref.extractall(data_folder)

# show loaded feelings: every folder in data folder is a feeling
feelings = [p for p in Path(data_folder).iterdir() if p.is_dir()]
if feelings:
    st.success(f"Found {len(feelings)} feelings: {', '.join([f.name for f in feelings])}")
else:
    st.error("No feelings found. Upload a zipped folder with images.")

# show images in each feeling
for feeling_folder in Path(data_folder).iterdir():

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
