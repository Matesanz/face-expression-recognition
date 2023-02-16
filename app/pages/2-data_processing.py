from pathlib import Path
import shutil

import cv2
import numpy as np
import streamlit as st

from app import utils, explainers
from app.settings import settings

IMAGES_FOLDER = settings.IMAGES_PATH
DATA_FOLDER = settings.DATA_PATH

Path(IMAGES_FOLDER).mkdir(parents=True, exist_ok=True)
Path(DATA_FOLDER).mkdir(parents=True, exist_ok=True)


st.title("2. Data Processing")
explainers.explain_data_processing_stage()

feelings = [p for p in Path(IMAGES_FOLDER).iterdir() if p.is_dir()]
if not feelings:
    st.error(
        "No feelings found. Upload a zipped folder with "
        "images in the data collection Section."
    )
else:
    st.success(f"Found {len(feelings)} feelings: {', '.join([f.name for f in feelings])}")
    # Get landmarks from all images in subfolders
    for imgs_feeling_folder in Path(IMAGES_FOLDER).iterdir():

        if not imgs_feeling_folder.is_dir():
            continue

        # create folder for landmarks in data folder
        feeling_folder_data = Path(DATA_FOLDER) / imgs_feeling_folder.name
        feeling_folder_data.mkdir(exist_ok=True, parents=True)

        images = []
        for image_path in Path(imgs_feeling_folder).iterdir():
            if not image_path.is_file():
                continue

            # get face landmarks from frame
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = utils.get_face_results(image)
            landmarks = utils.get_results_landmarks(results)
            norm_landmarks = utils.project_face_landmarks(landmarks)

            # save normalized landmarks
            landmarks_path = feeling_folder_data / (image_path.stem + ".npy")
            np.save(landmarks_path, norm_landmarks)

            image = utils.draw_face_mesh_results(image, results)
            if len(images) < 4:
                images.append(image)

        # plot landmarks
        st.subheader(f"{imgs_feeling_folder.name}")
        st.image(images, width=300)
