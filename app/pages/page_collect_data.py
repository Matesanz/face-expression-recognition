"""
This Streamlit page is used to get inut from webcam, video or images
and categorize the face by feeling
"""

from pathlib import Path
from time import time
from cv2 import imwrite
import streamlit as st
from typing import Any
import mediapipe as mp
import av
import cv2
from streamlit_webrtc import webrtc_streamer
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
model = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
)


def draw_face_mesh_results(image: np.ndarray, results: Any) -> np.ndarray:
    """
    takes an image and draws over it the face mesh results of mediapipe
    face mesh processing

    Args:
        image (np.ndarray): raw image
        results (Any): face mesh results

    Returns:
        np.ndarray: raw image with face mesh plotted over it
    """

    annotated_image = image.copy()

    if not results.multi_face_landmarks:
        return annotated_image

    for face_landmarks in results.multi_face_landmarks:

        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
        )
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
        )

        return annotated_image


def page() -> None:
    """
    Renders main part of the page when webcam input is selected
    """

    feelings = ["happy", "sad", "surprised", "angry", "normal"]
    selected_feeling = st.selectbox("I want to label this image as", feelings)

    data_folder = Path(__file__).parent.parent.parent / "data"
    feeling_folder: Path = data_folder / selected_feeling
    if not feeling_folder.exists():
        feeling_folder.mkdir(parents=True)

    class VideoProcessor:
        """
        class used by streamer to process real time frames
        """

        def recv(self, frame):
            """
            process frame
            """
            image = frame.to_ndarray(format="bgr24")
            results = model.process(image)

            if not results.multi_face_landmarks:
                return image

            # convert face mesh landmarks to array
            face = results.multi_face_landmarks[0]
            landmarks = np.array([(l.x, l.y, l.z) for l in face.landmark])

            now = str(time())

            ## save image
            # image_name = now + ".jpg"
            # img_path = feeling_folder / image_name
            # cv2.imwrite(str(img_path), image)

            # save normalized landmarks
            landmarks_name = now + ".npy"
            landmarks_path = feeling_folder / landmarks_name
            norm_landmarks = landmarks / np.linalg.norm(landmarks)
            np.save(landmarks_path, norm_landmarks)

            # show image with results in streamlit page
            image = draw_face_mesh_results(image, results)
            return av.VideoFrame.from_ndarray(image, format="bgr24")

    webrtc_streamer("webcam", video_processor_factory=VideoProcessor)
    st.text(f"Images are being labeled as {selected_feeling}")
