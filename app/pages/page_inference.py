import streamlit as st
from sklearn import datasets
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from streamlit_webrtc import webrtc_streamer
import mediapipe as mp
import av
import cv2
from typing import Any


data_path = Path(__file__).parent.parent.parent / "data"
dataset = datasets.load_files(data_path, load_content=False)
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


def load_model() -> Any:

    # Load Dataset
    x = np.array([np.load(f).ravel() for f in dataset.filenames])
    y = np.array(dataset.target)

    # Train/Test Split

    x_train, x_test, y_train, y_test = train_test_split(x, y)
    classifier = RandomForestClassifier()
    classifier.fit(x_train, y_train)

    return classifier


classifier = load_model()


def page() -> None:
    """
    Inference Page
    """
    st.title("Doing Inference")

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
            landmarks = np.array([(l.x, l.y, l.z) for l in face.landmark]).ravel()
            norm_landmarks = landmarks / np.linalg.norm(landmarks)

            classification = classifier.predict([norm_landmarks])
            feeling = dataset.target_names[classification.max()]

            image = cv2.putText(
                image,
                feeling,
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.FONT_HERSHEY_SIMPLEX,
            )
            # show image with results in streamlit page
            image = draw_face_mesh_results(image, results)
            return av.VideoFrame.from_ndarray(image, format="bgr24")

    webrtc_streamer("webcam", video_processor_factory=VideoProcessor)
