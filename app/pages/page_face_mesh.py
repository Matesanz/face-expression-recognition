"""
This Streamlit page is used to get inut from webcam, video or images
and categorize the face by feeling
"""

import streamlit as st
from typing import Any
import mediapipe as mp
import av
import cv2
from streamlit_webrtc import webrtc_streamer
import numpy as np
from skspatial.objects import Line, Plane


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
                return av.VideoFrame.from_ndarray(image, format="bgr24")

            # convert face mesh landmarks to array
            h, w = image.shape[:2]
            face = results.multi_face_landmarks[0]
            landmarks = np.array([(l.x, l.y, l.z) for l in face.landmark])

            lip_point = landmarks[0]
            nose_point = landmarks[1]
            frown_point = landmarks[9]
            right_eye_point = landmarks[33]
            left_eye_point = landmarks[263]

            eye_line = Line.from_points(left_eye_point, right_eye_point)
            lip_frown_line = Line.from_points(lip_point, frown_point)

            face_plane = Plane.from_vectors(
                nose_point,
                lip_frown_line.vector,
                eye_line.vector,
            )

            # https://stackoverflow.com/questions/8942950/how-do-i-find-the-orthogonal-projection-of-a-point-onto-a-plane
            v = landmarks - nose_point
            dist = v.dot(face_plane.vector.unit())
            proj_landmarks = landmarks - (
                face_plane.vector.unit() * dist[:, np.newaxis]
            )

            perpendicular_vector = np.cross(
                face_plane.vector.unit(), eye_line.vector.unit()
            )
            rot_matrix = np.array(
                [
                    eye_line.vector.unit(),
                    perpendicular_vector,
                    face_plane.vector.unit(),
                ]
            )

            front_landmarks = proj_landmarks - nose_point
            front_landmarks = front_landmarks.dot(rot_matrix.T)

            for x, y, z in front_landmarks:
                offset = 90
                landmark_coords = int(x * w + offset), int(y * h + offset)
                cv2.circle(image, landmark_coords, 2, (255, 0, 0), -1)

            return av.VideoFrame.from_ndarray(image, format="bgr24")

    webrtc_streamer("webcam", video_processor_factory=VideoProcessor)
