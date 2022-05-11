"""
This Streamlit page is used to get inut from webcam, video or images
and categorize the face by feeling
"""

from typing import Union
from pathlib import Path
from time import time

import streamlit as st
import av
import cv2

from streamlit_webrtc import webrtc_streamer
import numpy as np

from .. import utils


def data_collection(data_folder: Union[str, Path]) -> None:
    """
    Renders main part of the page when webcam input is selected
    """

    st.title("Data Collection")
    feelings = ["happy", "sad", "surprised", "angry", "wink", "normal"]

    selected_feeling = st.selectbox("I want to label this image as", feelings)
    save_data = st.checkbox("Save data")

    data_folder = Path(data_folder)  # force data_folder to be of type Path
    feeling_folder: Path = data_folder / selected_feeling
    if not feeling_folder.exists():
        feeling_folder.mkdir(parents=True)

    class VideoProcessorDetection:
        """
        class used by streamer to process real time frames
        """

        def recv(self, frame):
            """
            process frame
            """
            # get face landmarks from frame
            image = frame.to_ndarray(format="bgr24")
            results = utils.get_face_results(image)
            if results is None:
                # return cam frame if no face found
                return av.VideoFrame.from_ndarray(image, format="bgr24")

            landmarks = utils.get_results_landmarks(results)
            norm_landmarks = utils.project_face_landmarks(landmarks)
            # show image with results in streamlit page

            # save normalized landmarks
            if save_data:
                now = str(time())
                landmarks_name = now + ".npy"
                landmarks_path = feeling_folder / landmarks_name
                np.save(landmarks_path, norm_landmarks)

            image = utils.draw_face_mesh_results(image, results)
            # plot landmarks
            side = 150
            offset = 80
            for x, y in norm_landmarks:
                landmark_coords = int(x * side + offset), int(y * side + offset)
                cv2.circle(image, landmark_coords, 2, (255, 0, 0), -1)
            return av.VideoFrame.from_ndarray(image, format="bgr24")

    webrtc_streamer(
        "detection", 
        video_processor_factory=VideoProcessorDetection,
        rtc_configuration={  # streamlit cloud
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })
