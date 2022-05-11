import cv2
import pickle
import streamlit as st
from typing import Union
from pathlib import Path
from sklearn import datasets
import av
from streamlit_webrtc import webrtc_streamer
from .. import utils


def inference(
    data_folder: Union[str, Path],
    classifier_path: Union[str, Path],
) -> None:

    st.title("Inference")

    inferenceable = classifier_path.exists()
    if not inferenceable:
        st.info("No model found")

    else:
        classifier = pickle.load(open(classifier_path, "rb"))
        st.success("Loaded Model")

        class VideoProcessorInference:
            """
            class used by streamer to process real time frames
            """

            def recv(self, frame):
                """
                process frame
                """
                image = frame.to_ndarray(format="bgr24")
                results = utils.get_face_results(image)
                if results is None:
                    # return cam frame if no face found
                    return av.VideoFrame.from_ndarray(image, format="bgr24")

                # convert face mesh landmarks to array
                landmarks = utils.get_results_landmarks(results)

                norm_landmarks = utils.project_face_landmarks(landmarks)
                norm_landmarks = norm_landmarks[:, :2].ravel()

                dataset = datasets.load_files(data_folder, load_content=False)
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
                image = utils.draw_face_mesh_results(image, results)
                return av.VideoFrame.from_ndarray(image, format="bgr24")

        if inferenceable:
            webrtc_streamer(
                "webcam_inference", video_processor_factory=VideoProcessorInference
            )
