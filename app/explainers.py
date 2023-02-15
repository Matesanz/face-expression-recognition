import streamlit as st
from PIL import Image
from typing import Union
from pathlib import Path
import numpy as np


ASSETS_FOLDER = Path(__file__).parent.parent.parent.parent / "assets"
INFO_ASSETS_FOLDER = ASSETS_FOLDER / "info"
SNIPPETS_ASSETS_FOLDER = ASSETS_FOLDER / "snippets"


def explain_image(
    image_path: Union[str, Path],
    expander_title: str,
    caption: str
) -> None:

    with st.expander(expander_title):
        image_path = str(image_path)
        image = np.asarray(Image.open(image_path))
        st.image(image, caption=caption)


def explain_face_plane() -> None:

    caption = """
    A Plane can be formed by two vectors and a point.
    Face plane is defined by the eye-to-eye vector, the
    frown-lip vector, and the nose point.

    To normalize data 3D landmarks are projected into the
    face plane and then rotated and normalized to the image plane.
    """
    img_path = INFO_ASSETS_FOLDER / "face_plane.png"
    explain_image(
        image_path=img_path,
        expander_title="ℹ️ What is the Face Plane?",
        caption=caption
        )


def explain_face_mesh_code() -> None:


    caption = """
    Using mediapipe we can easilly use powerfull pretrained
    face mesh models.
    """
    img_path = SNIPPETS_ASSETS_FOLDER / "code_face_mesh.png"
    explain_image(
        image_path=img_path,
        expander_title="ℹ️ Python Code: How can I detect the Face Mesh?",
        caption=caption
        )


def explain_project_landmarks_code() -> None:


    caption = """
    3D landmarks are project into the face plane. Removing its depth 
    as it is not crutial to detect face emotions. The points are rotated
    an normalized into the camera plane. Results of this geometry problem can
    be seen in the Face Plane explanation.
    """
    img_path = SNIPPETS_ASSETS_FOLDER / "code_project_points.png"
    explain_image(
        image_path=img_path,
        expander_title="ℹ️ Python Code: How are 3D points projected into camera plane?",
        caption=caption
        )


def explain_load_save_model_code() -> None:


    caption = """
    Scikit-learn models can be easily saved loaded using pickle.
    """
    img_path = SNIPPETS_ASSETS_FOLDER / "code_load_save_model.png"
    explain_image(
        image_path=img_path,
        expander_title="ℹ️ Python Code: How to load/save sklearn models?",
        caption=caption
        )

def explain_load_train_data_code() -> None:

    caption = """
    Scikit-learn models can be easily saved loaded using pickle.
    """
    img_path = SNIPPETS_ASSETS_FOLDER / "code_load_train_data.png"
    explain_image(
        image_path=img_path,
        expander_title="ℹ️ Python Code: How to load data to train/test models?",
        caption=caption
        )

def explain_train_kmeans_code() -> None:

    caption = """
    Training models in sklearn is simple, fast and reliable.
    """
    img_path = SNIPPETS_ASSETS_FOLDER / "code_train_kmeans.png"
    explain_image(
        image_path=img_path,
        expander_title="ℹ️ Python Code: How to load data to train/test models?",
        caption=caption
        )

def explain_inference_code() -> None:

    caption = """
    Full inference code, including data preparation.
    """
    img_path = SNIPPETS_ASSETS_FOLDER / "code_inference.png"
    explain_image(
        image_path=img_path,
        expander_title="ℹ️ Python Code: Show full inference snippet",
        caption=caption
        )

def explain_data_collection_stage() -> None:
    """
    Explain the data collection stage
    """
    st.markdown(
        """
        First step is to collect data to train the model on.
        Data is a collection of images of faces with different emotions.
        Images of the same emotion are placed in the same folder.
        """
    )

def explain_data_processing_stage() -> None:
    """
    Explain the data processing stage
    """
    st.markdown(
        """
        Second step is to extract features from the images.
        Features are the **coordinates of the landmarks on the face**.
        To extract the features, we use the 
        [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html)
        model.
        """
    )
    
    explain_face_mesh_code()

    st.markdown(
        """
        Once we have the landmarks, we **normalize** them. Normalization is done by
        projecting the landmarks to a **2D plane**. The plane is defined by the
        eye-to-eye vector, the frown-lip vector, and the nose point.
        """
    )

    explain_face_plane()
    explain_project_landmarks_code()

def explain_model_training_stage() -> None:
    """
    Explain the model training stage
    """
    st.markdown(
        """
        The last step is to **train the classifier**. We use the
        [K-Means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
        algorithm to cluster the data into different emotions.
        Landmarks of the same emotion are clustered together.
        """
    )
    explain_load_train_data_code()
    explain_train_kmeans_code()

def explain_model_prediction_stage() -> None:
    """
    Explain the model prediction stage
    """


    st.markdown(
        """
        Now that we have trained the model, we can use it to perform inference.
        Inference is the process of predicting the emotion of a face.
        You can use your webcam to **perform inference on your face in REAL TIME!**
        """
    )
    explain_load_save_model_code()
    explain_inference_code()
