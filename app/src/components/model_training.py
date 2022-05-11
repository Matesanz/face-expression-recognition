"""
Load Model Streamlit Component
"""
from typing import List, Union
from pathlib import Path
import streamlit as st
from sklearn.metrics import accuracy_score
import pickle
from sklearn.neighbors import KNeighborsClassifier

from .. import utils


def model_training(data_folder: Union[str, Path], model_path: Union[str, Path]) -> None:
    """_summary_

    Args:
        data_folder (Union[str, Path]): _description_
    """

    # Train Component
    st.title("Training KMeans Classifier")

    trainable = False

    emotions_paths: List[Path] = []
    for path in data_folder.iterdir():

        path_files = list(path.iterdir())
        path_is_empty = len(path_files) == 0

        # ignore files and empty folders
        if path.is_file() or path_is_empty:
            continue
        emotions_paths.append(path)

    if not emotions_paths:
        st.warning(
            "⚠️ There is no data to train the model on." "Please, collect more data"
        )

    elif len(emotions_paths) == 1:
        st.warning(
            f"⚠️ Just one emotion ({emotions_paths[0].stem}) was detected. "
            "Please, collect more data in order to train the model on "
            "different face expressions"
        )

    else:
        emotions = [p.stem for p in emotions_paths]
        st.success(
            f"Model can be trained to detect {len(emotions)} "
            f"emotions: {', '.join(emotions)}"
        )
        trainable = False

    col1, col2 = st.columns(2)
    with col1:
        train_button = st.button("Train", disabled=trainable)
    if train_button:

        x_train, x_test, y_train, y_test = utils.load_dataset(data_folder)
        classifier = KNeighborsClassifier(n_neighbors=len(emotions))
        with col2:
            with st.spinner("Training Model"):
                classifier.fit(x_train, y_train)

        # Initialization
        # save the model to disk
        model_path = Path(model_path)
        if model_path.exists():
            model_path.unlink()

        pickle.dump(classifier, open(model_path, "wb"))

        y_pred = classifier.predict(x_test)
        acc = accuracy_score(y_test, y_pred)

        st.success(f"✅ Model Trained with an accuracy of {acc*100}%!")
