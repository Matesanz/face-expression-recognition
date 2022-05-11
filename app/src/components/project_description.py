import streamlit as st
from pathlib import Path


ASSETS_FOLDER = Path(__file__).parent.parent.parent.parent / "assets"


def project_description() -> None:
    st.title("😃 Face Emotion Recognition!")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(str(ASSETS_FOLDER / ("emotions/happy.png")))
    with col2:
        st.image(str(ASSETS_FOLDER / ("emotions/surprise.png")))
    with col3:
        st.image(str(ASSETS_FOLDER / ("emotions/angry.png")))