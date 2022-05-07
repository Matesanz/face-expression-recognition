"""
Main Streamlit APP
"""

from pages import page_collect_data, page_inference, page_face_mesh
import streamlit as st

if __name__ == "__main__":

    data_colletion_page_name = "Data Collection"
    inference_page_name = "Inference"
    face_mesh_page_name = "Face Mesh"

    with st.sidebar:
        page = st.radio(
            "Go to",
            [
                face_mesh_page_name,
                data_colletion_page_name,
                inference_page_name,
            ],
        )

    if page == face_mesh_page_name:

        page_face_mesh.page()

    elif page == data_colletion_page_name:

        page_collect_data.page()

    elif page == inference_page_name:

        page_inference.page()

    else:

        page_face_mesh.page()
