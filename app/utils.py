from pathlib import Path
from typing import Any, Tuple, Union
import numpy as np
import mediapipe as mp
from skspatial.objects import Line, Plane
from sklearn import datasets
from sklearn.model_selection import train_test_split


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
model = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
)


def project_face_landmarks(
    landmarks: np.ndarray,
    lip_index: int = 0,
    nose_index: int = 1,
    frown_index: int = 9,
    right_eye_index: int = 33,
    left_eye_index: int = 263,
) -> np.ndarray:
    """
    Project 3D landmark points into camera plane.
    Converting the world position of the landmarks to its
    normalized position [0, 1) in the ortographic projection of
    the face plane. Face plane is the one given by the eyes vector,
    the lip-frown vector passing throught the nose point.

    Args:
        landmarks (np.ndarray): array of landmark coordinates. Shape (N, 3), being
            N total number of landmarks and theit x,y,z 3D coordinates.
        lip_index (int, optional): landmark index of the center of the bottom lip.
            Defaults to 0.
        nose_index (int, optional): landmark index of the nose.
            Defaults to 1.
        frown_index (int, optional): landmark index of the center of the frown.
            Defaults to 9.
        right_eye_index (int, optional): landmark right eye's outer part.
            Defaults to 33.
        left_eye_index (int, optional): landmark left eye's outer part.
            Defaults to 263.

    Returns:
        np.ndarray: proyected normalized 2D key keypoints. shape (N, 3).
            N for every landmark and their x, y, x coordinates. x and y
            are normalized [0, 1) pixel coordinates. z is the normalized
            depth of the landmark.
    """
    lip_point = landmarks[lip_index]
    nose_point = landmarks[nose_index]
    frown_point = landmarks[frown_index]
    right_eye_point = landmarks[right_eye_index]
    left_eye_point = landmarks[left_eye_index]

    eye_line = Line.from_points(left_eye_point, right_eye_point)
    lip_frown_line = Line.from_points(lip_point, frown_point)

    face_plane = Plane.from_vectors(
        nose_point,
        lip_frown_line.vector,
        eye_line.vector,
    )

    # Project 3D points into face plane
    # https://stackoverflow.com/questions/8942950/how-do-i-find-the-orthogonal-projection-of-a-point-onto-a-plane
    landmarks_relative_vectors = landmarks - nose_point
    distances_to_face_plane = landmarks_relative_vectors.dot(face_plane.vector.unit())
    proj_landmarks = landmarks - (
        face_plane.vector.unit() * distances_to_face_plane[:, np.newaxis]
    )

    # tangential vector to face plane normal and eyes vector
    # similar to lip-frown vector, but perfectly ortogonal
    perpendicular_vector = np.cross(face_plane.vector.unit(), eye_line.vector.unit())

    # Compute head rotation
    rot_matrix = np.array(
        [
            eye_line.vector.unit(),
            perpendicular_vector,
            face_plane.vector.unit(),
        ]
    )

    # project landmarks in face plane to the Z plane (a.k.a. camera plane)
    front_landmarks = proj_landmarks - nose_point
    front_landmarks = front_landmarks.dot(rot_matrix.T)

    # normalize x, y and z landmarks into range 0-1
    norm_landmarks = front_landmarks - front_landmarks.min(axis=0)
    norm_landmarks /= norm_landmarks.max(axis=0)
    return norm_landmarks[:, :2]  # return x, y


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


def load_dataset(
    data_path: Union[str, Path],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    loads RandomForestClassifier

    Returns:
        Any: _description_
    """
    dataset = datasets.load_files(data_path, load_content=False)

    # Load Dataset
    x = np.array([np.load(f).ravel() for f in dataset.filenames])
    y = np.array(dataset.target)
    return train_test_split(x, y)


def get_face_results(image: np.ndarray) -> Union[Any, None]:
    """
    return face landmarks with shape (480, 3) with the mediapipe 3D
    landmark's coordinatesif face is found, otherwise return None.

    Args:
        image (np.ndarray): Image of a face

    Returns:
        Union[np.ndarray, None]: x, y, z coordinates of landmarks if
            face is found, otherwise returns None.
    """
    results = model.process(image)

    if not results.multi_face_landmarks:
        return None
    else:
        return results


def get_results_landmarks(results: Any) -> np.ndarray:
    """_summary_

    Args:
        results (Any): _description_

    Raises:
        ValueError: _description_

    Returns:
        np.ndarray: _description_
    """

    if not results.multi_face_landmarks:
        raise ValueError("No landmarks found in results")

    # convert face mesh landmarks to array
    face = results.multi_face_landmarks[0]  # get first and only face
    return np.array([(l.x, l.y, l.z) for l in face.landmark])
