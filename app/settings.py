from dataclasses import dataclass
from pathlib import Path

PROJECT_FOLDER = Path(__file__).parent.parent

@dataclass
class settings:
    DATA_PATH: str = PROJECT_FOLDER / "data"
    IMAGES_PATH: str = PROJECT_FOLDER / "images"
    MODEL_PATH: str = PROJECT_FOLDER / "classifier.sav"
    ASSETS_PATH: str = PROJECT_FOLDER / "assets"
