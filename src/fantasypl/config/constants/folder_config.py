"""Constants for absolute folder paths."""

from pathlib import Path


ROOT_FOLDER: Path = Path(__file__).parents[4].absolute()
DATA_FOLDER_FPL: Path = ROOT_FOLDER / "data" / "fpl"
DATA_FOLDER_FBREF: Path = ROOT_FOLDER / "data" / "fbref"
DATA_FOLDER_REF: Path = ROOT_FOLDER / "data" / "references"
MODEL_FOLDER: Path = ROOT_FOLDER / "models"
