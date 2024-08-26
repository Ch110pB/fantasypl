from enum import Enum

from pydantic import BaseModel


class Season(BaseModel):
    folder: str
    fbref_name: str
    fbref_long_name: str


class Seasons(Enum):
    SEASON_2324 = Season(
        folder="2324",
        fbref_name="2023-24",
        fbref_long_name="2023-2024",
    )
    SEASON_2425 = Season(
        folder="2425",
        fbref_name="2024-25",
        fbref_long_name="2024-2025",
    )
