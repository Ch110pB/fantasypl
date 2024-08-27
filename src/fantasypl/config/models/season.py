"""Contains the Season class and the Seasons enums."""

from enum import Enum

from pydantic import BaseModel


class Season(BaseModel):
    """
    The Season class.

    Attributes
    ----------
        folder: Folder marker for season.
        fbref_name: Season short name in FBRef.
        fbref_long_name: Season full name in FBRef.

    """

    folder: str
    fbref_name: str
    fbref_long_name: str


class Seasons(Enum):
    """
    The Seasons enums.

    Contains:
        The Season enums for different seasons.
    """

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
