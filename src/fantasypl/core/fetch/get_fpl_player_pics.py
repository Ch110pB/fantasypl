"""Functions for getting player photos from FPL."""

import pandas as pd
import requests
import rich.progress
from loguru import logger

from fantasypl.config.constants import (
    DATA_FOLDER_FPL,
    FPL_PHOTOS_URL,
    RESOURCE_FOLDER,
)
from fantasypl.config.schemas import Season, Seasons
from fantasypl.utils import save_requests_response


def get_player_photos(season: Season) -> None:
    """
    Get FPL player photos.

    Parameters
    ----------
    season
        The season under process.

    """
    df_fpl_players: pd.DataFrame = pd.read_csv(
        DATA_FOLDER_FPL / season.folder / "players.csv"
    )
    player_codes: list[str] = df_fpl_players["code"].to_list()

    for code in rich.progress.track(
        player_codes,
        "Downloading player photos: ",
    ):
        url: str = f"{FPL_PHOTOS_URL}/p{code}.png"
        response: requests.Response = requests.get(url, timeout=2)
        save_requests_response(
            response,
            RESOURCE_FOLDER / season.folder / "photos" / f"photo_{code}.png",
        )
    logger.info("All player photos downloaded.")


if __name__ == "__main__":
    get_player_photos(Seasons.SEASON_2425.value)
