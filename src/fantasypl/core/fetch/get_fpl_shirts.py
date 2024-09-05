"""Functions for getting shirt graphics from FPL."""

import pandas as pd
import requests
import rich.progress
from loguru import logger

from fantasypl.config.constants import (
    DATA_FOLDER_FPL,
    FPL_BADGES_URL,
    FPL_SHIRTS_URL,
    RESOURCE_FOLDER,
)
from fantasypl.config.schemas import Season, Seasons
from fantasypl.utils import save_requests_response


def get_shirts(season: Season) -> None:
    """

    Parameters
    ----------
    season
        The season under process.

    """
    df_fpl_teams = pd.read_csv(DATA_FOLDER_FPL / season.folder / "teams.csv")
    team_codes = df_fpl_teams["code"].tolist()

    for code in rich.progress.track(
        team_codes, "Downloading shirt graphics: "
    ):
        url: str = f"{FPL_SHIRTS_URL}/shirt_{code}-220.png"
        response: requests.Response = requests.get(url, timeout=5)
        save_requests_response(
            response,
            RESOURCE_FOLDER / season.folder / "shirts" / f"shirt_{code}.png",
        )
        url = f"{FPL_SHIRTS_URL}/shirt_{code}_1-220.png"
        response = requests.get(url, timeout=5)
        save_requests_response(
            response,
            RESOURCE_FOLDER
            / season.folder
            / "shirts"
            / f"shirt_{code}_gk.png",
        )
        url = f"{FPL_BADGES_URL}/t{code}@x2.png"
        response = requests.get(url, timeout=5)
        save_requests_response(
            response,
            RESOURCE_FOLDER / season.folder / "badges" / f"badge_{code}.png",
        )
    logger.info("All shirt and badge graphics downloaded.")


if __name__ == "__main__":
    get_shirts(Seasons.SEASON_2425.value)
