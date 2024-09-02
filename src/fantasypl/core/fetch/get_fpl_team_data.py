"""Functions for getting my last gameweek team and transfers data."""

import requests
from loguru import logger

from fantasypl.config.constants.folder_config import MODEL_FOLDER
from fantasypl.config.constants.web_config import FPL_TEAM_URL
from fantasypl.utils.save_helper import save_json


def get_my_transfers(team_id: int, gameweek: int) -> None:
    """

    Args:
    ----
        team_id: FPL Team ID.
        gameweek: Gameweek.

    """
    url: str = f"{FPL_TEAM_URL}/{team_id}/transfers"
    response: requests.Response = requests.get(url)
    save_json(
        response.json(),
        MODEL_FOLDER
        / "predictions/player"
        / f"gameweek_{gameweek}"
        / "team_transfers.json",
    )
    logger.info("My transfers downloaded.")


def get_current_team(team_id: int, gameweek: int) -> None:
    """

    Args:
    ----
        team_id: FPL Team ID.
        gameweek: Gameweek.

    """
    url: str = f"{FPL_TEAM_URL}/{team_id}/event/{gameweek}/picks/"
    response: requests.Response = requests.get(url)
    save_json(
        response.json(),
        MODEL_FOLDER
        / "predictions/player"
        / f"gameweek_{gameweek}"
        / "team_last_gw.json",
    )
    logger.info("Last gameweek team downloaded.")


if __name__ == "__main__":
    get_my_transfers(85599, 4)
    get_current_team(85599, 4)
