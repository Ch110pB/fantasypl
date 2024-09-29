"""Functions for getting latest gameweek team and transfers data."""

import requests
from loguru import logger

from fantasypl.config.constants import FPL_TEAM_URL, MODEL_FOLDER
from fantasypl.utils import save_json


def get_all_transfers(team_id: int, gameweek: int) -> None:
    """
    Get FPL squad transfers data.

    Parameters
    ----------
    team_id
        FPL team ID.
    gameweek
        The gameweek under process.

    """
    url: str = f"{FPL_TEAM_URL}/{team_id}/transfers/"
    response: requests.Response = requests.get(url, timeout=5)
    save_json(
        response.json(),
        MODEL_FOLDER
        / "predictions/player"
        / f"gameweek_{gameweek}"
        / "team_transfers.json",
    )
    logger.info("All transfers downloaded.")


def get_current_team(team_id: int, gameweek: int) -> None:
    """
    Get FPL current squad data.

    Parameters
    ----------
    team_id
        FPL team ID.
    gameweek
        The gameweek under process.

    """
    url: str = f"{FPL_TEAM_URL}/{team_id}/event/{gameweek}/picks/"
    response: requests.Response = requests.get(url, timeout=5)
    save_json(
        response.json(),
        MODEL_FOLDER
        / "predictions/player"
        / f"gameweek_{gameweek}"
        / "team_last_gw.json",
    )
    logger.info("Latest gameweek team downloaded.")


if __name__ == "__main__":
    get_all_transfers(85599, 6)
    get_current_team(85599, 5)
