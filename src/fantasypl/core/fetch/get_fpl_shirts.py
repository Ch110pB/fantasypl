"""Download shirt graphics from FPL."""

import pandas as pd
import requests
import rich.progress

from fantasypl.config.constants.folder_config import DATA_FOLDER_FPL, RESOURCE_FOLDER
from fantasypl.config.constants.web_config import FPL_SHIRTS_URL
from fantasypl.config.models.season import Season, Seasons
from fantasypl.utils.save_helper import save_requests_response


def get_shirts(season: Season) -> None:
    """

    Args:
    ----
        season: Season.

    """
    df_fpl_teams = pd.read_csv(DATA_FOLDER_FPL / season.folder / "teams.csv")
    team_codes = df_fpl_teams["code"].tolist()

    for code in rich.progress.track(team_codes, "Downloading shirt graphics: "):
        url: str = f"{FPL_SHIRTS_URL}/shirt_{code}-220.png"
        response: requests.Response = requests.get(url)
        save_requests_response(
            response, RESOURCE_FOLDER / season.folder / f"shirt_{code}.png"
        )


if __name__ == "__main__":
    get_shirts(Seasons.SEASON_2425.value)
