"""Functions for getting FBRef team most used formation for last season."""

import json
import statistics
from pathlib import Path

import pandas as pd
import rich.progress

from fantasypl.config.constants.folder_config import DATA_FOLDER_FBREF, DATA_FOLDER_REF
from fantasypl.config.constants.web_config import FBREF_BASE_URL
from fantasypl.config.models.season import Season, Seasons
from fantasypl.config.models.team import Team
from fantasypl.utils.modeling_helper import get_fbref_teams
from fantasypl.utils.save_helper import save_pandas
from fantasypl.utils.web_helper import extract_table, get_content


with Path.open(DATA_FOLDER_REF / "teams.json", "r") as f:
    _list_teams: list[Team] = [
        Team.model_validate(el) for el in json.load(f).get("teams")
    ]
_promoted_teams: list[str] = ["Leicester City", "Southampton", "Ipswich Town"]


def get_formation_last_season(season: Season, last_season: Season) -> None:
    """

    Args:
    ----
        season: Season.
        last_season: Last Season.

    """
    _teams: list[str] = get_fbref_teams(season)
    dfs: list[dict[str, str]] = []
    for team_name in rich.progress.track(
        _teams, "Scraping last season league schedule of teams: "
    ):
        team: Team = next(el for el in _list_teams if el.fbref_name == team_name)
        url: str = (
            f"{FBREF_BASE_URL}/squads/{team.fbref_id}/"
            f"{last_season.fbref_long_name}/matchlogs/c9/schedule/"
        )
        if team.fbref_name in _promoted_teams:
            url = (
                f"{FBREF_BASE_URL}/squads/{team.fbref_id}/"
                f"{last_season.fbref_long_name}/matchlogs/c10/schedule/"
            )
        table_id: str = "matchlogs_for"
        content: str = get_content(url)
        df_: pd.DataFrame = extract_table(content, table_id)
        formation: str = statistics.mode(df_["formation"].tolist())
        dfs.append({"team": team.fbref_name, "formation": formation})
    df_formation: pd.DataFrame = pd.DataFrame(dfs) if dfs else pd.DataFrame()
    fpath: Path = (
        DATA_FOLDER_FBREF / last_season.folder / "team_season" / "formations.csv"
    )
    save_pandas(df_formation, fpath)


if __name__ == "__main__":
    get_formation_last_season(Seasons.SEASON_2425.value, Seasons.SEASON_2324.value)
