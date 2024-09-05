"""Functions for getting FBRef team most used formation for a season."""

import statistics
from typing import TYPE_CHECKING

import pandas as pd
import rich.progress

from fantasypl.config.constants import (
    DATA_FOLDER_FBREF,
    FBREF_BASE_URL,
)
from fantasypl.config.schemas import Season, Seasons, Team
from fantasypl.utils import (
    extract_table,
    get_content,
    get_fbref_teams,
    get_list_teams,
    save_pandas,
)


if TYPE_CHECKING:
    from pathlib import Path


_promoted_teams: list[str] = ["Leicester City", "Southampton", "Ipswich Town"]


def get_formation_last_season(season: Season, last_season: Season) -> None:
    """

    Parameters
    ----------
    season
        The season under process.
    last_season
        The previous season.

    """
    _teams: list[str] = get_fbref_teams(season)
    dfs: list[dict[str, str]] = []
    for team_name in rich.progress.track(
        _teams, "Scraping last season league schedule of teams: "
    ):
        team: Team = next(
            el for el in get_list_teams() if el.fbref_name == team_name
        )
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
        DATA_FOLDER_FBREF
        / last_season.folder
        / "team_season"
        / "formations.csv"
    )
    save_pandas(df_formation, fpath)


if __name__ == "__main__":
    get_formation_last_season(
        Seasons.SEASON_2425.value, Seasons.SEASON_2324.value
    )
