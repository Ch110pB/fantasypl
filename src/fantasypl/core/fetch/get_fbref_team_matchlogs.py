"""Functions for getting FBRef team matchlogs."""

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

import rich.progress
from loguru import logger

from fantasypl.config.constants.folder_config import DATA_FOLDER_FBREF, DATA_FOLDER_REF
from fantasypl.config.constants.web_config import FBREF_BASE_URL
from fantasypl.config.models.season import Season, Seasons
from fantasypl.config.models.team import Team
from fantasypl.utils.save_helper import save_pandas
from fantasypl.utils.web_helper import get_content, get_single_table


if TYPE_CHECKING:
    import pandas as pd


_promoted_teams: list[str] = ["Ipswich Town", "Leicester City", "Southampton"]
_relegated_teams: list[str] = ["Luton Town", "Burnley", "Sheffield Utd"]
_table_id_for: str = "matchlogs_for"
_table_id_against: str = "matchlogs_against"
_stat_tables: list[str] = [
    "schedule",
    "keeper",
    "shooting",
    "passing",
    "gca",
    "defense",
    "misc",
]


def get_matchlogs(season: Season, filter_teams: list[str] | None = None) -> None:
    """

    Args:
    ----
        season: Season.
        filter_teams: Optional list of team short names to run on.

    """
    with Path.open(DATA_FOLDER_REF / "teams.json", "r") as f:
        list_teams: list[Team] = [
            Team.model_validate(el) for el in json.load(f).get("teams")
        ]
    if filter_teams is not None:
        list_teams = [team for team in list_teams if team.short_name in filter_teams]
    with rich.progress.Progress() as progress:
        _task_id: rich.progress.TaskID = progress.add_task(
            "[cyan]Getting team matchlogs from FBRef: ",
            total=len(list_teams) * len(_stat_tables),
        )
        for team in list_teams:
            if (
                team.fbref_name in _relegated_teams
                and season == Seasons.SEASON_2425.value
            ):
                continue

            base_url: str = (
                f"{FBREF_BASE_URL}/squads/{team.fbref_id}/{season.fbref_long_name}/"
                f"matchlogs/c9/{{stat}}/"
            )
            if (
                team.fbref_name in _promoted_teams
                and season == Seasons.SEASON_2324.value
            ):
                base_url = (
                    f"{FBREF_BASE_URL}/squads/{team.fbref_id}/{season.fbref_long_name}/"
                    f"matchlogs/c10/{{stat}}/"
                )
            for stat in _stat_tables:
                url: str = base_url.format(stat=stat)
                content: str = get_content(url)
                tables: list[str]
                match stat:
                    case "schedule":
                        tables = [_table_id_for]
                    case _:
                        tables = [_table_id_for, _table_id_against]
                dfs: list[pd.DataFrame] = asyncio.run(
                    get_single_table(
                        content=content,
                        tables=tables,
                        dropna_cols=["match_report"],
                    ),
                )
                for i, df in enumerate(dfs):
                    fpath: Path = (
                        DATA_FOLDER_FBREF
                        / season.folder
                        / "team_matchlogs"
                        / team.short_name
                        / f"{stat}_{tables[i].removeprefix("matchlogs_")}.csv"
                    )
                    if df.empty:
                        logger.error(
                            "Data fetch error from FBRef: "
                            "Season = {} Team = {} "
                            "Stat = {} Table = {}",
                            season.fbref_name,
                            team.short_name,
                            stat,
                            tables[i],
                        )
                    save_pandas(df, fpath)
                progress.update(task_id=_task_id, advance=1)
    logger.info("Team matchlogs fetch completed for Season: {}", season.fbref_name)


if __name__ == "__main__":
    # get_matchlogs(Seasons.SEASON_2324.value)
    get_matchlogs(Seasons.SEASON_2425.value)
