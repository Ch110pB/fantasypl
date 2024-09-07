"""Functions for getting FBRef team matchlogs."""

import asyncio
from typing import TYPE_CHECKING

import rich.progress
from loguru import logger

from fantasypl.config.constants import (
    DATA_FOLDER_FBREF,
    FBREF_BASE_URL,
)
from fantasypl.config.schemas import Season, Seasons
from fantasypl.utils import (
    get_content,
    get_list_teams,
    get_single_table,
    save_pandas,
)


if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd


_table_id_for: str = "matchlogs_for"
_table_id_against: str = "matchlogs_against"
_stat_tables: list[str] = [
    "schedule",
    "keeper",
    "shooting",
    "passing",
    "passing_types",
    "gca",
    "defense",
    "possession",
    "misc",
]


def get_matchlogs(
    season: Season, filter_teams: list[str] | None = None
) -> None:
    """

    Parameters
    ----------
    season
        The season under process.
    filter_teams
         The optional list of team short names.

    """
    list_teams = get_list_teams()
    if filter_teams is not None:
        list_teams = [
            team for team in list_teams if team.short_name in filter_teams
        ]
    with rich.progress.Progress() as progress:
        _task_id: rich.progress.TaskID = progress.add_task(
            "[cyan]Getting team matchlogs from FBRef: ",
            total=len(list_teams) * len(_stat_tables),
        )
        for team in list_teams:
            base_url: str = (
                f"{FBREF_BASE_URL}/squads/{team.fbref_id}/{season.fbref_long_name}/"
                f"matchlogs/c9/{{stat}}/"
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
                    )
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
    logger.info(
        "Team matchlogs fetch completed for Season: {}", season.fbref_name
    )


if __name__ == "__main__":
    # get_matchlogs(Seasons.SEASON_2324.value)
    get_matchlogs(Seasons.SEASON_2425.value)
