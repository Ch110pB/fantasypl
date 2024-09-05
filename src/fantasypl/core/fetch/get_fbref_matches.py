"""Functions for getting FBRef match details"""

import asyncio
from pathlib import Path

import pandas as pd
import rich.progress
from loguru import logger

from fantasypl.config.constants import (
    DATA_FOLDER_FBREF,
    FBREF_BASE_URL,
)
from fantasypl.config.schemas import Season, Seasons, Team
from fantasypl.utils import (
    get_content,
    get_list_teams,
    get_single_table,
    save_pandas,
)


_tables: list[str] = [
    "stats_{}_summary",
    "stats_{}_passing",
    "stats_{}_defense",
    "stats_{}_misc",
    "keeper_stats_{}",
]


def get_fpath(
    season: Season, team_fbref_id: str, date: str, tables: list[str], j: int
) -> Path:
    """

    Parameters
    ----------
    season
        The season under process.
    team_fbref_id
        FBRef team ID.
    date
        The date of the match.
    tables
        The list of tables.
    j
        The iterator for finding table name.

    Returns
    -------
        Path for saving the file.

    Raises
    ------
    IndexError
        If team FBRef ID is not found.

    """
    try:
        team: Team = next(
            team for team in get_list_teams() if team.fbref_id == team_fbref_id
        )
    except StopIteration as err:
        logger.exception(f"{team_fbref_id} NOT FOUND!!")
        raise IndexError from err
    table_name: str = (
        tables[j].replace(f"stats_{team_fbref_id}", "").strip("_")
    )
    file_path: Path = (
        DATA_FOLDER_FBREF
        / season.folder
        / "matches"
        / team.short_name
        / f"{table_name}_{date}.csv"
    )
    return file_path


def get_matches(season: Season) -> None:
    """

    Parameters
    ----------
    season
        The season under progress.

    """
    logger.info("Downloading match data for season {}", season.fbref_name)
    df_links: pd.DataFrame = pd.read_csv(
        DATA_FOLDER_FBREF / season.folder / "match_links.csv"
    )

    with rich.progress.Progress() as progress:
        _task_id: rich.progress.TaskID = progress.add_task(
            "[cyan]Getting match_stats from FBRef: ",
            total=df_links.shape[0] * 2 * len(_tables),
        )
        for i in range(df_links.shape[0]):
            home_team: str = df_links["home_team"].to_numpy().item(i)
            away_team: str = df_links["away_team"].to_numpy().item(i)
            date: str = df_links["date"].to_numpy().item(i)
            match_link: str = df_links["match_link"].to_numpy().item(i)

            tables_home: list[str] = [
                table_idx.format(home_team) for table_idx in _tables
            ]
            tables_away: list[str] = [
                table_idx.format(away_team) for table_idx in _tables
            ]
            content: str = get_content(url=f"{FBREF_BASE_URL}/{match_link}")
            dfs_home: list[pd.DataFrame] = asyncio.run(
                get_single_table(content=content, tables=tables_home)
            )
            if not dfs_home:
                logger.error(
                    "Team {} Error on Match: {}", home_team, match_link
                )
            dfs_away: list[pd.DataFrame] = asyncio.run(
                get_single_table(content=content, tables=tables_away)
            )
            if not dfs_away:
                logger.error(
                    "Team {} Error on Match: {}", away_team, match_link
                )
            df: pd.DataFrame
            j: int
            fpath: Path
            for j, df in enumerate(dfs_home):
                df["team"] = home_team
                df["opponent"] = away_team
                df["date"] = date
                df["venue"] = "Home"
                fpath = get_fpath(season, home_team, date, tables_home, j)
                save_pandas(df=df, fpath=fpath)
                progress.update(task_id=_task_id, advance=1)
            for j, df in enumerate(dfs_away):
                df["team"] = away_team
                df["opponent"] = home_team
                df["date"] = date
                df["venue"] = "Away"
                fpath = get_fpath(season, away_team, date, tables_away, j)
                save_pandas(df=df, fpath=fpath)
                progress.update(task_id=_task_id, advance=1)


if __name__ == "__main__":
    get_matches(Seasons.SEASON_2425.value)
    # get_matches(Seasons.SEASON_2324.value)
