import asyncio
import json
from pathlib import Path

import pandas as pd
import rich.progress
from loguru import logger

from fantasypl.config.constants.folder_config import DATA_FOLDER_FBREF, DATA_FOLDER_REF
from fantasypl.config.models.season import Season, Seasons
from fantasypl.config.models.team import Team
from fantasypl.utils.save_helper import save_pandas
from fantasypl.utils.web_helper import get_content, get_single_table


_tables: list[str] = [
    "stats_{}_summary",
    "stats_{}_passing",
    "stats_{}_defense",
    "stats_{}_misc",
    "keeper_stats_{}",
]

with Path.open(DATA_FOLDER_REF / "teams.json", "r") as f:
    _list_teams: list[Team] = [
        Team.model_validate(el) for el in json.load(f).get("teams")
    ]


def get_fpath(
    season: Season,
    team_input: str,
    date: str,
    tables: list[str],
    j: int,
) -> Path:
    try:
        team: Team = next(team for team in _list_teams if team.fbref_id == team_input)
    except StopIteration as err:
        logger.exception(f"{team_input} NOT FOUND!!")
        raise IndexError from err
    return (
        DATA_FOLDER_FBREF
        / season.folder
        / "matches"
        / team.short_name
        / f"{tables[j].replace(f"stats_{team_input}","").strip("_")}_{date}.csv"
    )


def get_matches(season: Season) -> None:
    logger.info("Downloading match data for season {}", season.fbref_name)
    df_links: pd.DataFrame = pd.read_csv(
        DATA_FOLDER_FBREF / season.folder / "match_links.csv",
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
            url: str = f"https://fbref.com/{match_link}"
            content: str = get_content(url=url)
            dfs_home: list[pd.DataFrame] = asyncio.run(
                get_single_table(content=content, tables=tables_home),
            )
            if not dfs_home:
                logger.error("Team {} Error on Match: {}", home_team, match_link)
            dfs_away: list[pd.DataFrame] = asyncio.run(
                get_single_table(content=content, tables=tables_away),
            )
            if not dfs_away:
                logger.error("Team {} Error on Match: {}", away_team, match_link)
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
