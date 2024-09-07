"""Functions for getting FBRef team stats for complete season."""

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

from fantasypl.config.constants import DATA_FOLDER_FBREF, FBREF_BASE_URL
from fantasypl.config.schemas import Season, Seasons
from fantasypl.utils import get_content, get_single_table, save_pandas


if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd


_tables: list[str] = [
    "stats_squads_standard_for",
    "stats_squads_keeper_for",
    "stats_squads_keeper_adv_for",
    "stats_squads_shooting_for",
    "stats_squads_passing_for",
    "stats_squads_passing_types_for",
    "stats_squads_gca_for",
    "stats_squads_defense_for",
    "stats_squads_possession_for",
    "stats_squads_misc_for",
    "stats_squads_standard_against",
    "stats_squads_keeper_against",
    "stats_squads_keeper_adv_against",
    "stats_squads_shooting_against",
    "stats_squads_passing_against",
    "stats_squads_passing_types_against",
    "stats_squads_gca_against",
    "stats_squads_defense_against",
    "stats_squads_possession_against",
    "stats_squads_misc_against",
]


def get_league_season(season: Season, league_id: int) -> None:
    """

    Parameters
    ----------
    season
        The season under process.
    league_id
        The FBRef League ID (9=PL, 10=Championship).

    """
    content: str = get_content(
        f"{FBREF_BASE_URL}/comps/{league_id}/{season.fbref_long_name}/"
    )
    dfs: list[pd.DataFrame] = asyncio.run(
        get_single_table(content=content, tables=_tables)
    )
    for j, df in enumerate(dfs):
        fpath: Path = (
            DATA_FOLDER_FBREF
            / season.folder
            / "team_season"
            / str(league_id)
            / f"{
                _tables[j].removeprefix("stats_squads_").removesuffix("_for")
            }.csv"
        )

        if df.empty:
            logger.error(
                "Data fetch error from FBRef: "
                "Season = {} League = {} Stat = {}",
                season.fbref_name,
                league_id,
                f"{
                    _tables[j]
                    .removeprefix("stats_squads_")
                    .removesuffix("_for")
                }",
            )
        save_pandas(df=df, fpath=fpath)
    logger.info(
        "Data fetch complete for season: {} and league: {}",
        season.fbref_long_name,
        league_id,
    )


if __name__ == "__main__":
    get_league_season(Seasons.SEASON_2324.value, 9)
    get_league_season(Seasons.SEASON_2324.value, 10)
