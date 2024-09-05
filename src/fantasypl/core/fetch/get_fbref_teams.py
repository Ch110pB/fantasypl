"""Functions for getting FBRef teams."""

from typing import TYPE_CHECKING

from loguru import logger

from fantasypl.config.constants import DATA_FOLDER_FBREF, FBREF_BASE_URL
from fantasypl.config.schemas import Season, Seasons
from fantasypl.utils import extract_table, get_content, save_pandas


if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd


def get_teams(season: Season) -> None:
    """

    Parameters
    ----------
    season
        The season under process.

    """
    url: str = f"{FBREF_BASE_URL}/comps/9/{season.fbref_long_name}/"
    content: str = get_content(url=url, delay=0)
    table_id: str = f"results{season.fbref_long_name}91_overall"
    df_teams: pd.DataFrame = extract_table(
        content=content, table_id=table_id, href=True
    )
    df_teams["fbref_id"] = (
        df_teams["team"].str[1].str.strip().str.split("/").str[3]
    )
    df_teams["name"] = df_teams["team"].str[0].str.strip()
    df_teams = df_teams[["fbref_id", "name"]]
    fpath: Path = DATA_FOLDER_FBREF / season.folder / "teams.csv"
    save_pandas(df_teams, fpath)
    logger.info("FBRef Teams data saved for season {}", season.fbref_name)


if __name__ == "__main__":
    # get_teams(Seasons.SEASON_2324.value)
    get_teams(Seasons.SEASON_2425.value)
